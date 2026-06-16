"""
Prediction feedback loop.

Every run, the pipeline:
  1. EVALUATE — grades previously stored predictions against the official data
     just fetched (yfinance / SEC EDGAR). Errors land in DuckDB and become the
     training signal.
  2. CALIBRATE — applies a bias correction to today's new forecasts, learned
     from that ticker's own historical forecast errors (median signed % error
     per metric). The model literally learns from its last outputs.
  3. SEED — stores today's forecasts so the *next* run can grade them.

This is intentionally a transparent statistical correction rather than an
opaque retrain: every adjustment is logged and the raw value is preserved
alongside the calibrated one.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from allianceai.core.logging_config import get_logger
from allianceai.core.storage import storage

logger = get_logger(__name__)

# Don't grade a prediction against a statement date more than this far away.
_MATCH_TOLERANCE = pd.Timedelta("20D")
# Minimum graded samples before we trust a bias estimate.
_MIN_SAMPLES = 2
# Don't correct by more than this (a wildly wrong history shouldn't wreck a forecast).
_MAX_BIAS = 0.30
# Drop graded samples whose error is beyond this before estimating bias —
# protects calibration from degenerate backtest points (early micro-base
# periods, mixed annual/quarterly history, seasonal blowups).
_OUTLIER_ERROR = 0.60


# ------------------------------------------------------------------
# 1. Evaluate stored predictions against fresh official data
# ------------------------------------------------------------------

def evaluate_predictions(
    ticker: str,
    statements: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Grade pending predictions whose target period now has official data.
    *statements* maps 'income'/'balance'/'cashflow' to the freshly fetched
    DataFrames. Returns the newly graded rows (empty frame if none).
    """
    pending = storage.pending_predictions(ticker)
    if pending.empty:
        return pd.DataFrame()

    graded: list[dict] = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    for _, row in pending.iterrows():
        df = statements.get(row["statement"])
        if df is None or df.empty or row["metric"] not in df.columns:
            continue
        actuals = df[row["metric"]].dropna()
        if actuals.empty:
            continue
        target = pd.Timestamp(row["target_date"])
        # Find the closest official statement date to the predicted period.
        deltas = pd.Series(abs(actuals.index - target), index=actuals.index)
        nearest = deltas.idxmin()
        if deltas[nearest] > _MATCH_TOLERANCE:
            continue  # period not reported yet — keep as pending seed
        actual = float(actuals.loc[nearest])
        yhat = row["yhat"]
        pct_error = (yhat - actual) / abs(actual) if actual else np.nan
        graded.append({
            "ticker": ticker,
            "statement": row["statement"],
            "metric": row["metric"],
            "method": row.get("method"),
            "made_at": row.get("made_at"),
            "target_date": row["target_date"],
            "yhat": yhat,
            "actual": actual,
            "pct_error": float(pct_error) if pd.notna(pct_error) else None,
            "evaluated_at": now,
        })

    if graded:
        storage.record_outcomes(graded)
    return pd.DataFrame(graded)


# ------------------------------------------------------------------
# 2. Calibrate new forecasts from the ticker's own error history
# ------------------------------------------------------------------

def calibrate_forecasts(ticker: str, forecasts: dict) -> dict:
    """
    Adjust each metric's forecast by the median signed % error from this
    ticker's graded history. yhat_raw preserves the uncalibrated value;
    the correction applied is recorded under 'calibration_bias_pct'.
    """
    history = storage.outcome_history(ticker)
    if history.empty:
        return forecasts

    for stmt, metrics in forecasts.items():
        for metric, res in metrics.items():
            if not isinstance(res, dict):
                continue
            fc = res.get("forecast")
            if fc is None or fc.empty:
                continue
            h = history[(history["statement"] == stmt) & (history["metric"] == metric)]
            h = h.dropna(subset=["pct_error"])
            # Discard degenerate samples so a few garbage backtest points
            # (e.g. 800% errors off a tiny early base) can't poison the median.
            clean = h[h["pct_error"].abs() <= _OUTLIER_ERROR]
            if len(clean) < _MIN_SAMPLES:
                continue
            bias = float(np.clip(clean["pct_error"].median(), -_MAX_BIAS, _MAX_BIAS))
            if abs(bias) < 0.02:
                continue  # already well-calibrated
            for col in ("yhat", "yhat_lower", "yhat_upper"):
                if col in fc.columns:
                    fc[f"{col}_raw"] = fc[col]
                    fc[col] = fc[col] / (1 + bias)
            res["calibration_bias_pct"] = round(bias * 100, 2)
            res["calibration_samples"] = int(len(h))
            logger.info(
                "Calibrated %s.%s by %+.1f%% (median bias over %d graded predictions).",
                stmt, metric, -bias * 100, len(h),
            )
    return forecasts


# ------------------------------------------------------------------
# 3. Seed: store today's forecasts for the next run to grade
# ------------------------------------------------------------------

def save_forecast_seeds(ticker: str, forecasts: dict) -> int:
    """Persist every forecasted (metric, future period) as a prediction seed."""
    rows: list[dict] = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for stmt, metrics in (forecasts or {}).items():
        for metric, res in metrics.items():
            if not isinstance(res, dict):
                continue
            fc = res.get("forecast")
            if fc is None or fc.empty:
                continue
            for _, r in fc.iterrows():
                # Forecast frames carry the period date in the 'ds' column
                # (RangeIndex), matching Prophet's convention.
                ts = r["ds"] if "ds" in fc.columns else r.name
                rows.append({
                    "ticker": ticker,
                    "statement": stmt,
                    "metric": metric,
                    "method": res.get("method"),
                    "made_at": now,
                    "target_date": pd.Timestamp(ts).date(),
                    "yhat": float(r["yhat"]) if pd.notna(r.get("yhat")) else None,
                    "yhat_lower": float(r["yhat_lower"]) if pd.notna(r.get("yhat_lower")) else None,
                    "yhat_upper": float(r["yhat_upper"]) if pd.notna(r.get("yhat_upper")) else None,
                })
    if rows:
        storage.save_predictions(rows)
    return len(rows)


# ------------------------------------------------------------------
# Reporting helper: real vs predicted comparison table
# ------------------------------------------------------------------

def accuracy_table(ticker: str) -> pd.DataFrame:
    """All graded predictions for *ticker*, formatted for the report."""
    h = storage.outcome_history(ticker)
    if h.empty:
        return h
    out = h[["statement", "metric", "method", "target_date",
             "yhat", "actual", "pct_error"]].copy()
    out["abs_pct_error"] = out["pct_error"].abs()
    out = out.sort_values(["target_date", "statement", "metric"])
    out.columns = ["Statement", "Metric", "Method", "Period",
                   "Predicted", "Actual (official)", "Error", "Abs Error"]
    return out
