"""
Walk-forward backtester — synthetically generate a graded prediction history.

Instead of waiting quarters for live forecasts to mature, this rewinds the
statements and replays the forecaster through time:

    train on periods [0 .. t]  ->  forecast period t+1  ->  grade vs reality
    step forward, repeat.

Each graded result is written to the SAME `prediction_outcomes` table the live
loop uses, so the bias-calibration is trained from day one.

MODULAR by design — the same engine serves two very different needs:
  - "Greater purposes" / deep training: large `min_train`, `step=1`, every
    metric, across many tickers → hundreds of graded points.
  - Quick recent-only calibration: small window via `max_steps`, `step=2+`,
    one metric → finishes in well under a second.

Speed: defaults to the Holt-Winters fast path (no Prophet refit per step), so
a 10-year, single-ticker backtest runs in seconds and makes zero API calls.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from allianceai.core.logging_config import get_logger
from allianceai.core.storage import storage
from allianceai.models.forecaster import DEFAULT_FORECAST_METRICS, forecast_metric

logger = get_logger(__name__)


def backtest_metric(
    ticker: str,
    statement: str,
    series: pd.Series,
    *,
    min_train: int = 8,
    step: int = 1,
    max_steps: int | None = None,
    horizon: int = 1,
    method: str = "holt_winters",
) -> list[dict]:
    """
    Walk-forward backtest a single metric series. Returns graded outcome rows.

    min_train : minimum history before the first forecast (e.g. 8 = 2 years).
    step      : quarters to advance each iteration (1 = every quarter; 2+ = sparser/faster).
    max_steps : cap the number of forecasts (None = walk the whole series).
    horizon   : how many periods ahead to predict and grade (1 = next quarter).
    method    : "holt_winters" (fast) or None (let the forecaster try Prophet).
    """
    s = series.dropna().astype(float).sort_index()
    if len(s) < min_train + horizon:
        return []

    # Base-effect guard: skip target periods whose value is a tiny fraction of
    # the metric's mature scale (early micro-cap quarters produce meaningless
    # percentage errors — a $2B miss on a $4B base is +50%, on $50B is +4%).
    scale = float(s.tail(8).median()) if len(s) >= 8 else float(s.median())
    base_floor = abs(scale) * 0.15

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows: list[dict] = []
    cuts = range(min_train, len(s) - horizon + 1, step)
    if max_steps is not None:
        cuts = list(cuts)[-max_steps:]  # keep the most RECENT windows

    for cut in cuts:
        train = s.iloc[:cut]
        target_idx = cut + horizon - 1
        actual = float(s.iloc[target_idx])
        if abs(actual) < base_floor:
            continue  # immature period — error % would be dominated by base effect
        target_date = pd.Timestamp(s.index[target_idx]).date()
        try:
            res = forecast_metric(train, horizon=horizon, label=f"{ticker}.{series.name}",
                                  force_method=method)
            fc = res.get("forecast")
            if fc is None or fc.empty:
                continue
            yhat = float(fc["yhat"].iloc[horizon - 1])
        except Exception:
            continue
        pct_error = (yhat - actual) / abs(actual) if actual else None
        rows.append({
            "ticker": ticker,
            "statement": statement,
            "metric": str(series.name),
            "method": f"backtest_{res.get('method', method)}",
            "made_at": now,
            "target_date": target_date,
            "yhat": yhat,
            "actual": actual,
            "pct_error": float(pct_error) if pct_error is not None else None,
            "evaluated_at": now,
        })
    return rows


def backtest_ticker(
    ticker: str,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    *,
    metrics: dict[str, list[str]] | None = None,
    min_train: int = 8,
    step: int = 1,
    max_steps: int | None = None,
    horizon: int = 1,
    method: str = "holt_winters",
    persist: bool = True,
) -> pd.DataFrame:
    """
    Backtest all default metrics for one ticker and (optionally) persist the
    graded outcomes into DuckDB so calibration can use them immediately.
    Returns the graded rows as a DataFrame.
    """
    metrics = metrics or DEFAULT_FORECAST_METRICS
    frames = {"income": income, "balance": balance, "cashflow": cashflow}
    all_rows: list[dict] = []

    for stmt, cols in metrics.items():
        df = frames.get(stmt)
        if df is None or df.empty:
            continue
        for col in cols:
            if col not in df.columns:
                continue
            all_rows.extend(backtest_metric(
                ticker, stmt, df[col],
                min_train=min_train, step=step, max_steps=max_steps,
                horizon=horizon, method=method,
            ))

    if all_rows and persist:
        storage.record_outcomes(all_rows)
    logger.info("Backtest for '%s': %d graded prediction(s) generated.", ticker, len(all_rows))

    out = pd.DataFrame(all_rows)
    if not out.empty:
        out["abs_pct_error"] = out["pct_error"].abs()
    return out


def backtest_summary(graded: pd.DataFrame) -> pd.DataFrame:
    """Per-metric accuracy summary: count, median bias, mean absolute error."""
    if graded is None or graded.empty:
        return pd.DataFrame()
    g = graded.dropna(subset=["pct_error"])

    def _clean_median(x):
        c = x[x.abs() <= 0.60]            # same filter calibration applies
        return round(c.median() * 100, 1) if len(c) >= 2 else round(x.median() * 100, 1)

    def _reliability(x):
        # Fraction of samples within ±25% — how trustworthy this metric's
        # calibration signal is (low = mixed-frequency / seasonal / base-effect noise).
        frac = float((x.abs() <= 0.25).mean())
        return "good" if frac >= 0.6 else "fair" if frac >= 0.35 else "noisy"

    summary = g.groupby(["statement", "metric"]).agg(
        samples=("pct_error", "size"),
        calibration_bias_pct=("pct_error", _clean_median),
        mean_abs_error_pct=("abs_pct_error", lambda x: round(x.mean() * 100, 1)),
        reliability=("pct_error", _reliability),
    ).reset_index()
    return summary.sort_values("mean_abs_error_pct")
