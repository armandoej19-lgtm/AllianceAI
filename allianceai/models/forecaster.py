"""
Time-series forecasting with full statistical output.

Every forecast includes:
  - Point estimate (yhat)
  - 80% confidence interval (yhat_lower, yhat_upper)
  - Trend slope and its p-value (is the trend statistically real?)
  - R² of the trend fit
  - MAPE on the in-sample fit (forecast accuracy estimate)

Primary model: Prophet (Meta, open-source).
Fallback: Holt-Winters exponential smoothing for sparse data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from allianceai.core.config import settings
from allianceai.core.exceptions import InsufficientDataError, ModelError
from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)

_PROPHET_MIN_ROWS = 6   # Prophet needs at least this many data points.
_HW_MIN_ROWS      = 3   # Holt-Winters minimum.
_TREND_WINDOW     = 20  # Periods used for the descriptive trend fit (~5y quarterly).


# ------------------------------------------------------------------
# Prophet forecaster
# ------------------------------------------------------------------

def _forecast_prophet(series: pd.Series, horizon: int) -> pd.DataFrame:
    """
    Fit a Prophet model and return a DataFrame with columns:
        ds (date), yhat (forecast), yhat_lower, yhat_upper.
    """
    # Import here so projects that don't install prophet still work for
    # everything except forecasting.
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ModelError("Prophet is not installed.  Run: pip install prophet") from exc

    df_prophet = pd.DataFrame({
        "ds": series.index,
        "y":  series.values.astype(float),
    }).dropna()

    if len(df_prophet) < _PROPHET_MIN_ROWS:
        raise InsufficientDataError(
            f"Only {len(df_prophet)} non-NaN data points — need at least {_PROPHET_MIN_ROWS}."
        )

    # Silence Prophet's verbose stdout/stderr inside the library.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(
            yearly_seasonality=False,  # Financial quarters don't have intra-year seasonality.
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            interval_width=0.80,
        )
        model.fit(df_prophet)

    # Build future dates: assume quarterly frequency by default.
    freq = pd.infer_freq(series.index) or "QS"
    future = model.make_future_dataframe(periods=horizon, freq=freq)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).reset_index(drop=True)


# ------------------------------------------------------------------
# Holt-Winters fallback
# ------------------------------------------------------------------

def _forecast_holt_winters(series: pd.Series, horizon: int) -> pd.DataFrame:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    clean = series.dropna().astype(float)
    if len(clean) < _HW_MIN_ROWS:
        raise InsufficientDataError(
            f"Only {len(clean)} data points — cannot fit Holt-Winters."
        )

    model = ExponentialSmoothing(clean, trend="add", seasonal=None).fit(optimized=True)
    yhat = model.forecast(horizon)

    freq = pd.infer_freq(series.index) or "QS"
    last_date = series.index[-1]
    future_index = pd.date_range(last_date, periods=horizon + 1, freq=freq)[1:]

    return pd.DataFrame({
        "ds":         future_index,
        "yhat":       yhat.values,
        "yhat_lower": yhat.values * 0.85,  # ±15% naive confidence band.
        "yhat_upper": yhat.values * 1.15,
    })


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def _trend_stats(series: pd.Series) -> dict:
    """
    OLS trend fit: slope, p-value, R², and in-sample fit error.

    Two robustness measures keep the numbers honest for the kinds of series
    financial data actually contains:

    1. **Log-linear fit for exponential growth.**  A strictly-positive series
       spanning a wide dynamic range (e.g. a hyper-growth company's revenue
       going from $0.7B to $82B) is fit in LOG space.  A single straight line
       through ~18y of exponential data fits the tiny early values terribly and
       produces a fit error in the hundreds of percent that says nothing about
       forecast quality.  A log-linear fit models compound growth instead, so
       R²/slope%/error all describe the trend that's really there.

    2. **WAPE instead of raw MAPE.**  Mean absolute *percentage* error divides
       each residual by that period's actual value, so a single near-zero or
       sign-flipping period (common in net income) sends it to thousands of
       percent.  We report WAPE = Σ|actual−fitted| / Σ|actual|, an aggregate
       ratio that stays bounded and interpretable across the whole series.
    """
    from scipy import stats as sp_stats
    s = series.dropna().astype(float)
    if len(s) < 3:
        return {"slope": np.nan, "slope_pct_per_period": np.nan,
                "r_squared": np.nan, "p_value": np.nan, "mape_pct": np.nan,
                "trend_model": "none", "window": 0}

    # Fit the trend over the most recent window, not the entire history.  A
    # company's near-term trajectory is what a forecast cares about; dragging in
    # 15-year-old data (when revenue was 100× smaller) bends a single trend line
    # through two utterly different regimes and inflates the fit error for no
    # forecasting benefit.  We cap at ~5 years of quarters.
    window = min(len(s), _TREND_WINDOW)
    s = s.iloc[-window:]
    x = np.arange(len(s), dtype=float)
    vals = s.values

    def _linear() -> tuple:
        slope, intercept, r, p, _ = sp_stats.linregress(x, vals)
        fitted = intercept + slope * x
        mean = vals.mean()
        pct = float(slope / abs(mean) * 100.0) if mean != 0 else np.nan
        return "linear", float(slope), pct, float(r) ** 2, float(p), fitted

    def _log_linear() -> tuple:
        slope_l, intercept_l, r, p, _ = sp_stats.linregress(x, np.log(vals))
        fitted = np.exp(intercept_l + slope_l * x)
        # Absolute $/period = the fitted curve's most recent step (momentum);
        # %/period = the constant compound growth rate.
        slope_abs = float(fitted[-1] - fitted[-2]) if len(fitted) > 1 else np.nan
        return "log-linear", slope_abs, float(np.expm1(slope_l) * 100.0), float(r) ** 2, float(p), fitted

    # Candidate fits: always linear; add log-linear when the series is strictly
    # positive (a log fit is undefined otherwise).  Pick whichever reproduces
    # the history with the lower WAPE — data decides linear vs exponential.
    candidates = [_linear()]
    if bool(np.all(vals > 0)):
        candidates.append(_log_linear())

    def _wape(fitted: np.ndarray) -> float:
        denom = float(np.sum(np.abs(vals)))
        return float(np.sum(np.abs(vals - fitted)) / denom * 100.0) if denom > 0 else np.inf

    model, slope_abs, slope_pct, r2, p, fitted = min(candidates, key=lambda c: _wape(c[5]))
    return {
        "slope":                 round(float(slope_abs), 4) if slope_abs == slope_abs else np.nan,
        "slope_pct_per_period":  round(slope_pct, 2) if slope_pct == slope_pct else np.nan,
        "r_squared":             round(r2, 4),
        "p_value":               round(p, 4),
        "mape_pct":              round(_wape(fitted), 2),
        "trend_model":           model,
        "window":                int(window),
    }


# Default metrics forecast across the three statements — shared so the
# backtester and the live pipeline stay in sync.
DEFAULT_FORECAST_METRICS = {
    "income":   ["Total Revenue", "Net Income", "EBITDA", "Operating Income"],
    "balance":  ["Total Assets", "Total Debt", "Stockholders Equity"],
    "cashflow": ["Operating Cash Flow", "Free Cash Flow"],
}


def forecast_metric(
    series: pd.Series,
    horizon: int | None = None,
    label: str = "",
    force_method: str | None = None,
) -> dict:
    """
    Forecast *series* for *horizon* future periods.

    *force_method* = "holt_winters" skips Prophet entirely (fast path, used by
    the backtester where speed matters more than the most elegant fit).

    Returns a dict with:
      forecast  — DataFrame: ds, yhat, yhat_lower, yhat_upper
      stats     — trend slope, p-value, R², MAPE
      method    — 'prophet' or 'holt_winters'
    """
    h = horizon or settings.forecast_horizon_quarters
    label = label or (series.name or "metric")
    logger.info("Forecasting '%s' for %d periods.", label, h)

    stats = _trend_stats(series)
    method = "none"
    forecast_df = pd.DataFrame()

    if force_method != "holt_winters":
        try:
            forecast_df = _forecast_prophet(series, h)
            method = "prophet"
            logger.info("  Prophet forecast complete for '%s'.", label)
        except InsufficientDataError as e:
            logger.warning("  Prophet skipped for '%s': %s  Trying Holt-Winters.", label, e)
        except Exception as e:
            logger.warning("  Prophet failed for '%s': %s  Trying Holt-Winters.", label, e)

    if forecast_df.empty:
        try:
            forecast_df = _forecast_holt_winters(series, h)
            method = "holt_winters"
            logger.info("  Holt-Winters forecast complete for '%s'.", label)
        except Exception as e:
            logger.error("Both forecasters failed for '%s': %s", label, e)

    return {"forecast": forecast_df, "stats": stats, "method": method, "label": label}


def forecast_statements(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    metrics: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Forecast a selected set of metrics across all three statements.

    *metrics* maps statement name to list of column names.  Defaults to
    the most analytically significant ones if not provided.
    """
    if metrics is None:
        metrics = {
            "income":   ["Total Revenue", "Net Income", "EBITDA", "Operating Income"],
            "balance":  ["Total Assets", "Total Debt", "Stockholders Equity"],
            "cashflow": ["Operating Cash Flow", "Free Cash Flow"],
        }

    frames = {"income": income, "balance": balance, "cashflow": cashflow}
    results: dict[str, dict[str, pd.DataFrame]] = {}

    for stmt, cols in metrics.items():
        df = frames.get(stmt, pd.DataFrame())
        results[stmt] = {}
        for col in cols:
            if col not in df.columns or df[col].dropna().empty:
                logger.warning("Skipping forecast for '%s.%s' — no data.", stmt, col)
                continue
            try:
                results[stmt][col] = forecast_metric(df[col], label=f"{stmt}.{col}")
            except ModelError as e:
                logger.error("Forecast failed for '%s.%s': %s", stmt, col, e)

    return results
