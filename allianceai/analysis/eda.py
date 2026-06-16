"""
Exploratory Data Analysis (EDA) for financial statements.

Functions here produce summary statistics, detect outliers, assess
data completeness, and generate trend signals — the building blocks
that every downstream model and narrative relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Completeness & quality
# ------------------------------------------------------------------

def assess_data_quality(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Return a per-column quality report: count, null%, mean, std, skew.
    Logs a warning for any column where more than 30 % of values are NaN.
    """
    report = pd.DataFrame({
        "count":    df.count(),
        "null_pct": df.isnull().mean().round(4),
        "mean":     df.mean(numeric_only=True),
        "std":      df.std(numeric_only=True),
        "skew":     df.skew(numeric_only=True),
    })
    heavy_missing = report[report["null_pct"] > 0.30].index.tolist()
    if heavy_missing:
        logger.warning(
            "[%s] Heavy missingness (>30%%) in columns: %s", label, heavy_missing
        )
    return report


def describe_statement(df: pd.DataFrame) -> pd.DataFrame:
    """Extended describe() — adds median, IQR, and coefficient of variation."""
    if df.empty:
        return pd.DataFrame()

    desc = df.describe().T
    desc["median"] = df.median()
    desc["iqr"] = df.quantile(0.75) - df.quantile(0.25)

    mean = desc["mean"].replace(0, np.nan)
    desc["cv"] = desc["std"] / mean.abs()  # coefficient of variation

    return desc


# ------------------------------------------------------------------
# Trend detection
# ------------------------------------------------------------------

def linear_trend(series: pd.Series) -> dict:
    """
    Fit a simple OLS trend on *series* (time as integer index).
    Returns slope, r², p-value, and a human-readable direction label.

    Useful for quickly answering 'is revenue growing/shrinking systematically?'
    """
    clean = series.dropna()
    if len(clean) < 3:
        logger.debug("Not enough data points for trend analysis on '%s'.", series.name)
        return {"slope": np.nan, "r_squared": np.nan, "p_value": np.nan, "direction": "unknown"}

    x = np.arange(len(clean), dtype=float)
    slope, intercept, r, p, se = stats.linregress(x, clean.values.astype(float))

    direction = "flat"
    if p < 0.05:
        direction = "upward" if slope > 0 else "downward"

    return {
        "slope":     slope,
        "r_squared": r ** 2,
        "p_value":   p,
        "direction": direction,
    }


def trend_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Run linear_trend() across all (or selected) numeric columns."""
    cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    records = []
    for col in cols:
        if col not in df.columns:
            continue
        t = linear_trend(df[col])
        t["metric"] = col
        records.append(t)
    return pd.DataFrame(records).set_index("metric") if records else pd.DataFrame()


# ------------------------------------------------------------------
# Outlier detection
# ------------------------------------------------------------------

def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    """
    Flag values whose |z-score| exceeds *threshold*.
    Returns a boolean DataFrame of the same shape (True = outlier).

    Z-score outliers in financial data often signal one-time charges,
    write-downs, or accounting restatements — worth investigating manually.
    """
    numeric = df.select_dtypes(include=np.number)
    z = numeric.apply(stats.zscore, nan_policy="omit")
    flags = z.abs() > threshold

    n_flags = flags.sum().sum()
    if n_flags:
        logger.info("Detected %d outlier value(s) by z-score (threshold=%.1f).", n_flags, threshold)
    return flags


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Classic IQR fence method for a single series.
    Returns a boolean mask (True = outlier).
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)


# ------------------------------------------------------------------
# Growth rates
# ------------------------------------------------------------------

def period_over_period_growth(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Quarter-over-quarter (or year-over-year) percentage change for each column.
    Negative base values are replaced with NaN to avoid misleading sign flips.
    """
    cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    result = pd.DataFrame(index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].copy().astype(float)
        # Avoid nonsensical growth rates when the prior period was negative.
        prev = s.shift(1)
        growth = s.pct_change()
        growth[prev <= 0] = np.nan
        result[f"{col}_growth"] = growth
    return result


# ------------------------------------------------------------------
# Correlation
# ------------------------------------------------------------------

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation among numeric columns; NaN-safe."""
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr(method="pearson", min_periods=4)


# ------------------------------------------------------------------
# Full EDA pipeline
# ------------------------------------------------------------------

def run_full_eda(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> dict[str, object]:
    """
    Run the complete EDA pipeline over all three statements.
    Returns a dict of labelled DataFrames and summary objects for
    consumption by the report generator or narrative model.
    """
    logger.info("Running full EDA pipeline.")
    results: dict[str, object] = {}

    for label, df in [("income", income), ("balance", balance), ("cashflow", cashflow)]:
        if df.empty:
            logger.warning("Skipping EDA for '%s' — empty DataFrame.", label)
            continue
        results[f"{label}_quality"] = assess_data_quality(df, label)
        results[f"{label}_describe"] = describe_statement(df)
        results[f"{label}_trends"] = trend_summary(df)
        results[f"{label}_outliers"] = detect_outliers_zscore(df)
        results[f"{label}_growth"] = period_over_period_growth(df)

    logger.info("EDA pipeline complete.")
    return results
