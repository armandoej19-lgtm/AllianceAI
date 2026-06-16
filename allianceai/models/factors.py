"""
Factor analysis and statistical decomposition.

Answers the questions a quantitative analyst actually asks:
  - Which ratio is most correlated with future margin?
  - Is revenue growth statistically significant or noise?
  - How much of net income change is explained by revenue vs cost?
  - What is the leading indicator of cash flow stress?

All outputs are numerical: correlation coefficients, p-values,
R², regression slopes, explained variance percentages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Correlation with future performance
# ------------------------------------------------------------------

def predictive_correlations(
    ratios: dict[str, pd.DataFrame],
    income: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
    """
    For every ratio, compute its Pearson correlation with net margin
    *lag* periods ahead.  Includes p-value and confidence interval.

    This tells you which current ratio is the best leading indicator
    of future profitability.
    """
    logger.info("Computing predictive correlations (lag=%d periods).", lag)

    rev = income.get("Total Revenue", pd.Series(dtype=float))
    ni  = income.get("Net Income",    pd.Series(dtype=float))
    rev_safe = rev.replace(0, np.nan)
    future_margin = (ni / rev_safe).shift(-lag)

    records = []
    for group_name, df in ratios.items():
        for col in df.select_dtypes(include=np.number).columns:
            series = df[col].reindex(future_margin.index, method="nearest", tolerance=pd.Timedelta("95D"))
            combined = pd.concat([series, future_margin], axis=1).dropna()
            if len(combined) < 4:
                continue
            r, p = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
            n = len(combined)
            # 95% confidence interval via Fisher z-transformation.
            z  = np.arctanh(r)
            se = 1 / np.sqrt(n - 3)
            ci_lo = np.tanh(z - 1.96 * se)
            ci_hi = np.tanh(z + 1.96 * se)
            records.append({
                "group":       group_name,
                "ratio":       col,
                "correlation": round(r, 4),
                "p_value":     round(p, 4),
                "significant": p < 0.05,
                "ci_95_low":   round(ci_lo, 4),
                "ci_95_high":  round(ci_hi, 4),
                "n_periods":   n,
            })

    if not records:
        return pd.DataFrame()

    df_out = pd.DataFrame(records).sort_values("correlation", key=abs, ascending=False)
    top = df_out[df_out["significant"]].head(3)
    if not top.empty:
        logger.info(
            "Top predictive ratios: %s",
            ", ".join(f"{r['ratio']}(r={r['correlation']:.2f})" for _, r in top.iterrows()),
        )
    return df_out.reset_index(drop=True)


# ------------------------------------------------------------------
# Variance decomposition: revenue vs cost contribution to income change
# ------------------------------------------------------------------

def decompose_income_change(income: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose the change in net income into:
      - Revenue contribution  (ΔRevenue × prior margin)
      - Margin contribution   (ΔMargin × current revenue)
      - Residual

    Returns percentage attribution for each period so you can answer:
    'Was the profit improvement driven by selling more or cutting costs?'
    """
    logger.info("Decomposing income change into revenue vs margin contributions.")

    rev = income.get("Total Revenue", pd.Series(dtype=float))
    ni  = income.get("Net Income",    pd.Series(dtype=float))
    rev_s = rev.replace(0, np.nan)

    margin = ni / rev_s
    d_rev    = rev.diff()
    d_margin = margin.diff()

    rev_contrib    = d_rev    * margin.shift(1)
    margin_contrib = d_margin * rev
    total_change   = rev_contrib + margin_contrib
    total_safe     = total_change.replace(0, np.nan)

    out = pd.DataFrame({
        "net_income_change":        ni.diff().round(0),
        "revenue_contribution":     rev_contrib.round(0),
        "margin_contribution":      margin_contrib.round(0),
        "revenue_contrib_pct":      (rev_contrib    / total_safe * 100).round(1),
        "margin_contrib_pct":       (margin_contrib / total_safe * 100).round(1),
    }, index=income.index)

    return out


# ------------------------------------------------------------------
# Trend significance
# ------------------------------------------------------------------

def trend_statistics(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    For each column, fit OLS and return:
      slope, slope_per_period_pct, r_squared, p_value, t_statistic,
      direction (RISING / FALLING / FLAT), is_significant (p < 0.05)

    This is the mathematical answer to 'is this trend real or noise?'
    """
    cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    records = []

    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(float)
        if len(s) < 3:
            continue
        x = np.arange(len(s), dtype=float)
        slope, intercept, r, p, se = stats.linregress(x, s.values)
        mean = s.mean()
        slope_pct = (slope / abs(mean) * 100) if mean != 0 else np.nan

        direction = "FLAT"
        if p < 0.05:
            direction = "RISING" if slope > 0 else "FALLING"

        records.append({
            "metric":             col,
            "slope":              round(slope, 4),
            "slope_pct_per_period": round(slope_pct, 2),
            "r_squared":          round(r ** 2, 4),
            "p_value":            round(p, 4),
            "t_statistic":        round(slope / se, 4) if se > 0 else np.nan,
            "direction":          direction,
            "is_significant":     p < 0.05,
            "n_periods":          len(s),
        })

    return pd.DataFrame(records).set_index("metric") if records else pd.DataFrame()


# ------------------------------------------------------------------
# Principal Component Analysis on ratios
# ------------------------------------------------------------------

def ratio_pca(ratios: dict[str, pd.DataFrame], n_components: int = 3) -> dict:
    """
    Run PCA across all ratio groups to find the underlying factors
    driving the company's financial profile.

    Returns:
      explained_variance_pct  — how much variance each component captures
      loadings                — which ratios load onto each component
      scores                  — the company's factor scores over time

    Interpretation: PC1 often represents 'overall financial health',
    PC2 often represents 'growth vs stability' trade-off, etc.
    """
    logger.info("Running PCA on financial ratios.")

    # Concatenate all ratio DataFrames horizontally.
    frames = [df.select_dtypes(include=np.number) for df in ratios.values()]
    combined = pd.concat(frames, axis=1).dropna(how="all")
    # Drop columns that are still >50% NaN after combining.
    combined = combined.loc[:, combined.isnull().mean() < 0.5]
    combined = combined.fillna(combined.median())

    if combined.shape[0] < 3 or combined.shape[1] < 2:
        logger.warning("Not enough data for PCA.")
        return {}

    scaler = StandardScaler()
    X = scaler.fit_transform(combined.values)

    n = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n)
    scores = pca.fit_transform(X)

    ev_pct = (pca.explained_variance_ratio_ * 100).round(1)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=combined.columns,
        columns=[f"PC{i+1}" for i in range(n)],
    ).round(3)

    logger.info(
        "PCA explained variance: %s",
        "  ".join(f"PC{i+1}={v:.0f}%" for i, v in enumerate(ev_pct)),
    )

    return {
        "explained_variance_pct": ev_pct.tolist(),
        "loadings":               loadings,
        "scores":                 pd.DataFrame(scores, index=combined.index,
                                               columns=[f"PC{i+1}" for i in range(n)]).round(3),
    }


# ------------------------------------------------------------------
# Full factor report
# ------------------------------------------------------------------

def run_factor_analysis(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    ratios: dict[str, pd.DataFrame],
) -> dict:
    logger.info("Running full factor analysis.")
    return {
        "predictive_correlations":  predictive_correlations(ratios, income),
        "income_decomposition":     decompose_income_change(income),
        "trend_statistics":         trend_statistics(
            pd.concat([income.select_dtypes(include=np.number),
                       cashflow.select_dtypes(include=np.number)], axis=1)
        ),
        "pca":                      ratio_pca(ratios),
    }
