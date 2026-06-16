"""
Composite financial health scoring.

This module implements two complementary scoring systems:

1. Altman Z-Score  — the classic bankruptcy-prediction model (1968, revised 2000
   for non-manufacturing / private firms).  Produces a numerical score that maps
   to three zones: Safe (Z > 2.99), Grey (1.81–2.99), Distress (Z < 1.81).

2. AllianceAI Health Score — a proprietary 0–100 composite that incorporates
   liquidity, leverage, profitability, and cash-flow quality.  Unlike the
   Altman Z-Score it is not domain-restricted and works for REITs, ETFs,
   and growth-stage startups (which often have negative earnings but healthy
   cash flows).

Both scores are computed for each available time period so analysts can
observe how financial health evolves over time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Altman Z-Score  (public company variant)
# ------------------------------------------------------------------

def _is_quarterly(idx: pd.DatetimeIndex) -> bool:
    """True when the statement index is spaced roughly one quarter apart."""
    if len(idx) < 2:
        return False
    median_gap = pd.Series(idx).diff().median()
    return median_gap < pd.Timedelta("200D")


def _annualize(series: pd.Series, quarterly: bool) -> pd.Series:
    """
    Convert a flow series (revenue, EBIT, …) to a trailing-twelve-month basis.
    For quarterly data we sum the trailing 4 quarters; where fewer than 4 are
    available the partial sum is scaled up so early periods aren't deflated.
    Annual data is returned unchanged.
    """
    if not quarterly:
        return series
    ttm = series.rolling(4, min_periods=1).sum()
    count = series.rolling(4, min_periods=1).count()
    return ttm * (4 / count.replace(0, np.nan))


def altman_z_score(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    market_cap: float | None = None,
    non_manufacturing: bool = False,
) -> pd.DataFrame:
    """
    Compute the Altman Z-Score for each available period.

    Original (public manufacturing):
        Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5
    Z''-Score (non-manufacturers / service firms, Altman 1995):
        Z'' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4      (no asset-turnover term)
    where:
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT (TTM) / Total Assets
        X4 = Market Value of Equity / Total Liabilities  (book equity fallback)
        X5 = Revenue (TTM) / Total Assets

    Flow variables are annualized to TTM when the input is quarterly —
    feeding raw quarterly revenue/EBIT into the annual-calibrated weights
    understates Z by roughly 4×.
    """
    variant = "Z''-non-manufacturing" if non_manufacturing else "Z-original"
    logger.info("Computing Altman Z-Score (%s).", variant)

    idx = balance.index
    out = pd.DataFrame(index=idx)
    quarterly = _is_quarterly(income.index if not income.empty else idx)

    def _get(df: pd.DataFrame, col: str) -> pd.Series:
        if col in df.columns:
            return df[col].reindex(idx, method="nearest", tolerance=pd.Timedelta("95D")).astype(float)
        logger.debug("Altman Z-Score: column '%s' not found — using NaN.", col)
        return pd.Series(np.nan, index=idx)

    def _get_flow(df: pd.DataFrame, col: str) -> pd.Series:
        """Flow metric annualized on its native index, then aligned to *idx*."""
        if col in df.columns:
            ttm = _annualize(df[col].astype(float).sort_index(), quarterly)
            return ttm.reindex(idx, method="nearest", tolerance=pd.Timedelta("95D"))
        return pd.Series(np.nan, index=idx)

    ca  = _get(balance, "Current Assets")
    cl  = _get(balance, "Current Liabilities")
    ta  = _get(balance, "Total Assets")
    re  = _get(balance, "Retained Earnings")
    eq  = _get(balance, "Stockholders Equity")
    tl  = _get(balance, "Total Liabilities Net Minority Interest")
    rev = _get_flow(income, "Total Revenue")
    op  = _get_flow(income, "Operating Income")

    wc = ca - cl

    ta_safe = ta.replace(0, np.nan)
    tl_safe = tl.replace(0, np.nan)

    x1 = wc  / ta_safe
    x2 = re  / ta_safe
    x3 = op  / ta_safe
    if market_cap is not None and market_cap > 0:
        # Approximate historical market value by scaling today's market cap by
        # the period's book equity relative to the latest period.  Far closer
        # to Altman's intent than raw book equity for buyback-heavy firms,
        # without applying today's cap to decade-old balance sheets.
        eq_latest = eq.dropna().iloc[-1] if eq.notna().any() else np.nan
        if pd.notna(eq_latest) and eq_latest > 0:
            mv = market_cap * (eq / eq_latest).clip(lower=0)
        else:
            mv = pd.Series(market_cap, index=idx)
        x4 = mv / tl_safe
    else:
        x4 = eq / tl_safe   # Book equity fallback when market value is unknown.
    x5 = rev / ta_safe

    if non_manufacturing:
        z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
        bins = [-np.inf, 1.1, 2.6, np.inf]
    else:
        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        bins = [-np.inf, 1.81, 2.99, np.inf]

    out["z_score"] = z
    out["z_zone"] = pd.cut(z, bins=bins, labels=["distress", "grey", "safe"])
    out.attrs["variant"] = variant

    logger.info("Z-Score range: min=%.2f, max=%.2f", z.min(), z.max())
    return out


# ------------------------------------------------------------------
# AllianceAI Composite Health Score
# ------------------------------------------------------------------

# Each sub-score is normalised to [0, 1] before weighting.
_WEIGHTS = {
    "liquidity":     0.20,
    "leverage":      0.25,
    "profitability": 0.30,
    "cashflow":      0.25,
}


def _clip_norm(series: pd.Series, low: float, high: float) -> pd.Series:
    """Linearly map [low, high] → [0, 1], clipping outside the range."""
    return ((series - low) / (high - low)).clip(0, 1)


def _liquidity_subscore(balance: pd.DataFrame) -> pd.Series:
    ca = balance.get("Current Assets",      pd.Series(np.nan, index=balance.index)).astype(float)
    cl = balance.get("Current Liabilities", pd.Series(np.nan, index=balance.index)).astype(float)
    cl_safe = cl.replace(0, np.nan)
    current_ratio = ca / cl_safe
    # Current ratio: < 1 is risky, 1–3 is healthy, > 3 may signal idle assets.
    # We map [0.5, 3.0] → [0, 1].
    return _clip_norm(current_ratio, 0.5, 3.0)


def _leverage_subscore(balance: pd.DataFrame, income: pd.DataFrame) -> pd.Series:
    debt = balance.get("Total Debt",            pd.Series(np.nan, index=balance.index)).astype(float)
    eq   = balance.get("Stockholders Equity",   pd.Series(np.nan, index=balance.index)).astype(float)
    eq_safe = eq.replace(0, np.nan)
    de = debt / eq_safe
    # D/E: 0 = no debt (best), 4+ = highly leveraged (worst).
    # We invert so higher score = lower leverage.
    return 1 - _clip_norm(de, 0, 4.0)


def _profitability_subscore(income: pd.DataFrame) -> pd.Series:
    rev = income.get("Total Revenue", pd.Series(np.nan, index=income.index)).astype(float)
    ni  = income.get("Net Income",    pd.Series(np.nan, index=income.index)).astype(float)
    rev_safe = rev.replace(0, np.nan)
    margin = ni / rev_safe
    # Net margin: < -20 % is very bad, > 20 % is excellent.
    return _clip_norm(margin, -0.20, 0.20)


def _cashflow_subscore(cashflow: pd.DataFrame, income: pd.DataFrame) -> pd.Series:
    ocf = cashflow.get("Operating Cash Flow", pd.Series(np.nan, index=cashflow.index)).astype(float)
    rev = income.get("Total Revenue", pd.Series(np.nan, index=cashflow.index)).astype(float)
    rev_aligned = rev.reindex(cashflow.index, method="nearest", tolerance=pd.Timedelta("95D"))
    rev_safe = rev_aligned.replace(0, np.nan)
    ocf_margin = ocf / rev_safe
    return _clip_norm(ocf_margin, -0.10, 0.30)


def compute_health_score(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the composite AllianceAI Health Score (0–100) for each period.
    Also returns each sub-score so analysts can see which dimension is weak.
    """
    logger.info("Computing AllianceAI composite health score.")

    # Align all sub-scores to the balance sheet index (widest coverage).
    idx = balance.index
    income_r   = income.reindex(idx,    method="nearest", tolerance=pd.Timedelta("95D"))
    cashflow_r = cashflow.reindex(idx,  method="nearest", tolerance=pd.Timedelta("95D"))

    liq  = _liquidity_subscore(balance)
    lev  = _leverage_subscore(balance, income_r)
    prof = _profitability_subscore(income_r)
    cf   = _cashflow_subscore(cashflow_r, income_r)

    composite = (
        _WEIGHTS["liquidity"]     * liq  +
        _WEIGHTS["leverage"]      * lev  +
        _WEIGHTS["profitability"] * prof +
        _WEIGHTS["cashflow"]      * cf
    ) * 100  # scale to 0–100

    out = pd.DataFrame({
        "health_score":          composite.round(1),
        "liquidity_subscore":    (liq  * 100).round(1),
        "leverage_subscore":     (lev  * 100).round(1),
        "profitability_subscore": (prof * 100).round(1),
        "cashflow_subscore":     (cf   * 100).round(1),
    }, index=idx)

    logger.info(
        "Health score — latest: %.1f | min: %.1f | max: %.1f",
        out["health_score"].iloc[-1] if not out.empty else float("nan"),
        out["health_score"].min(),
        out["health_score"].max(),
    )
    return out
