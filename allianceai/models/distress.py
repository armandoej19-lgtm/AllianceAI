"""
Quantitative financial distress probability models.

Three academically validated models are combined into a single ensemble
probability.  Each model was calibrated on real bankruptcy datasets and
outputs a direct probability — not a qualitative label.

Models used:
  1. Altman Z-Score (1968, revised 2000) — discriminant analysis
  2. Ohlson O-Score (1980) — logistic regression, direct probability output
  3. Zmijewski Score (1984) — probit regression

Ensemble: probability = weighted average of the three model probabilities.
Weights reflect out-of-sample accuracy in academic literature:
  Ohlson: 0.45, Zmijewski: 0.35, Altman: 0.20
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)


def _safe(series: pd.Series, default: float = np.nan) -> pd.Series:
    return series.fillna(default)


# ------------------------------------------------------------------
# Ohlson O-Score → P(distress)
# ------------------------------------------------------------------

def ohlson_o_score(income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame) -> pd.DataFrame:
    """
    Ohlson (1980) logistic regression distress model.

    Coefficients trained on 2,163 US industrial firms (1970-1976).
    Output: probability between 0 and 1.

    O = -1.32 - 0.407*SIZE + 6.03*TLTA - 1.43*WCTA + 0.076*CLCA
            - 1.72*OENEG - 2.37*NITA - 1.83*FUTL + 0.285*INTWO - 0.521*CHIN

    where:
      SIZE  = log(Total Assets / GNP price-level index)  ≈ log(Total Assets)
      TLTA  = Total Liabilities / Total Assets
      WCTA  = Working Capital / Total Assets
      CLCA  = Current Liabilities / Current Assets
      OENEG = 1 if Total Liabilities > Total Assets
      NITA  = Net Income / Total Assets
      FUTL  = Funds From Operations / Total Liabilities  ≈ OCF / Total Liabilities
      INTWO = 1 if net loss in both current and prior period
      CHIN  = (NI_t - NI_{t-1}) / (|NI_t| + |NI_{t-1}|)  — earnings change
    """
    idx = balance.index
    out = pd.DataFrame(index=idx)

    ta   = _safe(balance.get("Total Assets",   pd.Series(dtype=float)).reindex(idx))
    tl   = _safe(balance.get("Total Liabilities Net Minority Interest", pd.Series(dtype=float)).reindex(idx))
    ca   = _safe(balance.get("Current Assets",  pd.Series(dtype=float)).reindex(idx))
    cl   = _safe(balance.get("Current Liabilities", pd.Series(dtype=float)).reindex(idx))
    ni   = _safe(income.get("Net Income",  pd.Series(dtype=float)).reindex(idx, method="nearest", tolerance=pd.Timedelta("95D")))
    ocf  = _safe(cashflow.get("Operating Cash Flow", pd.Series(dtype=float)).reindex(idx, method="nearest", tolerance=pd.Timedelta("95D")))

    ta_s  = ta.replace(0, np.nan)
    tl_s  = tl.replace(0, np.nan)
    ca_s  = ca.replace(0, np.nan)
    cl_s  = cl.replace(0, np.nan)

    SIZE  = np.log(ta_s.clip(lower=1))
    TLTA  = tl_s / ta_s
    WCTA  = (ca - cl) / ta_s
    CLCA  = cl_s / ca_s
    OENEG = (tl > ta).astype(float)
    NITA  = ni / ta_s
    FUTL  = ocf / tl_s
    INTWO = ((ni < 0) & (ni.shift(1) < 0)).astype(float)
    ni_abs_sum = ni.abs() + ni.shift(1).abs()
    CHIN  = (ni - ni.shift(1)) / ni_abs_sum.replace(0, np.nan)

    O = (-1.32
         - 0.407 * SIZE
         + 6.03  * TLTA
         - 1.43  * WCTA
         + 0.076 * CLCA
         - 1.72  * OENEG
         - 2.37  * NITA
         - 1.83  * FUTL
         + 0.285 * INTWO
         - 0.521 * CHIN)

    out["o_score"]           = O.round(4)
    out["p_distress_ohlson"] = (1 / (1 + np.exp(-O))).round(4)
    return out


# ------------------------------------------------------------------
# Zmijewski Score → P(distress)
# ------------------------------------------------------------------

def zmijewski_score(income: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    """
    Zmijewski (1984) probit model.

    X = -4.336 - 4.513*(NI/TA) + 5.679*(TL/TA) + 0.004*(CA/CL)

    P(distress) = Φ(X)  where Φ is the standard normal CDF.
    """
    idx = balance.index

    ta  = _safe(balance.get("Total Assets", pd.Series(dtype=float)).reindex(idx)).replace(0, np.nan)
    tl  = _safe(balance.get("Total Liabilities Net Minority Interest", pd.Series(dtype=float)).reindex(idx))
    ca  = _safe(balance.get("Current Assets", pd.Series(dtype=float)).reindex(idx))
    cl  = _safe(balance.get("Current Liabilities", pd.Series(dtype=float)).reindex(idx)).replace(0, np.nan)
    ni  = _safe(income.get("Net Income", pd.Series(dtype=float)).reindex(idx, method="nearest", tolerance=pd.Timedelta("95D")))

    X = -4.336 - 4.513 * (ni / ta) + 5.679 * (tl / ta) + 0.004 * (ca / cl)

    from scipy.stats import norm
    p = pd.Series(norm.cdf(X.values.astype(float)), index=idx)

    out = pd.DataFrame(index=idx)
    out["zmijewski_score"]      = X.round(4)
    out["p_distress_zmijewski"] = p.round(4)
    return out


# ------------------------------------------------------------------
# Ensemble
# ------------------------------------------------------------------

def compute_distress_ensemble(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    market_cap: float | None = None,
    non_manufacturing: bool = False,
) -> pd.DataFrame:
    """
    Combine all three models into a single distress probability per period.

    Returns a DataFrame with columns:
      p_distress          — ensemble probability (0–1)
      p_distress_ohlson   — Ohlson model
      p_distress_zmijewski— Zmijewski model
      distress_zone       — LOW / ELEVATED / HIGH / CRITICAL
    """
    logger.info("Computing distress probability ensemble.")

    ohlson   = ohlson_o_score(income, balance, cashflow)
    zmijewski = zmijewski_score(income, balance)

    from allianceai.analysis.health_score import altman_z_score
    altman = altman_z_score(income, balance, market_cap=market_cap,
                            non_manufacturing=non_manufacturing)

    # Convert Altman Z to a probability with a sigmoid calibrated so the
    # variant's distress threshold maps to p≈0.70 and its safe threshold to
    # p≈0.20 (thresholds differ between the original Z and Z'' zones).
    if non_manufacturing:
        k, c = 1.49, 1.67   # Z'' zones: distress < 1.1, safe > 2.6
    else:
        k, c = 1.89, 2.26   # Original zones: distress < 1.81, safe > 2.99
    z = altman["z_score"].reindex(ohlson.index, method="nearest", tolerance=pd.Timedelta("95D"))
    p_altman = (1 / (1 + np.exp(k * (z - c)))).clip(0, 1)

    # Weighted ensemble — Ohlson has highest empirical accuracy.
    p_ohlson    = ohlson["p_distress_ohlson"].reindex(ohlson.index)
    p_zmijewski = zmijewski["p_distress_zmijewski"].reindex(ohlson.index)

    ensemble = (0.45 * p_ohlson + 0.35 * p_zmijewski + 0.20 * p_altman).clip(0, 1)

    out = pd.DataFrame({
        "p_distress":            ensemble.round(4),
        "p_distress_ohlson":     p_ohlson.round(4),
        "p_distress_zmijewski":  p_zmijewski.round(4),
        "p_distress_altman":     p_altman.round(4),
    }, index=ohlson.index)

    out["distress_zone"] = pd.cut(
        ensemble,
        bins=[-np.inf, 0.20, 0.40, 0.65, np.inf],
        labels=["LOW", "ELEVATED", "HIGH", "CRITICAL"],
    )

    latest = out["p_distress"].dropna()
    if not latest.empty:
        logger.info(
            "Distress probability — latest: %.1f%%  |  zone: %s",
            latest.iloc[-1] * 100,
            out["distress_zone"].iloc[-1],
        )
    return out
