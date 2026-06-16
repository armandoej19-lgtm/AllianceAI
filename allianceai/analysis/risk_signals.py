"""
Risk-signal detection and strategic interpretation.

This module translates numerical patterns into actionable risk signals —
the kind of conclusions a financial analyst would draw from looking at trends
in leverage, dilution, and capital allocation.  Each signal comes with:
  - A severity level (LOW / MEDIUM / HIGH / CRITICAL).
  - A human-readable explanation of why it was triggered.
  - Numerical evidence (the value that tripped the threshold).

These signals are later consumed by the narrative generator to produce
the AI's natural-language interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)

Severity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass
class RiskSignal:
    category:    str
    name:        str
    severity:    Severity
    explanation: str
    evidence:    dict = field(default_factory=dict)


def _latest(series: pd.Series) -> float:
    """Return the most recent non-NaN value or NaN."""
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else float("nan")


def _trend_slope_normalised(series: pd.Series) -> float:
    """Slope of OLS fit divided by the series mean (relative change per period)."""
    clean = series.dropna().astype(float)
    if len(clean) < 3:
        return float("nan")
    x = np.arange(len(clean), dtype=float)
    slope = np.polyfit(x, clean.values, 1)[0]
    mean = clean.mean()
    return float(slope / abs(mean)) if mean != 0 else float("nan")


# ------------------------------------------------------------------
# Individual signal detectors
# ------------------------------------------------------------------

def _check_leverage(balance: pd.DataFrame, income: pd.DataFrame) -> list[RiskSignal]:
    signals = []

    total_debt = balance.get("Total Debt",            pd.Series(dtype=float))
    equity     = balance.get("Stockholders Equity",   pd.Series(dtype=float))
    ebitda     = income.get("EBITDA",                  pd.Series(dtype=float))
    cash       = balance.get("Cash And Cash Equivalents", pd.Series(dtype=float))

    # --- Debt-to-Equity ---
    eq_safe = equity.replace(0, np.nan)
    de = (total_debt / eq_safe).dropna()
    de_latest = _latest(de)
    if not np.isnan(de_latest):
        if de_latest > 5:
            signals.append(RiskSignal(
                "Leverage", "Extreme Debt-to-Equity", "CRITICAL",
                f"D/E ratio of {de_latest:.1f}x is dangerously high.  The firm may struggle "
                f"to refinance or attract equity investment.  Watch for covenant breaches.",
                {"debt_to_equity": de_latest},
            ))
        elif de_latest > 2:
            signals.append(RiskSignal(
                "Leverage", "High Debt-to-Equity", "HIGH",
                f"D/E of {de_latest:.1f}x — leverage is elevated.  Rising interest rates "
                f"could materially compress earnings.",
                {"debt_to_equity": de_latest},
            ))

    # --- Net Debt / EBITDA ---
    # EBITDA is a flow: annualize to TTM so it's comparable to the net-debt
    # stock (a single quarter would inflate the ratio ~4×).
    from allianceai.analysis.ratios import to_ttm
    ebitda_aligned = to_ttm(ebitda).reindex(balance.index, method="nearest", tolerance=pd.Timedelta("95D"))
    net_debt = (total_debt - cash).reindex(ebitda_aligned.index)
    nd_ebitda = (net_debt / ebitda_aligned.replace(0, np.nan)).dropna()
    nd_latest = _latest(nd_ebitda)
    if not np.isnan(nd_latest) and nd_latest > 4:
        signals.append(RiskSignal(
            "Leverage", "High Net Debt / EBITDA", "HIGH",
            f"Net Debt/EBITDA = {nd_latest:.1f}x.  Lenders typically become uncomfortable "
            f"above 4×; this limits the firm's capacity to take on additional debt.",
            {"net_debt_ebitda": nd_latest},
        ))

    # --- Rapid debt accumulation ---
    slope = _trend_slope_normalised(total_debt)
    if not np.isnan(slope) and slope > 0.15:
        signals.append(RiskSignal(
            "Leverage", "Accelerating Debt Growth", "MEDIUM",
            f"Total debt is growing at ~{slope*100:.0f}% per period on a trend basis.  "
            f"Verify whether this funds productive CapEx or operational deficits.",
            {"debt_growth_rate_per_period": slope},
        ))

    return signals


def _check_dilution(cashflow: pd.DataFrame, balance: pd.DataFrame) -> list[RiskSignal]:
    signals = []

    issuance = cashflow.get("Common Stock Issuance", pd.Series(dtype=float))
    equity   = balance.get("Stockholders Equity",   pd.Series(dtype=float))

    if issuance.empty or equity.empty:
        return signals

    # Large share issuance relative to equity signals dilution risk.
    eq_aligned = equity.reindex(cashflow.index, method="nearest", tolerance=pd.Timedelta("95D"))
    relative_issuance = (issuance / eq_aligned.replace(0, np.nan)).dropna()
    latest = _latest(relative_issuance)

    if not np.isnan(latest) and latest > 0.10:
        signals.append(RiskSignal(
            "Dilution", "Significant Share Issuance", "MEDIUM",
            f"New stock issuance was {latest*100:.1f}% of equity in the latest period.  "
            f"This can fund growth but dilutes existing shareholders — check whether "
            f"book value per share is growing alongside.",
            {"relative_issuance": latest},
        ))
    return signals


def _check_liquidity_stress(balance: pd.DataFrame) -> list[RiskSignal]:
    signals = []

    ca = balance.get("Current Assets",      pd.Series(dtype=float))
    cl = balance.get("Current Liabilities", pd.Series(dtype=float))

    if ca.empty or cl.empty:
        return signals

    cr = (ca / cl.replace(0, np.nan)).dropna()
    cr_latest = _latest(cr)

    if not np.isnan(cr_latest):
        if cr_latest < 1.0:
            signals.append(RiskSignal(
                "Liquidity", "Current Ratio Below 1", "CRITICAL",
                f"Current ratio = {cr_latest:.2f}.  The firm has more short-term obligations "
                f"than liquid assets — immediate liquidity risk.",
                {"current_ratio": cr_latest},
            ))
        elif cr_latest < 1.5:
            signals.append(RiskSignal(
                "Liquidity", "Tight Liquidity", "MEDIUM",
                f"Current ratio = {cr_latest:.2f} — limited buffer against short-term shocks.",
                {"current_ratio": cr_latest},
            ))

    # Deteriorating trend even if currently healthy.
    slope = _trend_slope_normalised(cr)
    if not np.isnan(slope) and slope < -0.10 and cr_latest > 1.5:
        signals.append(RiskSignal(
            "Liquidity", "Declining Liquidity Trend", "LOW",
            f"Current ratio has been shrinking at ~{abs(slope)*100:.0f}% per period.  "
            f"Not yet critical but merits monitoring.",
            {"cr_trend_slope": slope},
        ))

    return signals


def _check_earnings_quality(income: pd.DataFrame, cashflow: pd.DataFrame) -> list[RiskSignal]:
    """
    Cash conversion check: if net income consistently exceeds operating cash
    flow, accruals may be inflating reported earnings.
    """
    signals = []

    ni  = income.get("Net Income",           pd.Series(dtype=float))
    ocf = cashflow.get("Operating Cash Flow", pd.Series(dtype=float))

    if ni.empty or ocf.empty:
        return signals

    ni_aligned = ni.reindex(cashflow.index, method="nearest", tolerance=pd.Timedelta("95D"))
    gap = (ni_aligned - ocf).dropna()

    # If NI > OCF in >60% of periods the firm is booking more profit than cash.
    positive_gap_pct = (gap > 0).mean()
    if positive_gap_pct > 0.60:
        signals.append(RiskSignal(
            "Earnings Quality", "Accrual Inflation Suspicion", "MEDIUM",
            f"Net income exceeded operating cash flow in {positive_gap_pct*100:.0f}% of "
            f"periods.  High accruals can precede earnings restatements.  Verify receivables "
            f"and deferred revenue trends.",
            {"ni_gt_ocf_frequency": positive_gap_pct},
        ))

    return signals


def _check_capex_sustainability(cashflow: pd.DataFrame, income: pd.DataFrame) -> list[RiskSignal]:
    signals = []

    ocf   = cashflow.get("Operating Cash Flow", pd.Series(dtype=float))
    capex = cashflow.get("Capital Expenditure",  pd.Series(dtype=float)).abs()

    if ocf.empty or capex.empty:
        return signals

    # CapEx/OCF > 1 means the firm spends more on assets than it earns from ops.
    ratio = (capex / ocf.replace(0, np.nan)).dropna()
    latest = _latest(ratio)

    if not np.isnan(latest) and latest > 0.80:
        signals.append(RiskSignal(
            "Capital Allocation", "High CapEx / OCF Ratio", "MEDIUM" if latest < 1.2 else "HIGH",
            f"CapEx consumed {latest*100:.0f}% of operating cash flow.  "
            f"{'This is unsustainable without external financing.' if latest >= 1.0 else 'Leave little free cash for debt service or buybacks.'}",
            {"capex_to_ocf": latest},
        ))

    return signals


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

def detect_all_signals(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> list[RiskSignal]:
    """
    Run all signal detectors and return a consolidated list sorted by
    severity (CRITICAL → HIGH → MEDIUM → LOW).
    """
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

    all_signals: list[RiskSignal] = []
    all_signals.extend(_check_leverage(balance, income))
    all_signals.extend(_check_dilution(cashflow, balance))
    all_signals.extend(_check_liquidity_stress(balance))
    all_signals.extend(_check_earnings_quality(income, cashflow))
    all_signals.extend(_check_capex_sustainability(cashflow, income))

    all_signals.sort(key=lambda s: severity_order.get(s.severity, 99))

    logger.info(
        "Risk signals detected: CRITICAL=%d  HIGH=%d  MEDIUM=%d  LOW=%d",
        sum(1 for s in all_signals if s.severity == "CRITICAL"),
        sum(1 for s in all_signals if s.severity == "HIGH"),
        sum(1 for s in all_signals if s.severity == "MEDIUM"),
        sum(1 for s in all_signals if s.severity == "LOW"),
    )
    return all_signals
