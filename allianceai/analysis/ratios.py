"""
Financial ratio computation.

All ratios are computed from the three core statements.  Each function accepts
DataFrames in (date × metric) format as produced by allianceai.data.fetcher.

Design choices:
  - Every function returns a DataFrame indexed by date.
  - Division is safe: we use pd.Series.div() with fill_value=np.nan so
    zero denominators produce NaN rather than raising ZeroDivisionError.
  - Missing input columns produce NaN in output (logged at DEBUG level).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series; zeros in the denominator become NaN."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def _is_quarterly(idx: pd.Index) -> bool:
    """Infer reporting frequency from the median spacing of period-ends."""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return True  # the pipeline defaults to quarterly statements
    median_gap = pd.Series(idx.sort_values()).diff().dt.days.median()
    return bool(median_gap < 135)  # ~91d quarterly vs ~365d annual


def to_ttm(flow: pd.Series) -> pd.Series:
    """
    Convert a *flow* series (revenue, EBITDA, net income …) to a
    trailing-twelve-month basis when the data is quarterly.

    A flow accumulates over a period, so a single quarter cannot be compared
    against a point-in-time *stock* like debt or equity — doing so (e.g.
    Net Debt / one-quarter-EBITDA) overstates the ratio ~4×.  We sum the
    trailing 4 quarters, scaling up partial windows so the first few periods
    aren't understated.  Annual data is already on a 12-month basis and is
    returned unchanged.
    """
    s = flow.dropna().astype(float).sort_index()
    if s.empty or not _is_quarterly(s.index):
        return flow
    count = s.rolling(4, min_periods=1).count()
    ttm = s.rolling(4, min_periods=1).sum()
    return ttm * (4.0 / count)


def compute_liquidity_ratios(balance: pd.DataFrame) -> pd.DataFrame:
    """
    Current Ratio, Quick Ratio, Cash Ratio.
    These measure short-term solvency — whether the firm can pay bills due
    within the next 12 months without selling long-term assets.
    """
    out = pd.DataFrame(index=balance.index)

    ca = balance.get("Current Assets", pd.Series(np.nan, index=balance.index))
    cl = balance.get("Current Liabilities", pd.Series(np.nan, index=balance.index))
    cash = balance.get("Cash And Cash Equivalents", pd.Series(np.nan, index=balance.index))

    out["current_ratio"] = _safe_div(ca, cl)
    out["cash_ratio"] = _safe_div(cash, cl)

    logger.debug("Computed liquidity ratios for %d periods.", len(out))
    return out


def compute_leverage_ratios(balance: pd.DataFrame, income: pd.DataFrame) -> pd.DataFrame:
    """
    Debt-to-Equity, Debt-to-Assets, Interest Coverage, Net Debt / EBITDA.
    These reveal how much the firm relies on borrowed money versus its own
    equity.  High leverage amplifies returns but also risk.
    """
    out = pd.DataFrame(index=balance.index)

    total_debt = balance.get("Total Debt", pd.Series(np.nan, index=balance.index))
    equity = balance.get("Stockholders Equity", pd.Series(np.nan, index=balance.index))
    total_assets = balance.get("Total Assets", pd.Series(np.nan, index=balance.index))
    cash = balance.get("Cash And Cash Equivalents", pd.Series(np.nan, index=balance.index))

    out["debt_to_equity"] = _safe_div(total_debt, equity)
    out["debt_to_assets"] = _safe_div(total_debt, total_assets)

    # Net Debt / EBITDA — net debt is a point-in-time stock, so EBITDA must be
    # on the same 12-month basis (TTM) rather than a single quarter.
    ebitda = income.get("EBITDA", pd.Series(dtype=float))
    ebitda_ttm = to_ttm(ebitda)
    ebitda_aligned = ebitda_ttm.reindex(balance.index, method="nearest", tolerance=pd.Timedelta("95D"))
    net_debt = total_debt - cash
    out["net_debt_to_ebitda"] = _safe_div(net_debt, ebitda_aligned)

    interest = income.get("Interest Expense", pd.Series(dtype=float))
    interest_aligned = interest.reindex(balance.index, method="nearest", tolerance=pd.Timedelta("95D"))
    ebit = ebitda_aligned  # Approximate; use Operating Income if available.
    op_income = income.get("Operating Income", pd.Series(dtype=float))
    if not op_income.empty:
        ebit = op_income.reindex(balance.index, method="nearest", tolerance=pd.Timedelta("95D"))
    out["interest_coverage"] = _safe_div(ebit, interest_aligned.abs())

    logger.debug("Computed leverage ratios for %d periods.", len(out))
    return out


def compute_profitability_ratios(income: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    """
    Gross Margin, Operating Margin, Net Margin, ROE, ROA, EBITDA Margin.
    Margins show how efficiently revenue converts to profit at each stage.
    ROE/ROA measure how well the firm uses shareholder funds and total assets.
    """
    out = pd.DataFrame(index=income.index)

    revenue = income.get("Total Revenue", pd.Series(np.nan, index=income.index))
    gross_profit = income.get("Gross Profit", pd.Series(np.nan, index=income.index))
    op_income = income.get("Operating Income", pd.Series(np.nan, index=income.index))
    net_income = income.get("Net Income", pd.Series(np.nan, index=income.index))
    ebitda = income.get("EBITDA", pd.Series(np.nan, index=income.index))

    out["gross_margin"] = _safe_div(gross_profit, revenue)
    out["operating_margin"] = _safe_div(op_income, revenue)
    out["net_margin"] = _safe_div(net_income, revenue)
    out["ebitda_margin"] = _safe_div(ebitda, revenue)

    # ROE and ROA need balance sheet data aligned by date.
    equity = balance.get("Stockholders Equity", pd.Series(dtype=float))
    assets = balance.get("Total Assets", pd.Series(dtype=float))
    eq_aligned = equity.reindex(income.index, method="nearest", tolerance=pd.Timedelta("95D"))
    at_aligned = assets.reindex(income.index, method="nearest", tolerance=pd.Timedelta("95D"))

    out["roe"] = _safe_div(net_income, eq_aligned)
    out["roa"] = _safe_div(net_income, at_aligned)

    logger.debug("Computed profitability ratios for %d periods.", len(out))
    return out


def compute_cashflow_ratios(cashflow: pd.DataFrame, balance: pd.DataFrame, income: pd.DataFrame) -> pd.DataFrame:
    """
    Operating Cash Flow Margin, Free Cash Flow Yield, CapEx Intensity.
    Cash flow ratios are harder to manipulate than accrual-based metrics and
    are therefore a critical cross-check on earnings quality.
    """
    out = pd.DataFrame(index=cashflow.index)

    ocf = cashflow.get("Operating Cash Flow", pd.Series(np.nan, index=cashflow.index))
    fcf = cashflow.get("Free Cash Flow", pd.Series(np.nan, index=cashflow.index))
    capex = cashflow.get("Capital Expenditure", pd.Series(np.nan, index=cashflow.index))

    revenue = income.get("Total Revenue", pd.Series(dtype=float))
    rev_aligned = revenue.reindex(cashflow.index, method="nearest", tolerance=pd.Timedelta("95D"))

    out["ocf_margin"] = _safe_div(ocf, rev_aligned)
    out["capex_intensity"] = _safe_div(capex.abs(), rev_aligned)

    assets = balance.get("Total Assets", pd.Series(dtype=float))
    at_aligned = assets.reindex(cashflow.index, method="nearest", tolerance=pd.Timedelta("95D"))
    out["fcf_to_assets"] = _safe_div(fcf, at_aligned)

    logger.debug("Computed cash flow ratios for %d periods.", len(out))
    return out


def compute_all_ratios(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run all ratio groups and return them in a labelled dict."""
    return {
        "liquidity":    compute_liquidity_ratios(balance),
        "leverage":     compute_leverage_ratios(balance, income),
        "profitability": compute_profitability_ratios(income, balance),
        "cashflow":     compute_cashflow_ratios(cashflow, balance, income),
    }
