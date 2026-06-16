"""
SEC EDGAR XBRL client — deep financial-statement history.

yfinance only exposes ~5 quarters / ~4 years of statements.  The SEC
companyfacts API (free, no key, US filers only) carries the full XBRL filing
history, often 10+ years.  This module fetches it and maps us-gaap concepts
onto the same column names the yfinance fetcher produces, so downstream code
is agnostic about where a period came from.

Endpoints:
  - https://www.sec.gov/files/company_tickers.json          (ticker → CIK)
  - https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json (all facts)

SEC fair-access rules require a descriptive User-Agent (see config).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import requests

from allianceai.core.config import settings
from allianceai.core.exceptions import DataFetchError
from allianceai.core.logging_config import get_logger
from allianceai.core.storage import storage

logger = get_logger(__name__)

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Maps the standard column name (used throughout AllianceAI) to an ordered
# list of us-gaap tags — the first tag with data wins.
# "duration" facts cover a period (income/cashflow); "instant" facts are
# point-in-time (balance sheet).  Sign: EDGAR reports payments as positive
# outflows, while yfinance reports them negative; we flip where needed.
_INCOME_TAGS: dict[str, list[str]] = {
    "Total Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                      "Revenues", "SalesRevenueNet"],
    "Cost Of Revenue": ["CostOfGoodsAndServicesSold", "CostOfRevenue"],
    "Gross Profit": ["GrossProfit"],
    "Operating Income": ["OperatingIncomeLoss"],
    "Net Income": ["NetIncomeLoss"],
    "Basic EPS": ["EarningsPerShareBasic"],
    "Diluted EPS": ["EarningsPerShareDiluted"],
    "Interest Expense": ["InterestExpense"],
    "Tax Provision": ["IncomeTaxExpenseBenefit"],
    "Research And Development": ["ResearchAndDevelopmentExpense"],
}
_BALANCE_TAGS: dict[str, list[str]] = {
    "Total Assets": ["Assets"],
    "Total Liabilities Net Minority Interest": ["Liabilities"],
    "Stockholders Equity": ["StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "Cash And Cash Equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "Current Assets": ["AssetsCurrent"],
    "Current Liabilities": ["LiabilitiesCurrent"],
    "Long Term Debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "Retained Earnings": ["RetainedEarningsAccumulatedDeficit"],
    "Common Stock Equity": ["StockholdersEquity"],
}
_CASHFLOW_TAGS: dict[str, list[str]] = {
    "Operating Cash Flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "Investing Cash Flow": ["NetCashProvidedByUsedInInvestingActivities"],
    "Financing Cash Flow": ["NetCashProvidedByUsedInFinancingActivities"],
    "Capital Expenditure": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "Dividends Paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "Repurchase Of Capital Stock": ["PaymentsForRepurchaseOfCommonStock"],
}
# Tags whose EDGAR sign convention (positive outflow) must be flipped to match
# yfinance (negative outflow).
_NEGATE = {"Capital Expenditure", "Dividends Paid", "Repurchase Of Capital Stock"}


def _http_get(url: str) -> dict:
    resp = requests.get(url, headers={"User-Agent": settings.edgar_user_agent}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _cik_for_ticker(ticker: str) -> int:
    """Resolve a ticker to its SEC CIK number (cached)."""
    cache_key = "edgar::ticker_map"
    cached = storage.load_dataframe(cache_key)
    if cached is None:
        logger.info("Downloading SEC ticker→CIK map.")
        raw = _http_get(_TICKER_MAP_URL)
        cached = pd.DataFrame(list(raw.values()))  # columns: cik_str, ticker, title
        storage.cache_dataframe(cache_key, cached)
    row = cached[cached["ticker"].str.upper() == ticker.upper()]
    if row.empty:
        raise DataFetchError(f"Ticker '{ticker}' not found in SEC EDGAR registry.")
    return int(row.iloc[0]["cik_str"])


def _select_facts(
    units: list[dict[str, Any]],
    quarterly: bool,
    instant: bool,
) -> pd.Series:
    """
    Reduce a raw us-gaap unit list to a date-indexed Series.

    Duration facts are filtered by period length (quarterly ≈ 1 quarter,
    annual ≈ 1 year) — this naturally excludes the cumulative year-to-date
    values that 10-Q cash-flow statements report.  Instant facts are filtered
    by the filing's fiscal period instead.  When the same end date appears in
    multiple filings (originals + amendments), the most recently filed wins.
    """
    rows = []
    for f in units:
        end = f.get("end")
        val = f.get("val")
        if end is None or val is None:
            continue
        if instant:
            fp = f.get("fp", "")
            if not quarterly and fp != "FY":
                continue
        else:
            start = f.get("start")
            if start is None:
                continue
            days = (pd.Timestamp(end) - pd.Timestamp(start)).days
            lo, hi = (76, 100) if quarterly else (340, 390)
            if not (lo <= days <= hi):
                continue
        rows.append((pd.Timestamp(end), f.get("filed", ""), float(val)))

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows, columns=["end", "filed", "val"])
    df = df.sort_values(["end", "filed"]).groupby("end").last()
    return df["val"]


def _build_statement(
    facts: dict[str, Any],
    tag_map: dict[str, list[str]],
    quarterly: bool,
    instant: bool,
) -> pd.DataFrame:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    cols: dict[str, pd.Series] = {}
    for std_col, tags in tag_map.items():
        # Union the series from every tag that carries data for this concept.
        # A single concept is frequently split across multiple us-gaap tags over
        # time — e.g. a filer adopting ASC 606 reports revenue under
        # RevenueFromContractWithCustomer for a few years, then reverts to the
        # generic Revenues tag.  Taking only the first non-empty tag (the old
        # behaviour) silently dropped whole spans of history, leaving multi-year
        # gaps that corrupted every trailing/forecast metric downstream.
        # Earlier-listed tags take priority on overlapping dates; later tags fill
        # the periods the earlier ones don't cover.
        combined = pd.Series(dtype=float)
        for tag in tags:
            concept = gaap.get(tag)
            if not concept:
                continue
            units = concept.get("units", {})
            # Monetary tags report in USD; EPS tags in USD/shares.
            series_raw = units.get("USD") or units.get("USD/shares") or []
            s = _select_facts(series_raw, quarterly=quarterly, instant=instant)
            if s.empty:
                continue
            s = -s if std_col in _NEGATE else s
            combined = combined.combine_first(s)
        if not combined.empty:
            cols[std_col] = combined.sort_index()
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df


def fetch_edgar_statements(
    ticker: str, quarterly: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch full-history income / balance / cashflow statements from SEC EDGAR.
    Returns dict with keys 'income', 'balance', 'cashflow' (DataFrames may be
    empty for non-US filers or tickers without XBRL data).  Results cached.
    """
    freq = "quarterly" if quarterly else "annual"
    cache_keys = {name: f"edgar::{ticker}::{name}::{freq}"
                  for name in ("income", "balance", "cashflow")}
    cached = {name: storage.load_dataframe(k) for name, k in cache_keys.items()}
    if all(v is not None for v in cached.values()):
        return cached  # type: ignore[return-value]

    cik = _cik_for_ticker(ticker)
    logger.info("Fetching EDGAR companyfacts for '%s' (CIK %010d).", ticker, cik)
    facts = _http_get(_COMPANYFACTS_URL.format(cik=cik))

    out = {
        "income":   _build_statement(facts, _INCOME_TAGS,   quarterly, instant=False),
        "balance":  _build_statement(facts, _BALANCE_TAGS,  quarterly, instant=True),
        "cashflow": _build_statement(facts, _CASHFLOW_TAGS, quarterly, instant=False),
    }
    for name, df in out.items():
        logger.info("  EDGAR %-8s — %d periods.", name, len(df))
        storage.cache_dataframe(cache_keys[name], df)
    return out


def extend_history(yf_df: pd.DataFrame, edgar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge EDGAR history into a yfinance statement.  yfinance values win on
    overlapping (date, column) cells; EDGAR fills older periods and gaps.
    """
    if edgar_df is None or edgar_df.empty:
        return yf_df
    if yf_df is None or yf_df.empty:
        return edgar_df

    # Align EDGAR period-ends to yfinance dates that are within a few days
    # (fiscal calendars can differ by a day or two across sources).
    edgar = edgar_df.copy()
    for ts in list(edgar.index):
        close = yf_df.index[abs(yf_df.index - ts) <= pd.Timedelta("7D")]
        if len(close) and ts not in yf_df.index:
            edgar = edgar.rename(index={ts: close[0]})

    combined = yf_df.combine_first(edgar)
    return combined.sort_index()
