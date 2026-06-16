"""
Financial data fetcher.

Primary source: yfinance (free, no key required).
Fallback for macroeconomic series: FRED (free key, optional).

The fetcher is intentionally resilient:
  - Retries transient HTTP errors with exponential back-off (tenacity).
  - Fills structural missing columns with NaN so downstream code always
    receives a DataFrame with a predictable shape.
  - Logs every missing field so analysts know what data is absent.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from allianceai.core.exceptions import DataFetchError, InsufficientDataError
from allianceai.core.logging_config import get_logger
from allianceai.core.storage import storage

logger = get_logger(__name__)

# Columns we always want present on each financial statement.
# Downstream analysis code can rely on these existing (though they may be NaN).
_INCOME_COLS = [
    "Total Revenue", "Cost Of Revenue", "Gross Profit",
    "Operating Income", "EBITDA", "Net Income",
    "Basic EPS", "Diluted EPS", "Interest Expense",
    "Tax Provision", "Research And Development",
]
_BALANCE_COLS = [
    "Total Assets", "Total Liabilities Net Minority Interest",
    "Stockholders Equity", "Total Debt", "Cash And Cash Equivalents",
    "Current Assets", "Current Liabilities",
    "Long Term Debt", "Retained Earnings",
    "Common Stock Equity", "Goodwill And Other Intangible Assets",
]
_CASHFLOW_COLS = [
    "Operating Cash Flow", "Capital Expenditure",
    "Free Cash Flow", "Investing Cash Flow", "Financing Cash Flow",
    "Dividends Paid", "Repurchase Of Capital Stock",
    "Common Stock Issuance",
]


def _ensure_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Add any missing columns as NaN and log which ones were absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(
            "Missing %d field(s) in statement — filling with NaN: %s",
            len(missing),
            missing,
        )
    for col in missing:
        df[col] = np.nan
    return df


def _transpose_yf_statement(raw: pd.DataFrame | None) -> pd.DataFrame:
    """
    yfinance returns statements with metrics as rows and dates as columns.
    We transpose to (date × metric) for consistency with standard DataFrame
    conventions used throughout the rest of the codebase.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.T.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.sort_index(inplace=True)
    return df


# ------------------------------------------------------------------
# Core fetching functions
# ------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _fetch_ticker_raw(ticker: str) -> yf.Ticker:
    logger.debug("Requesting yfinance Ticker object for '%s'.", ticker)
    t = yf.Ticker(ticker)
    # Force a lightweight property access to detect invalid tickers early.
    info = t.info
    if not info or info.get("regularMarketPrice") is None and info.get("navPrice") is None:
        # Some valid tickers (ETFs, REITs) use navPrice instead of regularMarketPrice.
        # If both are absent the ticker is likely invalid.
        logger.warning(
            "Ticker '%s' returned no market price — it may be invalid or delisted.", ticker
        )
    return t


def fetch_company_info(ticker: str) -> dict[str, Any]:
    """
    Return a dict of metadata (sector, industry, market cap, description …).
    Missing keys are filled with None so callers can rely on key presence.
    """
    cache_key = f"info::{ticker}"
    cached = storage.load_dataframe(cache_key)
    if cached is not None:
        return cached.iloc[0].to_dict()

    logger.info("Fetching company info for '%s'.", ticker)
    try:
        t = _fetch_ticker_raw(ticker)
        info = t.info or {}
    except Exception as exc:
        raise DataFetchError(f"Could not fetch info for '{ticker}'.") from exc

    wanted = [
        "longName", "sector", "industry", "country", "exchange",
        "marketCap", "enterpriseValue", "trailingPE", "forwardPE",
        "priceToBook", "dividendYield", "beta", "longBusinessSummary",
        "fullTimeEmployees", "website", "quoteType",
    ]
    result = {k: info.get(k) for k in wanted}

    # Persist as a single-row DataFrame.
    storage.cache_dataframe(cache_key, pd.DataFrame([result]))
    return result


def fetch_income_statement(ticker: str, quarterly: bool = True) -> pd.DataFrame:
    """
    Fetch income statement.  Returns a (date × metric) DataFrame.
    Always contains _INCOME_COLS columns (NaN where data is unavailable).
    """
    freq = "quarterly" if quarterly else "annual"
    cache_key = f"income::{ticker}::{freq}"
    cached = storage.load_dataframe(cache_key)
    if cached is not None:
        return cached

    logger.info("Fetching %s income statement for '%s'.", freq, ticker)
    try:
        t = _fetch_ticker_raw(ticker)
        raw = t.quarterly_income_stmt if quarterly else t.income_stmt
    except Exception as exc:
        raise DataFetchError(f"Income statement fetch failed for '{ticker}'.") from exc

    df = _transpose_yf_statement(raw)
    if df.empty:
        logger.warning("Income statement for '%s' is empty.", ticker)
        return df

    df = _ensure_columns(df, _INCOME_COLS)
    storage.cache_dataframe(cache_key, df)
    return df


def fetch_balance_sheet(ticker: str, quarterly: bool = True) -> pd.DataFrame:
    """
    Fetch balance sheet.  Same shape contract as fetch_income_statement.
    """
    freq = "quarterly" if quarterly else "annual"
    cache_key = f"balance::{ticker}::{freq}"
    cached = storage.load_dataframe(cache_key)
    if cached is not None:
        return cached

    logger.info("Fetching %s balance sheet for '%s'.", freq, ticker)
    try:
        t = _fetch_ticker_raw(ticker)
        raw = t.quarterly_balance_sheet if quarterly else t.balance_sheet
    except Exception as exc:
        raise DataFetchError(f"Balance sheet fetch failed for '{ticker}'.") from exc

    df = _transpose_yf_statement(raw)
    if df.empty:
        logger.warning("Balance sheet for '%s' is empty.", ticker)
        return df

    df = _ensure_columns(df, _BALANCE_COLS)
    storage.cache_dataframe(cache_key, df)
    return df


def fetch_cash_flow(ticker: str, quarterly: bool = True) -> pd.DataFrame:
    """
    Fetch cash flow statement.  Same shape contract as fetch_income_statement.
    """
    freq = "quarterly" if quarterly else "annual"
    cache_key = f"cashflow::{ticker}::{freq}"
    cached = storage.load_dataframe(cache_key)
    if cached is not None:
        return cached

    logger.info("Fetching %s cash flow statement for '%s'.", freq, ticker)
    try:
        t = _fetch_ticker_raw(ticker)
        raw = t.quarterly_cashflow if quarterly else t.cashflow
    except Exception as exc:
        raise DataFetchError(f"Cash flow fetch failed for '{ticker}'.") from exc

    df = _transpose_yf_statement(raw)
    if df.empty:
        logger.warning("Cash flow statement for '%s' is empty.", ticker)
        return df

    df = _ensure_columns(df, _CASHFLOW_COLS)
    storage.cache_dataframe(cache_key, df)
    return df


def fetch_price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch OHLCV price history.  *period* uses yfinance notation (1d, 5d, 1mo,
    3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max).
    """
    cache_key = f"price::{ticker}::{period}"
    cached = storage.load_dataframe(cache_key)
    if cached is not None:
        return cached

    logger.info("Fetching price history for '%s' (period=%s).", ticker, period)
    try:
        t = _fetch_ticker_raw(ticker)
        df = t.history(period=period)
    except Exception as exc:
        raise DataFetchError(f"Price history fetch failed for '{ticker}'.") from exc

    if df.empty:
        raise InsufficientDataError(f"No price history returned for '{ticker}'.")

    df.index.name = "date"
    storage.cache_dataframe(cache_key, df)
    return df


def fetch_all(ticker: str, quarterly: bool = True) -> dict[str, pd.DataFrame | dict]:
    """
    Convenience wrapper that fetches all statements for a ticker in one call.
    Returns a dict with keys: info, income, balance, cashflow, prices.
    Partial failures are caught per-statement so one bad endpoint doesn't
    abort the entire pipeline.
    """
    logger.info("=== Beginning full data fetch for '%s' ===", ticker)
    result: dict[str, Any] = {}

    for name, fn, kwargs in [
        ("info",     fetch_company_info,    {"ticker": ticker}),
        ("income",   fetch_income_statement, {"ticker": ticker, "quarterly": quarterly}),
        ("balance",  fetch_balance_sheet,    {"ticker": ticker, "quarterly": quarterly}),
        ("cashflow", fetch_cash_flow,        {"ticker": ticker, "quarterly": quarterly}),
        ("prices",   fetch_price_history,    {"ticker": ticker}),
    ]:
        try:
            result[name] = fn(**kwargs)
            logger.info("  [OK] %-10s fetched.", name)
        except Exception as exc:
            logger.error("  [FAIL] %-10s — %s", name, exc)
            result[name] = {} if name == "info" else pd.DataFrame()

    # --------------------------------------------------------------
    # Deep history: yfinance only carries ~5 quarters.  When coverage is
    # thin, extend the statements with SEC EDGAR XBRL history (US filers).
    # Failures are non-fatal — yfinance data is kept as-is.
    # --------------------------------------------------------------
    from allianceai.core.config import settings as _settings

    statement_keys = ("income", "balance", "cashflow")
    thin = any(
        isinstance(result.get(k), pd.DataFrame) and len(result[k]) < _settings.edgar_min_periods
        for k in statement_keys
    )
    quote_type = (result.get("info") or {}).get("quoteType", "") or ""
    if thin and quote_type.upper() not in ("ETF", "MUTUALFUND", "INDEX", "MONEYMARKET"):
        try:
            from allianceai.data.edgar import extend_history, fetch_edgar_statements
            edgar = fetch_edgar_statements(ticker, quarterly=quarterly)
            for k in statement_keys:
                before = len(result[k]) if isinstance(result[k], pd.DataFrame) else 0
                result[k] = extend_history(result[k], edgar.get(k))
                after = len(result[k])
                if after > before:
                    logger.info("  [OK] %-10s extended via EDGAR: %d → %d periods.",
                                k, before, after)
        except Exception as exc:
            logger.warning("EDGAR history extension skipped for '%s': %s", ticker, exc)

    logger.info("=== Fetch complete for '%s' ===", ticker)
    return result
