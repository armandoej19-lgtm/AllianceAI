"""
Analysis confidence — a global data-quality badge for the whole report.

Distinct from the decision's confidence (which rates the *verdict*), this rates
the *inputs*: how complete and deep the underlying data is. A verdict computed
from 5 sparse quarters deserves a visible caveat regardless of how the numbers
came out.

Drivers:
  - HISTORY    : how many periods of statements are available (more = better).
  - COMPLETENESS: fraction of the key metrics actually present (not NaN).
  - SOURCES    : whether deep SEC EDGAR history backed yfinance, and whether
                 price data is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)

# The metrics every downstream model leans on — completeness is measured here.
_KEY_METRICS = {
    "income": ["Total Revenue", "Net Income", "Operating Income", "EBITDA"],
    "balance": ["Total Assets", "Total Liabilities Net Minority Interest",
                "Stockholders Equity", "Current Assets", "Current Liabilities"],
    "cashflow": ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure"],
}


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _assess_price_only(prices: pd.DataFrame | None) -> dict:
    """
    Confidence for funds / index trackers, which file no statements. Rated on
    the price series instead: depth of history, recency, and bar coverage.
    """
    drivers: dict[str, float] = {}
    notes: list[str] = ["Fund/index — rated on price data (no financial statements apply)."]

    if prices is None or prices.empty or "Close" not in prices.columns:
        return {"score": 0.0, "level": "LOW", "drivers": {},
                "notes": ["No price history available."]}

    close = prices["Close"].dropna()
    idx = pd.to_datetime(close.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)

    years = (idx[-1] - idx[0]).days / 365.25
    drivers["history"] = _clip01(years / 5) * 100          # ~5y saturates
    notes.append(f"{years:.1f} years of price history ({len(close)} daily bars).")

    days_old = (pd.Timestamp.now().normalize() - idx[-1].normalize()).days
    drivers["recency"] = 100.0 if days_old <= 4 else 60.0 if days_old <= 10 else 25.0
    notes.append(f"Most recent price is {days_old} day(s) old.")

    expected_bars = max(years * 252, 1)
    drivers["coverage"] = _clip01(len(close) / expected_bars) * 100

    score = (0.60 * drivers["history"]
             + 0.20 * drivers["recency"]
             + 0.20 * drivers["coverage"])
    level = "HIGH" if score >= 67 else "MODERATE" if score >= 40 else "LOW"

    logger.info("Analysis data confidence (price-only): %s (%.0f/100).", level, score)
    return {
        "score": round(score, 1),
        "level": level,
        "drivers": {k: round(v, 1) for k, v in drivers.items()},
        "notes": notes,
    }


def assess_data_confidence(
    income: pd.DataFrame | None,
    balance: pd.DataFrame | None,
    cashflow: pd.DataFrame | None,
    prices: pd.DataFrame | None = None,
    edgar_extended: bool = False,
    price_only: bool = False,
) -> dict:
    """Return {score, level, drivers, notes} rating overall input quality.

    For funds/index trackers (no financial statements), pass ``price_only=True``
    to score on the price series instead of statement coverage — otherwise the
    statement-based rubric pins every fund at a misleading ~13/100.
    """
    if price_only:
        return _assess_price_only(prices)

    drivers: dict[str, float] = {}
    notes: list[str] = []
    frames = {"income": income, "balance": balance, "cashflow": cashflow}

    # History — deepest statement, saturating at 24 periods (~6 years quarterly).
    periods = max((len(df) for df in frames.values() if df is not None and not df.empty),
                  default=0)
    drivers["history"] = _clip01(periods / 24) * 100
    notes.append(f"{periods} periods of statement history available.")

    # Completeness — of the key metrics, how many are present and non-empty.
    present, total = 0, 0
    for name, cols in _KEY_METRICS.items():
        df = frames[name]
        for col in cols:
            total += 1
            if df is not None and not df.empty and col in df.columns and df[col].notna().any():
                present += 1
    drivers["completeness"] = (present / total * 100) if total else 0.0
    notes.append(f"{present}/{total} key metrics present.")

    # Sources — EDGAR deep history + price data each lift trust.
    sources = 0.5
    if edgar_extended:
        sources += 0.35
        notes.append("Extended with SEC EDGAR deep history.")
    if prices is not None and not prices.empty:
        sources += 0.15
    drivers["sources"] = _clip01(sources) * 100

    score = (0.40 * drivers["history"]
             + 0.40 * drivers["completeness"]
             + 0.20 * drivers["sources"])
    level = "HIGH" if score >= 67 else "MODERATE" if score >= 40 else "LOW"

    logger.info("Analysis data confidence: %s (%.0f/100).", level, score)
    return {
        "score": round(score, 1),
        "level": level,
        "drivers": {k: round(v, 1) for k, v in drivers.items()},
        "notes": notes,
    }
