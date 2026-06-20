"""
Price-based analytics for funds and index trackers (ETF/MUTUALFUND/INDEX).

These instruments file no financial statements, so the statement-driven
pipeline (ratios, health, distress, factors) does not apply. Instead we
characterise them from their price series and the fund metadata yfinance
exposes: trailing returns, annualised volatility, drawdown, risk-adjusted
return, trend, and headline fund facts (expense ratio, AUM, yield).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)

_TRADING_DAYS = 252


def _to_naive(idx: pd.Index) -> pd.DatetimeIndex:
    """Return a tz-naive DatetimeIndex (price indices are often tz-aware)."""
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx


def compute_price_metrics(prices: pd.DataFrame, info: dict | None = None) -> dict:
    """
    Compute performance / risk / level metrics from an OHLCV price frame plus
    headline fund facts from *info*. Returns {} when prices are unusable.
    """
    info = info or {}
    if prices is None or prices.empty or "Close" not in prices.columns:
        return {}

    close = prices["Close"].dropna()
    if close.empty:
        return {}
    close = close.copy()
    close.index = _to_naive(close.index)
    close.sort_index(inplace=True)

    last = float(close.iloc[-1])
    last_date = close.index[-1]

    # --- Trailing returns (point-to-point, % change) ---
    def ret_since(cutoff: pd.Timestamp):
        past = close[close.index <= cutoff]
        if past.empty:
            return None
        base = float(past.iloc[-1])
        return (last / base - 1) * 100 if base else None

    perf = {
        "1M": ret_since(last_date - pd.Timedelta(days=30)),
        "3M": ret_since(last_date - pd.Timedelta(days=91)),
        "6M": ret_since(last_date - pd.Timedelta(days=182)),
        "YTD": ret_since(pd.Timestamp(year=last_date.year, month=1, day=1)),
        "1Y": ret_since(last_date - pd.Timedelta(days=365)),
        "3Y": ret_since(last_date - pd.Timedelta(days=365 * 3)),
        "5Y": ret_since(last_date - pd.Timedelta(days=365 * 5)),
    }

    # --- CAGR over the full available span ---
    span_years = (last_date - close.index[0]).days / 365.25
    first = float(close.iloc[0])
    cagr = ((last / first) ** (1 / span_years) - 1) * 100 if span_years >= 0.5 and first > 0 else None

    # --- Risk (from daily returns) ---
    rets = close.pct_change().dropna()
    vol = float(rets.std() * np.sqrt(_TRADING_DAYS) * 100) if not rets.empty else None
    ann_ret = float(rets.mean() * _TRADING_DAYS) if not rets.empty else None
    sharpe = (ann_ret / (rets.std() * np.sqrt(_TRADING_DAYS))
              if not rets.empty and rets.std() > 0 else None)

    drawdown = close / close.cummax() - 1
    max_dd = float(drawdown.min() * 100) if not drawdown.empty else None

    # --- Levels / trend ---
    last_year = close[close.index >= last_date - pd.Timedelta(days=365)]
    high_52w = float(last_year.max()) if not last_year.empty else None
    low_52w = float(last_year.min()) if not last_year.empty else None
    pct_from_high = (last / high_52w - 1) * 100 if high_52w else None
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    above_200dma = (last > ma200) if ma200 is not None else None

    # --- Fund metadata (units vary across yfinance fields — see _fmt in report) ---
    inception = info.get("fundInceptionDate")
    if isinstance(inception, (int, float)):
        try:
            inception = pd.to_datetime(inception, unit="s").strftime("%Y-%m-%d")
        except Exception:
            inception = None

    def _clean(v):
        """Coerce yfinance NaN/empty values to None for clean rendering."""
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    fund = {
        "category": _clean(info.get("category")),
        "family": _clean(info.get("fundFamily")),
        "legal_type": _clean(info.get("legalType")),
        "expense_ratio": _clean(info.get("netExpenseRatio") or info.get("annualReportExpenseRatio")),
        "aum": _clean(info.get("totalAssets")),
        "yield": _clean(info.get("yield")),         # fractional (0.0042 = 0.42%)
        "beta": _clean(info.get("beta3Year") or info.get("beta")),
        "inception": inception,
    }

    metrics = {
        "as_of": last_date.strftime("%Y-%m-%d"),
        "performance": {k: (round(v, 2) if v is not None else None) for k, v in perf.items()},
        "cagr": round(cagr, 2) if cagr is not None else None,
        "risk": {
            "volatility": round(vol, 2) if vol is not None else None,
            "max_drawdown": round(max_dd, 2) if max_dd is not None else None,
            "sharpe": round(sharpe, 2) if sharpe is not None else None,
        },
        "levels": {
            "price": round(last, 2),
            "high_52w": round(high_52w, 2) if high_52w is not None else None,
            "low_52w": round(low_52w, 2) if low_52w is not None else None,
            "pct_from_high": round(pct_from_high, 2) if pct_from_high is not None else None,
            "above_200dma": above_200dma,
        },
        "fund": fund,
    }
    logger.info("Price metrics computed: 1Y=%s%%, vol=%s%%, maxDD=%s%%.",
                metrics["performance"]["1Y"], metrics["risk"]["volatility"],
                metrics["risk"]["max_drawdown"])
    return metrics
