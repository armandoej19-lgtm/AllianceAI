"""Unit tests for risk signal detection."""

import numpy as np
import pandas as pd

from allianceai.analysis.risk_signals import detect_all_signals


def _idx(n: int):
    return pd.date_range("2022-01-01", periods=n, freq="QS")


def test_no_signals_for_healthy_firm():
    idx = _idx(6)
    income = pd.DataFrame({
        "Total Revenue": [1000] * 6, "Net Income": [100] * 6,
        "Operating Income": [150] * 6, "EBITDA": [200] * 6,
        "Interest Expense": [10] * 6,
    }, index=idx)
    balance = pd.DataFrame({
        "Current Assets": [300] * 6, "Current Liabilities": [100] * 6,
        "Total Assets": [800] * 6, "Stockholders Equity": [500] * 6,
        "Total Debt": [100] * 6, "Cash And Cash Equivalents": [80] * 6,
        "Total Liabilities Net Minority Interest": [300] * 6,
    }, index=idx)
    cashflow = pd.DataFrame({
        "Operating Cash Flow": [150] * 6, "Free Cash Flow": [120] * 6,
        "Capital Expenditure": [-30] * 6, "Common Stock Issuance": [5] * 6,
    }, index=idx)
    signals = detect_all_signals(income, balance, cashflow)
    critical = [s for s in signals if s.severity == "CRITICAL"]
    assert len(critical) == 0, f"No CRITICAL signals expected for a healthy firm, got: {critical}"


def test_critical_liquidity_signal():
    idx = _idx(4)
    balance = pd.DataFrame({
        "Current Assets": [50] * 4, "Current Liabilities": [200] * 4,
        "Total Assets": [300] * 4, "Stockholders Equity": [100] * 4,
        "Total Debt": [150] * 4, "Cash And Cash Equivalents": [10] * 4,
        "Total Liabilities Net Minority Interest": [200] * 4,
    }, index=idx)
    income = pd.DataFrame({"Total Revenue": [500] * 4, "Net Income": [10] * 4,
                            "EBITDA": [50] * 4, "Operating Income": [30] * 4,
                            "Interest Expense": [15] * 4}, index=idx)
    cashflow = pd.DataFrame({"Operating Cash Flow": [30] * 4, "Free Cash Flow": [20] * 4,
                              "Capital Expenditure": [-10] * 4}, index=idx)
    signals = detect_all_signals(income, balance, cashflow)
    names = [s.name for s in signals]
    assert "Current Ratio Below 1" in names


def test_extreme_leverage_signal():
    idx = _idx(4)
    balance = pd.DataFrame({
        "Current Assets": [200] * 4, "Current Liabilities": [100] * 4,
        "Total Assets": [600] * 4, "Stockholders Equity": [50] * 4,
        "Total Debt": [500] * 4, "Cash And Cash Equivalents": [20] * 4,
        "Total Liabilities Net Minority Interest": [550] * 4,
    }, index=idx)
    income = pd.DataFrame({"Total Revenue": [500] * 4, "Net Income": [20] * 4,
                            "EBITDA": [80] * 4, "Operating Income": [60] * 4,
                            "Interest Expense": [40] * 4}, index=idx)
    cashflow = pd.DataFrame({"Operating Cash Flow": [60] * 4, "Free Cash Flow": [40] * 4,
                              "Capital Expenditure": [-20] * 4}, index=idx)
    signals = detect_all_signals(income, balance, cashflow)
    names = [s.name for s in signals]
    assert "Extreme Debt-to-Equity" in names


def test_signals_sorted_by_severity():
    idx = _idx(4)
    balance = pd.DataFrame({
        "Current Assets": [50] * 4, "Current Liabilities": [200] * 4,
        "Total Assets": [300] * 4, "Stockholders Equity": [50] * 4,
        "Total Debt": [500] * 4, "Cash And Cash Equivalents": [5] * 4,
        "Total Liabilities Net Minority Interest": [250] * 4,
    }, index=idx)
    income = pd.DataFrame({"Total Revenue": [200] * 4, "Net Income": [-50] * 4,
                            "EBITDA": [10] * 4, "Operating Income": [-30] * 4,
                            "Interest Expense": [30] * 4}, index=idx)
    cashflow = pd.DataFrame({"Operating Cash Flow": [-40] * 4, "Free Cash Flow": [-60] * 4,
                              "Capital Expenditure": [-20] * 4}, index=idx)
    signals = detect_all_signals(income, balance, cashflow)
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    severities = [order[s.severity] for s in signals]
    assert severities == sorted(severities), "Signals must be sorted CRITICAL → LOW."
