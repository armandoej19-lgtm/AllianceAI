"""
Unit tests for the ratio computation module.
These tests use synthetic DataFrames so no network access is required.
"""

import numpy as np
import pandas as pd
import pytest

from allianceai.analysis.ratios import (
    compute_liquidity_ratios,
    compute_leverage_ratios,
    compute_profitability_ratios,
)


def _make_balance(n: int = 4) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="QS")
    return pd.DataFrame({
        "Current Assets":                       [120, 130, 140, 150],
        "Current Liabilities":                  [80,  90, 100,  80],
        "Total Assets":                         [500, 520, 540, 560],
        "Total Liabilities Net Minority Interest": [300, 310, 320, 330],
        "Stockholders Equity":                  [200, 210, 220, 230],
        "Total Debt":                           [150, 160, 170, 180],
        "Cash And Cash Equivalents":            [50,  55,  60,  65],
    }, index=idx)


def _make_income(n: int = 4) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="QS")
    return pd.DataFrame({
        "Total Revenue":    [1000, 1050, 1100, 1150][:n],
        "Gross Profit":     [400,  430,  450,  470][:n],
        "Operating Income": [200,  210,  220,  230][:n],
        "Net Income":       [100,  110,  115,  120][:n],
        "EBITDA":           [250,  260,  270,  280][:n],
        "Interest Expense": [20,   20,   22,   22][:n],
    }, index=idx)


def test_current_ratio():
    balance = _make_balance()
    liq = compute_liquidity_ratios(balance)
    assert "current_ratio" in liq.columns
    # 150 / 80 = 1.875 for last period
    assert abs(liq["current_ratio"].iloc[-1] - 1.875) < 0.01


def test_cash_ratio():
    balance = _make_balance()
    liq = compute_liquidity_ratios(balance)
    # 65 / 80 ≈ 0.8125
    assert abs(liq["cash_ratio"].iloc[-1] - 0.8125) < 0.01


def test_debt_to_equity():
    balance = _make_balance()
    income  = _make_income()
    lev = compute_leverage_ratios(balance, income)
    # 180 / 230 ≈ 0.783
    assert abs(lev["debt_to_equity"].iloc[-1] - (180 / 230)) < 0.01


def test_zero_denominator_returns_nan():
    """Dividing by zero equity should produce NaN, not raise."""
    idx = pd.date_range("2023-01-01", periods=2, freq="QS")
    balance = pd.DataFrame({
        "Current Assets":      [100, 100],
        "Current Liabilities": [0,   0],   # zero → NaN, not inf
        "Total Assets":        [200, 200],
        "Stockholders Equity": [0,   0],
        "Total Debt":          [100, 100],
        "Cash And Cash Equivalents": [20, 20],
        "Total Liabilities Net Minority Interest": [150, 150],
    }, index=idx)
    income = _make_income(2)
    liq = compute_liquidity_ratios(balance)
    lev = compute_leverage_ratios(balance, income)
    # Infinite current ratio → NaN (current liabilities = 0)
    assert liq["cash_ratio"].isna().all()


def test_profitability_margins():
    income  = _make_income()
    balance = _make_balance()
    prof = compute_profitability_ratios(income, balance)
    # net margin = 120/1150 ≈ 10.4%
    assert abs(prof["net_margin"].iloc[-1] - 120 / 1150) < 0.001


def test_missing_column_fills_nan():
    """A statement missing a column should not raise; the ratio should be NaN."""
    idx = pd.date_range("2023-01-01", periods=4, freq="QS")
    # Balance sheet without 'Retained Earnings'
    balance = pd.DataFrame({
        "Total Assets":      [500] * 4,
        "Current Assets":    [200] * 4,
        "Current Liabilities": [100] * 4,
    }, index=idx)
    income = _make_income()
    liq = compute_liquidity_ratios(balance)
    assert "current_ratio" in liq.columns
    assert not liq["current_ratio"].isna().all()
