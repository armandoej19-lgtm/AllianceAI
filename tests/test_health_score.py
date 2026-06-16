"""Unit tests for health scoring and Altman Z-Score."""

import numpy as np
import pandas as pd
import pytest

from allianceai.analysis.health_score import altman_z_score, compute_health_score


def _make_healthy():
    idx = pd.date_range("2022-01-01", periods=6, freq="QS")
    income = pd.DataFrame({
        "Total Revenue":    [1000] * 6,
        "Net Income":       [120]  * 6,
        "Operating Income": [180]  * 6,
        "EBITDA":           [220]  * 6,
        "Interest Expense": [15]   * 6,
    }, index=idx)
    balance = pd.DataFrame({
        "Current Assets":      [300] * 6,
        "Current Liabilities": [100] * 6,
        "Total Assets":        [800] * 6,
        "Total Liabilities Net Minority Interest": [400] * 6,
        "Stockholders Equity": [400] * 6,
        "Total Debt":          [200] * 6,
        "Cash And Cash Equivalents": [80] * 6,
        "Retained Earnings":   [150] * 6,
    }, index=idx)
    cashflow = pd.DataFrame({
        "Operating Cash Flow": [180] * 6,
        "Free Cash Flow":      [130] * 6,
        "Capital Expenditure": [-50] * 6,
    }, index=idx)
    return income, balance, cashflow


def test_health_score_range():
    income, balance, cashflow = _make_healthy()
    health = compute_health_score(income, balance, cashflow)
    assert "health_score" in health.columns
    # All scores must be in [0, 100].
    assert (health["health_score"] >= 0).all() and (health["health_score"] <= 100).all()


def test_healthy_firm_scores_above_60():
    income, balance, cashflow = _make_healthy()
    health = compute_health_score(income, balance, cashflow)
    assert health["health_score"].mean() > 60, "A healthy firm should score above 60."


def test_distressed_firm_scores_below_40():
    idx = pd.date_range("2022-01-01", periods=4, freq="QS")
    income = pd.DataFrame({
        "Total Revenue": [500] * 4, "Net Income": [-200] * 4,
        "Operating Income": [-180] * 4, "EBITDA": [-150] * 4,
        "Interest Expense": [60] * 4,
    }, index=idx)
    balance = pd.DataFrame({
        "Current Assets": [50] * 4, "Current Liabilities": [200] * 4,
        "Total Assets": [300] * 4, "Total Liabilities Net Minority Interest": [400] * 4,
        "Stockholders Equity": [-100] * 4, "Total Debt": [350] * 4,
        "Cash And Cash Equivalents": [10] * 4, "Retained Earnings": [-200] * 4,
    }, index=idx)
    cashflow = pd.DataFrame({
        "Operating Cash Flow": [-150] * 4, "Free Cash Flow": [-180] * 4,
        "Capital Expenditure": [-30] * 4,
    }, index=idx)
    health = compute_health_score(income, balance, cashflow)
    assert health["health_score"].mean() < 40, "A distressed firm should score below 40."


def test_altman_z_produces_zone():
    income, balance, _ = _make_healthy()
    z = altman_z_score(income, balance)
    assert "z_score" in z.columns
    assert "z_zone" in z.columns
    assert set(z["z_zone"].dropna()).issubset({"safe", "grey", "distress"})
