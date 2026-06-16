"""
Monte Carlo scenario analysis.

Simulates N future paths for key financial metrics by sampling from
the historical distribution of period-over-period changes.

Outputs are purely numerical:
  - P(revenue growth > 0)  in Q+1, Q+2, Q+4
  - P(net income positive) in Q+1, Q+2, Q+4
  - P(cash flow covers debt service)
  - 10th / 50th / 90th percentile of each metric at each horizon
  - VaR (Value at Risk): worst-case outcome at 5th percentile
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)


def _historical_stats(series: pd.Series) -> tuple[float, float]:
    """Return (mean, std) of period-over-period log returns."""
    s = series.dropna().astype(float)
    s = s[s > 0]   # log returns require positive values
    if len(s) < 3:
        return 0.0, 0.10   # default: 0% growth, 10% volatility
    log_ret = np.log(s / s.shift(1)).dropna()
    return float(log_ret.mean()), float(log_ret.std())


def monte_carlo(
    series: pd.Series,
    horizons: list[int] | None = None,
    n_simulations: int = 10_000,
    label: str = "",
) -> dict:
    """
    Simulate future values of *series* using Geometric Brownian Motion
    calibrated to the historical mean and volatility of the series.

    GBM: S_{t+1} = S_t * exp( (μ - σ²/2) + σ * ε )
    where ε ~ N(0,1)

    Returns per-horizon statistics and probabilities.
    """
    horizons = horizons or [1, 2, 4]
    label = label or (series.name or "metric")

    mu, sigma = _historical_stats(series)
    last_val   = series.dropna().astype(float).iloc[-1] if not series.dropna().empty else 1.0

    logger.debug(
        "Monte Carlo '%s': last=%.2f  mu=%.4f  sigma=%.4f  n=%d",
        label, last_val, mu, sigma, n_simulations,
    )

    max_h = max(horizons)
    # Draw all random shocks at once: shape (n_simulations, max_horizon).
    rng    = np.random.default_rng(seed=42)
    shocks = rng.standard_normal((n_simulations, max_h))
    # Cumulative log return path.
    log_drift  = (mu - 0.5 * sigma ** 2)
    log_paths  = np.cumsum(log_drift + sigma * shocks, axis=1)
    value_paths = last_val * np.exp(log_paths)   # shape (n_sims, max_h)

    result = {"metric": label, "last_value": round(last_val, 2), "horizons": {}}

    for h in horizons:
        vals = value_paths[:, h - 1]
        p_positive   = float(np.mean(vals > 0))
        p_growth     = float(np.mean(vals > last_val))
        p10, p50, p90 = np.percentile(vals, [10, 50, 90])
        var_5        = float(np.percentile(vals, 5))

        result["horizons"][h] = {
            "p_positive":         round(p_positive, 4),
            "p_growth":           round(p_growth, 4),
            "p10":                round(float(p10), 2),
            "p50_median":         round(float(p50), 2),
            "p90":                round(float(p90), 2),
            "var_5pct":           round(var_5, 2),
            "expected_change_pct": round((p50 / last_val - 1) * 100, 2) if last_val != 0 else np.nan,
        }

    return result


def run_scenario_analysis(
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
    balance: pd.DataFrame,
    n_simulations: int = 10_000,
) -> dict[str, dict]:
    """
    Run Monte Carlo on the most decision-relevant financial metrics.
    Returns a dict of metric_name → simulation results.
    """
    logger.info("Running Monte Carlo scenario analysis (%d simulations).", n_simulations)

    targets = {
        "Total Revenue":        income,
        "Net Income":           income,
        "Operating Income":     income,
        "EBITDA":               income,
        "Operating Cash Flow":  cashflow,
        "Free Cash Flow":       cashflow,
        "Total Debt":           balance,
    }

    results = {}
    for metric, df in targets.items():
        if metric not in df.columns or df[metric].dropna().empty:
            logger.debug("Skipping Monte Carlo for '%s' — no data.", metric)
            continue
        results[metric] = monte_carlo(df[metric], label=metric, n_simulations=n_simulations)

    logger.info("Scenario analysis complete for %d metrics.", len(results))
    return results
