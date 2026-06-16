"""
AllianceAI Orchestrator — end-to-end quantitative analysis pipeline.

Call `analyse(ticker)` to run the full pipeline:
  1.  Fetch all financial data (with DuckDB caching).
  2.  Compute financial ratios.
  3.  Run EDA (trends, outliers, completeness).
  4.  Score financial health (AllianceAI composite + Altman Z).
  5.  Detect risk signals.
  6.  Detect anomalies with IsolationForest.
  7.  Distress probability ensemble (Ohlson, Zmijewski, Altman).
  8.  Factor analysis (predictive correlations, PCA, income decomposition).
  9.  Monte Carlo scenario analysis.
  10. Forecast key metrics (Prophet / Holt-Winters) with statistical output.
  11. Render and save the HTML report.

Returns a structured dict so callers (notebooks, APIs, CLI) can access any
intermediate result programmatically.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd

from allianceai.analysis.eda import run_full_eda
from allianceai.analysis.health_score import altman_z_score, compute_health_score
from allianceai.analysis.ratios import compute_all_ratios
from allianceai.analysis.risk_signals import detect_all_signals
from allianceai.core.logging_config import get_logger
from allianceai.data.fetcher import fetch_all
from allianceai.models.anomaly import detect_anomalies
from allianceai.models.distress import compute_distress_ensemble
from allianceai.models.factors import run_factor_analysis
from allianceai.models.forecaster import forecast_metric, forecast_statements
from allianceai.models.scenarios import run_scenario_analysis
from allianceai.reports.html_report import generate_html_report

logger = get_logger(__name__)


def analyse(
    ticker: str,
    quarterly: bool = True,
    output_dir: str | Path = ".",
    skip_forecast: bool = False,
    skip_report: bool = False,
) -> dict:
    """
    Run the complete AllianceAI quantitative analysis pipeline for *ticker*.

    Parameters
    ----------
    ticker       : Yahoo Finance ticker symbol (e.g. 'AAPL', 'VNQ', 'SPY').
    quarterly    : If True, use quarterly statements; else annual.
    output_dir   : Directory where the HTML report is saved.
    skip_forecast: Set True to skip Prophet/HW forecasting (faster).
    skip_report  : Set True to skip HTML report generation.

    Returns
    -------
    Dict with keys: ticker, info, income, balance, cashflow, prices,
    ratios, eda, health, altman, signals, anomalies,
    distress, factors, scenarios, forecasts, report_path.
    """
    ticker = ticker.upper().strip()
    logger.info("=" * 60)
    logger.info("AllianceAI analysis started for '%s'.", ticker)
    logger.info("=" * 60)

    result: dict = {"ticker": ticker}

    # ------------------------------------------------------------------
    # 1. Data fetching
    # ------------------------------------------------------------------
    data = fetch_all(ticker, quarterly=quarterly)
    info     = data["info"]
    income   = data["income"]
    balance  = data["balance"]
    cashflow = data["cashflow"]
    prices   = data["prices"]

    result.update(info=info, income=income, balance=balance, cashflow=cashflow, prices=prices)

    # Overall data-quality confidence for the whole analysis (global badge).
    # yfinance alone returns ~5-7 periods; more implies EDGAR deep history.
    try:
        from allianceai.analysis.data_confidence import assess_data_confidence
        max_periods = max((len(df) for df in (income, balance, cashflow)
                           if df is not None and not df.empty), default=0)
        result["data_confidence"] = assess_data_confidence(
            income, balance, cashflow, prices, edgar_extended=max_periods > 10)
    except Exception:
        logger.error("Data confidence assessment failed:\n%s", traceback.format_exc())
        result["data_confidence"] = None

    # ETFs, index funds, and money-market funds don't file income statements.
    quote_type = (info.get("quoteType") or "").upper()
    is_fund = quote_type in ("ETF", "MUTUALFUND", "INDEX", "MONEYMARKET")

    if is_fund:
        logger.info(
            "'%s' is a %s — skipping statement analysis, running price-only mode.",
            ticker, quote_type,
        )
        result["mode"] = "price_only"
        result.update(
            ratios={}, eda={}, health=pd.DataFrame(), altman=pd.DataFrame(),
            signals=[], anomalies=pd.DataFrame(),
            distress=pd.DataFrame(), factors={}, scenarios={},
        )
        if not prices.empty and not skip_forecast:
            try:
                result["forecasts"] = {"prices": {"Close": forecast_metric(prices["Close"], label="Close")}}
                logger.info("[OK] Price forecast complete.")
            except Exception:
                logger.error("Price forecast failed:\n%s", traceback.format_exc())
                result["forecasts"] = {}
        else:
            result["forecasts"] = {}
        result["report_path"] = None
        logger.info("Price-only analysis complete for '%s'.", ticker)
        return result

    if income.empty and balance.empty and cashflow.empty:
        logger.error("All financial statements are empty for '%s' — aborting.", ticker)
        result["error"] = "No financial data available."
        return result

    # ------------------------------------------------------------------
    # 1b. Prediction feedback: grade past forecasts against the official
    #     data just fetched. Errors become the calibration training set.
    # ------------------------------------------------------------------
    from allianceai.models.prediction_loop import (
        accuracy_table, calibrate_forecasts, evaluate_predictions, save_forecast_seeds,
    )
    try:
        newly_graded = evaluate_predictions(
            ticker, {"income": income, "balance": balance, "cashflow": cashflow})
        if not newly_graded.empty:
            logger.info("[OK] %d past prediction(s) graded against official data.",
                        len(newly_graded))
        result["prediction_accuracy"] = accuracy_table(ticker)
    except Exception:
        logger.error("Prediction evaluation failed:\n%s", traceback.format_exc())
        result["prediction_accuracy"] = pd.DataFrame()

    # ------------------------------------------------------------------
    # 2. Financial ratios
    # ------------------------------------------------------------------
    try:
        ratios = compute_all_ratios(income, balance, cashflow)
        result["ratios"] = ratios
        logger.info("[OK] Ratios computed.")
    except Exception:
        logger.error("Ratio computation failed:\n%s", traceback.format_exc())
        result["ratios"] = {}
        ratios = {}

    # ------------------------------------------------------------------
    # 3. EDA
    # ------------------------------------------------------------------
    try:
        eda = run_full_eda(income, balance, cashflow)
        result["eda"] = eda
        logger.info("[OK] EDA complete.")
    except Exception:
        logger.error("EDA failed:\n%s", traceback.format_exc())
        result["eda"] = {}

    # ------------------------------------------------------------------
    # 4. Health scoring
    # ------------------------------------------------------------------
    # Sectors where the original (manufacturing-calibrated) Z applies; everyone
    # else gets the Z''-Score variant, which drops the asset-turnover term.
    _MANUFACTURING_SECTORS = {"Industrials", "Basic Materials", "Consumer Cyclical",
                              "Consumer Defensive", "Energy"}
    non_mfg = (info.get("sector") or "") not in _MANUFACTURING_SECTORS
    market_cap = info.get("marketCap")

    try:
        health = compute_health_score(income, balance, cashflow)
        altman = altman_z_score(income, balance, market_cap=market_cap,
                                non_manufacturing=non_mfg)
        result.update(health=health, altman=altman)
        logger.info("[OK] Health scores computed.")
    except Exception:
        logger.error("Health scoring failed:\n%s", traceback.format_exc())
        health = pd.DataFrame()
        altman = pd.DataFrame()
        result.update(health=health, altman=altman)

    # ------------------------------------------------------------------
    # 5. Risk signals
    # ------------------------------------------------------------------
    try:
        signals = detect_all_signals(income, balance, cashflow)
        result["signals"] = signals
        logger.info("[OK] Risk signals: %d detected.", len(signals))
    except Exception:
        logger.error("Risk signal detection failed:\n%s", traceback.format_exc())
        result["signals"] = []
        signals = []

    # ------------------------------------------------------------------
    # 6. Anomaly detection
    # ------------------------------------------------------------------
    try:
        if not income.empty:
            result["anomalies"] = detect_anomalies(income)
            logger.info("[OK] Anomaly detection complete.")
        else:
            result["anomalies"] = pd.DataFrame()
    except Exception:
        logger.error("Anomaly detection failed:\n%s", traceback.format_exc())
        result["anomalies"] = pd.DataFrame()

    # ------------------------------------------------------------------
    # 7. Distress probability ensemble
    # ------------------------------------------------------------------
    try:
        distress = compute_distress_ensemble(income, balance, cashflow,
                                             market_cap=market_cap,
                                             non_manufacturing=non_mfg)
        result["distress"] = distress
        logger.info("[OK] Distress ensemble computed.")
    except Exception:
        logger.error("Distress computation failed:\n%s", traceback.format_exc())
        result["distress"] = pd.DataFrame()
        distress = pd.DataFrame()

    # ------------------------------------------------------------------
    # 7b. LLM highlights (Claude API; rule-based fallback without a key)
    # ------------------------------------------------------------------
    try:
        from allianceai.analysis.highlights import generate_highlights
        highlights = generate_highlights(
            info=info, health=health, altman=altman, distress=distress,
            signals=signals, income=income, balance=balance, cashflow=cashflow,
        )
        result["highlights"] = highlights
        logger.info("[OK] %d highlights generated.", len(highlights))
    except Exception:
        logger.error("Highlight generation failed:\n%s", traceback.format_exc())
        result["highlights"] = []

    # ------------------------------------------------------------------
    # 8. Factor analysis
    # ------------------------------------------------------------------
    try:
        factors = run_factor_analysis(income, balance, cashflow, ratios)
        result["factors"] = factors
        logger.info("[OK] Factor analysis complete.")
    except Exception:
        logger.error("Factor analysis failed:\n%s", traceback.format_exc())
        result["factors"] = {}
        factors = {}

    # ------------------------------------------------------------------
    # 9. Monte Carlo scenario analysis
    # ------------------------------------------------------------------
    try:
        scenarios = run_scenario_analysis(income, cashflow, balance)
        result["scenarios"] = scenarios
        logger.info("[OK] Scenario analysis complete.")
    except Exception:
        logger.error("Scenario analysis failed:\n%s", traceback.format_exc())
        result["scenarios"] = {}
        scenarios = {}

    # ------------------------------------------------------------------
    # 9b. Opportunity highlights (probability expectations from Monte Carlo)
    # ------------------------------------------------------------------
    try:
        from allianceai.analysis.highlights import generate_opportunities
        opportunities = generate_opportunities(scenarios)
        result["opportunities"] = opportunities
        logger.info("[OK] %d opportunity highlights generated.", len(opportunities))
    except Exception:
        logger.error("Opportunity highlights failed:\n%s", traceback.format_exc())
        result["opportunities"] = []

    # ------------------------------------------------------------------
    # 10. Forecasting
    # ------------------------------------------------------------------
    if not skip_forecast:
        try:
            forecasts = forecast_statements(income, balance, cashflow)
            # Learn from the ticker's own past forecast errors, then store
            # today's forecasts as seeds for the next run to grade.
            try:
                forecasts = calibrate_forecasts(ticker, forecasts)
                n_seeds = save_forecast_seeds(ticker, forecasts)
                logger.info("[OK] %d forecast seed(s) stored for future evaluation.", n_seeds)
            except Exception:
                logger.error("Prediction seeding failed:\n%s", traceback.format_exc())
            result["forecasts"] = forecasts
            logger.info("[OK] Forecasting complete.")
        except Exception:
            logger.error("Forecasting failed:\n%s", traceback.format_exc())
            result["forecasts"] = {}
    else:
        result["forecasts"] = {}
        logger.info("Forecasting skipped.")

    # ------------------------------------------------------------------
    # 10b. Investment-stance decision (health × growth × probabilities)
    # ------------------------------------------------------------------
    try:
        from allianceai.analysis.decision import make_decision
        decision = make_decision(health=health, distress=distress, altman=altman,
                                 scenarios=scenarios, income=income,
                                 prediction_accuracy=result.get("prediction_accuracy"))
        result["decision"] = decision
        logger.info("[OK] Decision: %s (%.0f/100).", decision["verdict"], decision["score"])
    except Exception:
        logger.error("Decision engine failed:\n%s", traceback.format_exc())
        result["decision"] = None

    # ------------------------------------------------------------------
    # 11. HTML report
    # ------------------------------------------------------------------
    if not skip_report:
        try:
            out_path = Path(output_dir) / f"{ticker}_report.html"
            generate_html_report(
                ticker=ticker,
                info=info,
                health=health,
                signals=signals,
                altman=altman if not altman.empty else None,
                distress=distress if not distress.empty else None,
                factors=factors,
                scenarios=scenarios,
                forecasts=result["forecasts"],
                highlights=result.get("highlights"),
                opportunities=result.get("opportunities"),
                decision=result.get("decision"),
                prediction_accuracy=result.get("prediction_accuracy"),
                data_confidence=result.get("data_confidence"),
                income=income if not income.empty else None,
                cashflow=cashflow if not cashflow.empty else None,
                balance=balance if not balance.empty else None,
                output_path=out_path,
            )
            result["report_path"] = str(out_path.resolve())
            logger.info("[OK] Report saved to '%s'.", result["report_path"])
        except Exception:
            logger.error("Report generation failed:\n%s", traceback.format_exc())
            result["report_path"] = None
    else:
        result["report_path"] = None

    logger.info("=" * 60)
    logger.info("AllianceAI analysis complete for '%s'.", ticker)
    logger.info("=" * 60)
    return result
