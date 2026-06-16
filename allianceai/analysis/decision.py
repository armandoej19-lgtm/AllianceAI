"""
Investment-stance decision engine.

Frames the analysis the way an investor risking their own long-saved equity
would: is this company healthy enough, growing enough, and statistically
likely enough to compound — and is NOW the time, or is waiting free?

The verdict is a transparent weighted composite (every input and weight is
visible in the rationale), not a black box, and explicitly NOT financial
advice — it's a structured summary of what the quantitative pipeline found.

Timing philosophy encoded here:
  - A healthy company that is actually growing rewards acting sooner
    (waiting has a real opportunity cost).
  - A healthy company with stagnant growth costs nothing to watch — the next
    quarterly filing is free information.
  - A weak or distressed company is only worth risk when the upside
    probabilities are unusually strong AND the balance sheet can survive
    being wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)

_WEIGHTS = {
    "health":   0.30,   # current financial health (0-100 composite)
    "growth":   0.25,   # trailing revenue growth, annualized
    "momentum": 0.20,   # avg Monte Carlo P(growth) across key metrics
    "safety":   0.15,   # 1 - distress probability
    "solvency": 0.10,   # Altman zone
}

_VERDICTS = [
    (75, "FAVORABLE",   "Strong candidate — fundamentals, growth, and probabilities align."),
    (60, "SELECTIVE",   "Solid but not exceptional — gradual position building is the measured path."),
    (45, "WATCH",       "Mixed picture — the next quarterly filing is free information; no urgency."),
    (30, "CAUTION",     "Weak signals dominate — risk is only justified by conviction the data doesn't show."),
    (0,  "UNFAVORABLE", "Fundamentals and probabilities both argue against deploying capital here."),
]


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _ttm_growth(income: pd.DataFrame | None) -> float | None:
    """Trailing-4-quarter revenue vs the prior 4 quarters, in %."""
    if income is None or income.empty or "Total Revenue" not in income.columns:
        return None
    s = income["Total Revenue"].dropna()
    if len(s) < 8:
        return None
    recent, prior = s.iloc[-4:].sum(), s.iloc[-8:-4].sum()
    if prior <= 0:
        return None
    return (recent - prior) / prior * 100


def _confidence(
    components: dict[str, float],
    health: pd.DataFrame,
    income: pd.DataFrame | None,
    prediction_accuracy: pd.DataFrame | None,
) -> dict:
    """
    How much to trust the verdict (0-100), from three drivers:
      - DEPTH    : how many periods of statement history back the analysis.
      - TRACK    : whether the model has graded past predictions, and how
                   accurate they were (low historical error → high confidence).
      - AGREEMENT: how aligned the five components are. A split decision
                   (some strong, some weak) is inherently less certain than a
                   uniform one.
    """
    drivers: dict[str, float] = {}

    # Depth — saturate at ~24 periods (6 years quarterly).
    periods = len(health) if health is not None else 0
    if income is not None and not income.empty:
        periods = max(periods, len(income))
    drivers["depth"] = _clip01(periods / 24) * 100

    # Track record — needs graded predictions; reward low mean abs error.
    if prediction_accuracy is not None and not prediction_accuracy.empty \
            and "Abs Error" in prediction_accuracy.columns:
        errs = prediction_accuracy["Abs Error"].dropna()
        n = len(errs)
        if n:
            mae = float(errs.mean())
            # 0% error -> 100, 25%+ error -> 0; scaled by sample count (≥6 → full).
            accuracy = _clip01(1 - mae / 0.25) * 100
            sample_w = _clip01(n / 6)
            drivers["track_record"] = accuracy * sample_w + 50 * (1 - sample_w)
        else:
            drivers["track_record"] = 40.0  # no graded history yet
    else:
        drivers["track_record"] = 40.0

    # Agreement — 100 minus the spread of the component scores.
    vals = list(components.values())
    spread = (max(vals) - min(vals)) if vals else 0.0
    drivers["agreement"] = _clip01(1 - spread / 100) * 100

    score = 0.40 * drivers["depth"] + 0.35 * drivers["track_record"] + 0.25 * drivers["agreement"]
    level = "HIGH" if score >= 67 else "MODERATE" if score >= 40 else "LOW"
    return {
        "score": round(score, 1),
        "level": level,
        "drivers": {k: round(v, 1) for k, v in drivers.items()},
    }


def make_decision(
    health: pd.DataFrame,
    distress: pd.DataFrame | None,
    altman: pd.DataFrame | None,
    scenarios: dict | None,
    income: pd.DataFrame | None,
    prediction_accuracy: pd.DataFrame | None = None,
) -> dict:
    """Return {verdict, score, confidence, summary, timing, rationale, components}."""
    components: dict[str, float] = {}
    rationale: list[str] = []

    # --- Health (0-100 already) ---
    health_now = float(health["health_score"].iloc[-1]) if health is not None and not health.empty else 50.0
    health_trend = 0.0
    if health is not None and len(health) >= 4:
        health_trend = health_now - float(health["health_score"].iloc[-4])
    components["health"] = health_now
    rationale.append(f"Financial health {health_now:.0f}/100"
                     + (f", {'improving' if health_trend > 1 else 'deteriorating' if health_trend < -1 else 'stable'}"
                        f" over the last year ({health_trend:+.0f} pts)." if health is not None and len(health) >= 4 else "."))

    # --- Growth ---
    growth = _ttm_growth(income)
    if growth is not None:
        # Map [-10%, +25%] -> [0, 100]
        components["growth"] = _clip01((growth + 10) / 35) * 100
        rationale.append(f"Trailing-twelve-month revenue growth: {growth:+.1f}%.")
    else:
        components["growth"] = 50.0
        rationale.append("Insufficient history to measure revenue growth — scored neutral.")

    # --- Momentum: average Monte Carlo P(growth) ---
    p_growths = []
    for sim in (scenarios or {}).values():
        horizons = (sim or {}).get("horizons") or {}
        if horizons:
            h = max(horizons, key=lambda k: int(k))
            pg = horizons[h].get("p_growth")
            if pg is not None:
                p_growths.append(pg)
    if p_growths:
        avg_pg = float(np.mean(p_growths))
        # Map [30%, 70%] -> [0, 100]
        components["momentum"] = _clip01((avg_pg - 0.30) / 0.40) * 100
        rationale.append(f"Monte Carlo average probability of metric growth: {avg_pg * 100:.0f}%.")
    else:
        components["momentum"] = 50.0

    # --- Safety: distress probability ---
    if distress is not None and not distress.empty:
        p_d = float(distress["p_distress"].iloc[-1])
        components["safety"] = (1 - _clip01(p_d / 0.65)) * 100
        rationale.append(f"Estimated distress probability: {p_d * 100:.1f}%.")
    else:
        components["safety"] = 50.0

    # --- Solvency: Altman zone ---
    zone = None
    if altman is not None and not altman.empty:
        zone = str(altman["z_zone"].iloc[-1])
        components["solvency"] = {"safe": 100.0, "grey": 50.0, "distress": 0.0}.get(zone, 50.0)
        rationale.append(f"Altman Z-Score zone: {zone}.")
    else:
        components["solvency"] = 50.0

    score = sum(_WEIGHTS[k] * components[k] for k in _WEIGHTS)

    verdict, summary = "WATCH", ""
    for threshold, v, s in _VERDICTS:
        if score >= threshold:
            verdict, summary = v, s
            break

    # --- Timing: the "when", driven by health × growth momentum ---
    growing = (growth or 0) > 5
    healthy = health_now >= 65
    if healthy and growing and health_trend >= 0:
        timing = ("Timing favors acting sooner: the company is healthy AND growing, "
                  "so waiting carries a real opportunity cost.")
    elif healthy and not growing:
        timing = ("No urgency: the company is healthy but growth is flat — the next "
                  "quarterly filing is free information, and patience costs nothing.")
    elif not healthy and growing:
        timing = ("High risk / high reward window: growth is real but the balance sheet "
                  "is fragile — only size a position you can afford to be wrong about.")
    else:
        timing = ("Stand aside: neither health nor growth supports deploying capital now; "
                  "re-evaluate after the next two filings.")

    confidence = _confidence(components, health, income, prediction_accuracy)
    rationale.append(
        f"Confidence {confidence['level']} ({confidence['score']:.0f}/100) — "
        f"based on data depth, prediction track record, and signal agreement."
    )

    logger.info("Decision: %s (score %.0f/100, confidence %s %.0f/100).",
                verdict, score, confidence["level"], confidence["score"])
    return {
        "verdict": verdict,
        "score": round(score, 1),
        "confidence": confidence,
        "summary": summary,
        "timing": timing,
        "rationale": rationale,
        "components": {k: round(v, 1) for k, v in components.items()},
        "disclaimer": ("Quantitative summary only — not financial advice. "
                       "Past data and simulations do not guarantee future results."),
    }
