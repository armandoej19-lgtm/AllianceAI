"""
LLM highlights — short, plain-language callouts on the most important metrics.

Design constraints (deliberate):
  - The numbers are computed HERE, deterministically, from the statements.
    The LLM only phrases them — it never invents or calculates figures, which
    keeps hallucination risk near zero.
  - Output is a handful of one-liners ("Debt grew $12.3B (+18%) over the last
    year — likely funding buybacks"), not a narrative essay.
  - Requires ANTHROPIC_API_KEY in the environment / .env.  Without it (or on
    any API failure) we fall back to rule-based template sentences, so the
    pipeline never breaks.
"""

from __future__ import annotations

import json

import pandas as pd

from allianceai.core.config import settings
from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Fact extraction (deterministic)
# ------------------------------------------------------------------

def _fmt_money(v: float) -> str:
    """$12.3B style formatting."""
    if v is None or pd.isna(v):
        return "n/a"
    sign = "-" if v < 0 else ""
    v = abs(v)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= div:
            return f"{sign}${v / div:.1f}{unit}"
    return f"{sign}${v:.0f}"


def _yoy_delta(df: pd.DataFrame, col: str, periods: int = 4) -> tuple[float, float] | None:
    """(absolute change, % change) over the last *periods* statements."""
    if df is None or df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    if len(s) <= periods:
        return None
    new, old = float(s.iloc[-1]), float(s.iloc[-1 - periods])
    if old == 0:
        return None
    return new - old, (new - old) / abs(old) * 100


def extract_key_facts(
    info: dict,
    health: pd.DataFrame,
    altman: pd.DataFrame | None,
    distress: pd.DataFrame | None,
    signals: list,
    income: pd.DataFrame | None,
    balance: pd.DataFrame | None,
    cashflow: pd.DataFrame | None,
) -> dict:
    """Reduce the full analysis to a small dict of pre-computed facts."""
    facts: dict = {
        "company": info.get("longName"),
        "sector": info.get("sector"),
        "market_cap": _fmt_money(info.get("marketCap")),
    }

    if health is not None and not health.empty:
        latest = health.iloc[-1]
        facts["health_score"] = f"{latest.get('health_score', float('nan')):.0f}/100"
        weakest = min(
            ["liquidity", "leverage", "profitability", "cashflow"],
            key=lambda k: latest.get(f"{k}_subscore", 100),
        )
        facts["weakest_dimension"] = f"{weakest} ({latest.get(f'{weakest}_subscore', float('nan')):.0f}/100)"

    if altman is not None and not altman.empty:
        facts["altman_z"] = f"{altman['z_score'].iloc[-1]:.1f} ({altman['z_zone'].iloc[-1]})"

    if distress is not None and not distress.empty:
        facts["distress_probability"] = f"{distress['p_distress'].iloc[-1] * 100:.1f}% ({distress['distress_zone'].iloc[-1]})"

    # Year-over-year flow changes (trailing 4 quarters vs prior 4-quarter point).
    for label, df, col in [
        ("revenue", income, "Total Revenue"),
        ("net_income", income, "Net Income"),
        ("total_debt", balance, "Total Debt"),
        ("operating_cash_flow", cashflow, "Operating Cash Flow"),
    ]:
        d = _yoy_delta(df, col)
        if d:
            # Explicitly label change vs level so the LLM can't confuse them.
            facts[f"{label}_change_yoy"] = f"{_fmt_money(d[0])} ({d[1]:+.0f}%)"
        if df is not None and not df.empty and col in df.columns:
            s = df[col].dropna()
            if not s.empty:
                facts[f"{label}_latest"] = _fmt_money(float(s.iloc[-1]))

    if cashflow is not None and not cashflow.empty:
        buyback = cashflow.get("Repurchase Of Capital Stock")
        if buyback is not None and buyback.notna().any():
            recent = buyback.dropna().iloc[-4:].sum()
            if recent != 0:
                facts["buybacks_last_4_quarters"] = _fmt_money(abs(recent))
        div = cashflow.get("Dividends Paid")
        if div is not None and div.notna().any():
            recent = div.dropna().iloc[-4:].sum()
            if recent != 0:
                facts["dividends_last_4_quarters"] = _fmt_money(abs(recent))

    if signals:
        facts["risk_signals"] = [f"[{s.severity}] {s.name}" for s in signals[:5]]

    return facts


# ------------------------------------------------------------------
# LLM phrasing (Claude Haiku) with rule-based fallback
# ------------------------------------------------------------------

_SYSTEM = """You write financial highlight bullets for a quantitative report.

Rules:
- Use ONLY the numbers given. Never compute, extrapolate, or invent figures.
- Each highlight is ONE short sentence in plain English a non-expert understands.
- Lead with what matters: big debt moves, profitability shifts, cash generation,
  distress risk, buybacks/dividends.
- 3 to 5 highlights total. Skip facts that aren't notable.
- No hedging filler, no investment advice, no adjectives like "impressive"."""


_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "highlights": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["highlights"],
    "additionalProperties": False,
}


def _clean(highlights: list[str]) -> list[str] | None:
    return [h.strip() for h in highlights if h and h.strip()][:5] or None


def _anthropic_highlights(facts: dict, system: str | None = None) -> list[str] | None:
    """Claude API directly (structured output via json_schema)."""
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    response = client.messages.create(
        model=settings.highlights_model,
        max_tokens=512,
        system=system or _SYSTEM,
        messages=[{
            "role": "user",
            "content": "Computed facts:\n" + json.dumps(facts, indent=2),
        }],
        output_config={"format": {"type": "json_schema", "schema": _JSON_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return _clean(json.loads(text)["highlights"])


def _openrouter_highlights(facts: dict, system: str | None = None) -> list[str] | None:
    """Any model via OpenRouter's OpenAI-compatible chat endpoint."""
    import requests

    resp = requests.post(
        f"{settings.openrouter_base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.openrouter_model,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": system or _SYSTEM},
                {"role": "user",
                 "content": "Computed facts:\n" + json.dumps(facts, indent=2)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "highlights", "strict": True,
                                "schema": _JSON_SCHEMA},
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _clean(json.loads(content)["highlights"])


def _resolve_provider() -> str:
    """Resolve the effective provider, honoring 'auto'."""
    import os
    provider = settings.llm_provider.lower()
    if provider != "auto":
        return provider
    if settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if settings.openrouter_api_key:
        return "openrouter"
    return "none"


def _llm_highlights(facts: dict) -> list[str] | None:
    provider = _resolve_provider()
    if provider == "none":
        logger.info("No LLM provider configured — using rule-based highlights.")
        return None
    try:
        if provider == "anthropic":
            return _anthropic_highlights(facts)
        if provider == "openrouter":
            return _openrouter_highlights(facts)
        logger.warning("Unknown llm_provider '%s' — using rule-based fallback.", provider)
        return None
    except Exception as exc:
        logger.warning("LLM highlights via %s failed (%s) — using rule-based fallback.",
                       provider, exc)
        return None


def _rule_based_highlights(facts: dict) -> list[str]:
    """Deterministic fallback — fixed phrasing, same underlying facts."""
    out = []
    if "total_debt_change_yoy" in facts:
        out.append(f"Total debt changed by {facts['total_debt_change_yoy']} over the last year.")
    if "revenue_change_yoy" in facts:
        out.append(f"Revenue changed by {facts['revenue_change_yoy']} year-over-year.")
    if "net_income_change_yoy" in facts:
        out.append(f"Net income changed by {facts['net_income_change_yoy']} year-over-year.")
    if "buybacks_last_4_quarters" in facts:
        out.append(f"Spent {facts['buybacks_last_4_quarters']} on share buybacks in the last 4 quarters.")
    if "distress_probability" in facts:
        out.append(f"Estimated distress probability: {facts['distress_probability']}.")
    if "health_score" in facts and "weakest_dimension" in facts:
        out.append(f"Overall health {facts['health_score']}; weakest area is {facts['weakest_dimension']}.")
    return out[:5]


def generate_highlights(
    info: dict,
    health: pd.DataFrame,
    altman: pd.DataFrame | None = None,
    distress: pd.DataFrame | None = None,
    signals: list | None = None,
    income: pd.DataFrame | None = None,
    balance: pd.DataFrame | None = None,
    cashflow: pd.DataFrame | None = None,
) -> list[str]:
    """Return 3–5 one-line highlights. LLM-phrased when a key is available."""
    facts = extract_key_facts(info, health, altman, distress, signals or [],
                              income, balance, cashflow)
    logger.info("Generating highlights from %d key facts.", len(facts))
    return _llm_highlights(facts) or _rule_based_highlights(facts)


# ------------------------------------------------------------------
# Opportunity highlights — probability expectations from Monte Carlo
# ------------------------------------------------------------------

_OPPORTUNITY_SYSTEM = """You write forward-looking opportunity bullets for a quantitative report,
based on Monte Carlo simulation probabilities.

Rules:
- Use ONLY the probabilities and figures given. Never compute or invent numbers.
- Each bullet is ONE short sentence framing the upside or expectation, e.g.
  "78% probability revenue grows over the next year, with a median gain of +9%."
- Mention the downside (VaR) only where it's material.
- 3 to 5 bullets total. Skip metrics that aren't notable.
- These are statistical expectations, not advice — never say "buy" or "should"."""

# Metrics worth surfacing, in priority order.
_OPPORTUNITY_METRICS = ["Total Revenue", "Net Income", "Free Cash Flow",
                        "Operating Cash Flow", "EBITDA"]


def extract_opportunity_facts(scenarios: dict | None) -> dict:
    """Reduce Monte Carlo scenario output to a small dict of expectations.

    Uses the longest available horizon per metric (typically 4 quarters)."""
    facts: dict = {}
    if not scenarios:
        return facts
    for metric in _OPPORTUNITY_METRICS:
        sim = scenarios.get(metric)
        if not isinstance(sim, dict):
            continue
        horizons = sim.get("horizons") or {}
        if not horizons:
            continue
        h = max(horizons, key=lambda k: int(k))
        hd = horizons[h]
        try:
            facts[metric] = {
                "horizon_quarters": int(h),
                "p_growth": f"{hd.get('p_growth', 0) * 100:.0f}%",
                "median_expected_change": f"{hd.get('expected_change_pct', 0):+.1f}%",
                "upside_p90": _fmt_money(hd.get("p90")),
                "downside_var_5pct": _fmt_money(hd.get("var_5pct")),
                "current_value": _fmt_money(sim.get("last_value")),
            }
        except (TypeError, ValueError):
            continue
    return facts


def _rule_based_opportunities(facts: dict) -> list[str]:
    out = []
    for metric, f in facts.items():
        out.append(
            f"{f['p_growth']} probability {metric.lower()} grows over the next "
            f"{f['horizon_quarters']} quarters (median {f['median_expected_change']})."
        )
    return out[:5]


def generate_opportunities(scenarios: dict | None) -> list[str]:
    """Return 3–5 probability-expectation bullets from Monte Carlo scenarios."""
    facts = extract_opportunity_facts(scenarios)
    if not facts:
        return []
    logger.info("Generating opportunity highlights for %d metrics.", len(facts))
    provider = _resolve_provider()
    if provider != "none":
        try:
            if provider == "anthropic":
                return _anthropic_highlights(facts, system=_OPPORTUNITY_SYSTEM) \
                    or _rule_based_opportunities(facts)
            if provider == "openrouter":
                return _openrouter_highlights(facts, system=_OPPORTUNITY_SYSTEM) \
                    or _rule_based_opportunities(facts)
        except Exception as exc:
            logger.warning("LLM opportunities via %s failed (%s) — using fallback.",
                           provider, exc)
    return _rule_based_opportunities(facts)
