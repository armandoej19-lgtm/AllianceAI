"""
HTML report generator — quantitative output only.

Produces a self-contained, professionally-styled HTML file with:
  - Sticky navigation + at-a-glance KPI band
  - Investment stance, health scores, distress probabilities
  - "Current vs Forecast" snapshot tying actuals to Monte Carlo probabilities
  - Risk signals (severity + evidence)
  - Factor analysis: predictive correlations, trend statistics, decomposition
  - Monte Carlo scenario table (p-values, percentiles, VaR)
  - Forecast table (slope, p-value, R², fit error, confidence intervals)
  - Inline Plotly charts (revenue/income, cash flow, debt-vs-revenue-vs-income,
    health over time)

No data is dropped — every section the pipeline produces is rendered; the
redesign only reorganises and styles it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from jinja2 import BaseLoader, Environment

from allianceai.analysis.risk_signals import RiskSignal
from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _color_score(val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "#aaa"
    return "#4caf50" if v >= 70 else "#ffeb3b" if v >= 50 else "#f44336"


def _color_prob(p) -> str:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "#aaa"
    return "#f44336" if v >= 0.65 else "#ff9800" if v >= 0.40 else "#ffeb3b" if v >= 0.20 else "#4caf50"


def _sev_color(sev: str) -> str:
    return {"CRITICAL": "#f44336", "HIGH": "#ff9800", "MEDIUM": "#ffeb3b", "LOW": "#4caf50"}.get(sev, "#aaa")


def _pct(val, digits=1) -> str:
    try:
        return f"{float(val) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt(val, digits=2) -> str:
    try:
        return f"{float(val):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _money(val) -> str:
    """Compact currency: $1.23B / $45.6M / $789."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "—"
    a = abs(v)
    if a >= 1e9:
        return f"${v / 1e9:,.2f}B"
    if a >= 1e6:
        return f"${v / 1e6:,.1f}M"
    if a >= 1e3:
        return f"${v / 1e3:,.1f}K"
    return f"${v:,.0f}"


def _pct_color(v: float, good_high: bool = True) -> str:
    """Green for positive growth, red for negative (flip with good_high)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "#aaa"
    pos = v >= 0
    if not good_high:
        pos = not pos
    return "#66bb6a" if pos else "#ef5350"


def _chart_html(traces: list[tuple[str, pd.Series]], title: str) -> str | None:
    """Render an interactive Plotly chart from (name, series) traces."""
    try:
        fig = go.Figure()
        for name, s in traces:
            if s is None or s.dropna().empty:
                continue
            s = s.dropna()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines+markers"))
        if not fig.data:
            return None
        fig.update_layout(
            title=title, template="plotly_dark",
            paper_bgcolor="#161922", plot_bgcolor="#161922",
            height=380, margin=dict(l=48, r=20, t=44, b=40),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
            xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)),
        )
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displaylogo": False, "responsive": True},
        )
    except Exception as exc:
        logger.warning("Chart '%s' failed: %s", title, exc)
        return None


def _df_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    if df is None or df.empty:
        return "<p class='muted'>No data.</p>"
    return df.to_html(
        classes="data-table", border=0,
        float_format=lambda x: f"{x:.4f}",
        na_rep="—",
    )


# Which statement a Monte Carlo metric is forecast under, for cross-referencing.
_SNAPSHOT_METRICS = [
    "Total Revenue", "Net Income", "EBITDA", "Operating Income",
    "Operating Cash Flow", "Free Cash Flow", "Total Debt",
]


def _snapshot_rows(scenarios: dict | None, forecasts: dict | None = None) -> list[dict]:
    """
    Build the explicit 'Current vs Forecast' table: latest actual alongside the
    Monte Carlo probability distribution (next-quarter median, expected change,
    P(growth)) at 1 and 4 quarters out.  Debt is framed so that *lower* is
    better.  The Prophet point forecast is shown separately in the Forecasts
    table; here every column comes from the same simulation so they're coherent.
    """
    if not scenarios:
        return []
    rows = []
    for metric in _SNAPSHOT_METRICS:
        sim = scenarios.get(metric)
        if not sim:
            continue
        horizons = sim.get("horizons") or {}
        h1 = horizons.get(1) or horizons.get("1") or {}
        h4 = horizons.get(4) or horizons.get("4") or {}
        cur = sim.get("last_value")
        # Use the Monte Carlo 1-quarter median so this column stays consistent
        # with the Expected-Δ and P(growth) columns next to it (all from the same
        # simulation).  The Prophet point forecast lives in the Forecasts table.
        next_median = h1.get("p50_median")

        debt_like = metric == "Total Debt"
        rows.append({
            "metric": metric,
            "current": _money(cur),
            "next_forecast": _money(next_median) if next_median is not None else "—",
            "exp_change": h1.get("expected_change_pct"),
            "p_growth_1q": h1.get("p_growth"),
            "median_1y": _money(h4.get("p50_median")),
            "p_growth_1y": h4.get("p_growth"),
            "range_1y": f"{_money(h4.get('p10'))} – {_money(h4.get('p90'))}"
                        if h4 else "—",
            "debt_like": debt_like,
        })
    return rows


# --------------------------------------------------------------------------- #
# Template
# --------------------------------------------------------------------------- #

_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AllianceAI — {{ ticker }}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root{
  --bg:#0d0f15; --bg2:#11141d; --card:#161922; --card2:#1d2130;
  --line:#262b3a; --ink:#e6e9f0; --ink-dim:#9aa3b5; --muted:#6b7488;
  --accent:#4fc3f7; --accent2:#7c4dff; --good:#66bb6a; --warn:#ffb300;
  --bad:#ef5350; --mid:#ffee58;
}
*{box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{font-family:'Segoe UI',Inter,Arial,sans-serif;background:
     radial-gradient(1200px 600px at 80% -10%,#16203a 0%,var(--bg) 55%);
     color:var(--ink);margin:0;padding:0 0 60px 0;font-size:14px;line-height:1.5;}
.wrap{max-width:1180px;margin:0 auto;padding:0 22px;}

/* sticky top nav */
.topbar{position:sticky;top:0;z-index:50;backdrop-filter:blur(10px);
        background:rgba(13,15,21,.82);border-bottom:1px solid var(--line);}
.topbar .wrap{display:flex;align-items:center;gap:16px;height:54px;}
.brand{font-weight:700;color:var(--accent);letter-spacing:.3px;white-space:nowrap;}
.brand b{color:var(--ink);}
.nav{display:flex;gap:4px;flex-wrap:wrap;margin-left:auto;}
.nav a{color:var(--ink-dim);text-decoration:none;font-size:.82em;padding:5px 9px;
       border-radius:7px;transition:.15s;}
.nav a:hover{color:var(--ink);background:var(--card2);}
.pill{padding:4px 12px;border-radius:20px;font-weight:700;font-size:.8em;
      border:1px solid currentColor;}

/* hero */
.hero{padding:30px 0 6px 0;}
.hero h1{font-size:1.7em;margin:0 0 4px 0;color:var(--ink);font-weight:700;}
.hero .sub{color:var(--muted);font-size:.92em;}
.conf-badge{display:inline-block;margin-left:10px;padding:3px 12px;border-radius:12px;
            font-size:.8em;font-weight:700;vertical-align:middle;cursor:help;}

.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
      gap:12px;margin:18px 0 6px;}
.kpi{background:linear-gradient(160deg,var(--card2),var(--card));border:1px solid var(--line);
     border-radius:12px;padding:14px 16px;}
.kpi .k-lbl{font-size:.72em;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);}
.kpi .k-val{font-size:1.5em;font-weight:700;margin-top:3px;}
.kpi .k-sub{font-size:.78em;color:var(--ink-dim);margin-top:2px;}

/* section cards */
h2{color:var(--ink);font-size:1.18em;margin:0 0 2px 0;display:flex;align-items:center;gap:9px;}
h2 .ic{width:6px;height:20px;border-radius:3px;background:linear-gradient(var(--accent),var(--accent2));display:inline-block;}
h3{color:#aab4cc;margin:18px 0 4px;font-size:1.0em;}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
      padding:20px 22px;margin:18px 0;box-shadow:0 1px 0 rgba(255,255,255,.02) inset;}
.section-note,.muted{color:var(--muted);font-size:.84em;margin:4px 0 0;}
.muted{color:var(--muted);}

.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;}
.box{background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:14px;text-align:center;}
.box .val{font-size:1.7em;font-weight:700;}
.box .lbl{font-size:.78em;color:var(--ink-dim);margin-top:4px;}

.risk-item{border-left:4px solid #555;padding:11px 15px;margin:9px 0;border-radius:0 8px 8px 0;}
.CRITICAL{border-color:var(--bad);background:#1c0a0a;}
.HIGH{border-color:var(--warn);background:#1c1200;}
.MEDIUM{border-color:var(--mid);background:#1a1800;}
.LOW{border-color:var(--good);background:#0a1a0a;}
.sev{font-weight:700;font-size:.72em;letter-spacing:1px;}

table.data-table{border-collapse:collapse;width:100%;font-size:.84em;margin-top:10px;}
table.data-table th,table.data-table td{padding:7px 10px;border:1px solid var(--line);text-align:right;}
table.data-table th{background:var(--card2);text-align:center;color:var(--accent);}
table.data-table td:first-child{text-align:left;font-weight:600;color:var(--ink);}
table.data-table tbody tr:nth-child(even){background:rgba(255,255,255,.018);}
table.data-table tbody tr:hover{background:rgba(79,195,247,.07);}
.table-scroll{overflow:auto;border-radius:10px;}

/* snapshot table */
.snap td,.snap th{text-align:center;}
.snap td:first-child{text-align:left;}
.bar{height:7px;border-radius:4px;background:var(--line);position:relative;overflow:hidden;min-width:54px;}
.bar > i{position:absolute;left:0;top:0;bottom:0;border-radius:4px;}
.tag{display:inline-block;padding:2px 8px;border-radius:6px;font-weight:700;font-size:.9em;}

.chart{border-radius:10px;overflow:hidden;margin-top:10px;border:1px solid var(--line);}
.hl-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
@media (max-width:900px){.hl-grid{grid-template-columns:1fr;}.nav{display:none;}}
.mono{font-family:ui-monospace,Consolas,monospace;}
footer{margin-top:40px;color:var(--muted);font-size:.8em;text-align:center;}
</style>
</head>
<body>

<!-- ===== STICKY NAV ===== -->
<div class="topbar"><div class="wrap">
  <span class="brand">Alliance<b>AI</b> · {{ ticker }}</span>
  {% if decision %}
  <span class="pill" style="color:{{ verdict_color }}">{{ decision.verdict }}</span>
  {% endif %}
  <nav class="nav">
    {% if price_metrics %}<a href="#fund">Profile</a>{% endif %}
    <a href="#stance">Stance</a>
    <a href="#snapshot">Current vs Forecast</a>
    <a href="#health">Health</a>
    <a href="#distress">Distress</a>
    <a href="#risks">Risks</a>
    <a href="#factors">Factors</a>
    <a href="#montecarlo">Monte Carlo</a>
    <a href="#forecasts">Forecasts</a>
    <a href="#charts">Charts</a>
  </nav>
</div></div>

<div class="wrap">

<!-- ===== HERO ===== -->
<div class="hero">
  <h1>{{ info.get('longName', ticker) }} <span style="color:var(--muted);font-weight:500">({{ ticker }})</span>
  {% if data_confidence %}
  <span class="conf-badge" style="background:{{ {'HIGH':'#16331b','MODERATE':'#332b10','LOW':'#331010'}.get(data_confidence.level,'#222') }}; color:{{ {'HIGH':'#81c784','MODERATE':'#ffd54f','LOW':'#ef9a9a'}.get(data_confidence.level,'#aaa') }}"
        title="{{ data_confidence.notes | join(' ') }}">data confidence {{ data_confidence.level }} ({{ data_confidence.score }}/100)</span>
  {% endif %}
  </h1>
  <div class="sub">{{ info.get('sector','') }}{% if info.get('industry') %} · {{ info.get('industry') }}{% endif %}{% if info.get('country') %} · {{ info.get('country') }}{% endif %}</div>

  {% if hero_kpis %}
  <div class="kpis">
    {% for k in hero_kpis %}
    <div class="kpi">
      <div class="k-lbl">{{ k.label }}</div>
      <div class="k-val" style="color:{{ k.color }}">{{ k.value }}</div>
      <div class="k-sub">{{ k.sub }}</div>
    </div>
    {% endfor %}
  </div>
  {% endif %}
</div>

{% if price_metrics %}
<!-- ===== FUND PROFILE & PRICE METRICS ===== -->
<div class="card" id="fund">
  <h2><span class="ic"></span>Fund Profile &amp; Price Analytics</h2>
  <p class="section-note">This instrument files no financial statements, so it is analysed from its
    price series and fund metadata rather than fundamentals. Returns are point-to-point; volatility is
    annualised daily standard deviation; Sharpe uses a 0% risk-free rate. As of {{ price_metrics.as_of }}.</p>

  {% if fund_facts %}
  <div class="grid" style="margin-top:12px">
    {% for label, val in fund_facts %}
    <div class="box"><div class="val" style="font-size:1.05em">{{ val }}</div><div class="lbl">{{ label }}</div></div>
    {% endfor %}
  </div>
  {% endif %}

  <div style="display:flex;gap:18px;flex-wrap:wrap;margin-top:16px">
    {% if perf_rows %}
    <div style="flex:1;min-width:240px">
      <h3>Trailing Returns</h3>
      <table class="data-table"><tbody>
        {% for label, val, color in perf_rows %}
        <tr><td>{{ label }}</td><td class="mono" style="color:{{ color }};text-align:right">{{ val }}</td></tr>
        {% endfor %}
      </tbody></table>
    </div>
    {% endif %}
    {% if risk_rows %}
    <div style="flex:1;min-width:240px">
      <h3>Risk</h3>
      <table class="data-table"><tbody>
        {% for label, val in risk_rows %}
        <tr><td>{{ label }}</td><td class="mono" style="text-align:right">{{ val }}</td></tr>
        {% endfor %}
      </tbody></table>
    </div>
    {% endif %}
    {% if level_rows %}
    <div style="flex:1;min-width:240px">
      <h3>Levels &amp; Trend</h3>
      <table class="data-table"><tbody>
        {% for label, val, color in level_rows %}
        <tr><td>{{ label }}</td><td class="mono" style="color:{{ color }};text-align:right">{{ val }}</td></tr>
        {% endfor %}
      </tbody></table>
    </div>
    {% endif %}
  </div>
</div>
{% endif %}

{% if decision %}
<!-- ===== INVESTMENT STANCE ===== -->
<div class="card" id="stance" style="border-color:{{ verdict_color }}">
  <h2><span class="ic"></span>Investment Stance</h2>
  <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;margin-top:10px">
    <div class="box" style="min-width:160px">
      <div class="val" style="color:{{ verdict_color }}">{{ decision.verdict }}</div>
      <div class="lbl">{{ decision.score }}/100 composite</div>
    </div>
    {% if decision.confidence %}
    <div class="box" style="min-width:140px">
      <div class="val" style="color:{{ {'HIGH':'#4caf50','MODERATE':'#ffeb3b','LOW':'#f44336'}.get(decision.confidence.level,'#aaa') }}">{{ decision.confidence.level }}</div>
      <div class="lbl">confidence · {{ decision.confidence.score }}/100</div>
    </div>
    {% endif %}
    <div style="flex:1;min-width:280px">
      <p style="margin:4px 0"><strong>{{ decision.summary }}</strong></p>
      <p style="margin:4px 0;color:#90caf9">{{ decision.timing }}</p>
    </div>
  </div>
  <ul style="color:#ccd2df;font-size:.9em;margin:12px 0 4px;padding-left:22px">
    {% for r in decision.rationale %}<li>{{ r }}</li>{% endfor %}
  </ul>
  <div class="grid" style="margin-top:10px">
    {% for name, val in decision.components.items() %}
    <div class="box">
      <div class="val" style="font-size:1.2em;color:{{ '#4caf50' if val >= 70 else '#ffeb3b' if val >= 50 else '#f44336' }}">{{ val }}</div>
      <div class="lbl">{{ name }}</div>
    </div>
    {% endfor %}
  </div>
  <p class="muted" style="margin-top:10px">{{ decision.disclaimer }}</p>
</div>
{% endif %}

{% if highlights or opportunities %}
<div class="hl-grid">
  {% if highlights %}
  <div class="card">
    <h2><span class="ic"></span>Highlights</h2>
    <ul style="line-height:1.9;font-size:1.0em;margin:10px 0 0;padding-left:22px">
      {% for h in highlights %}<li>{{ h }}</li>{% endfor %}
    </ul>
  </div>
  {% endif %}
  {% if opportunities %}
  <div class="card" style="border-left:3px solid var(--good)">
    <h2><span class="ic"></span>Opportunity Outlook</h2>
    <p class="muted">Monte Carlo probability expectations — statistical projections, not advice.</p>
    <ul style="line-height:1.9;font-size:1.0em;margin:8px 0 0;padding-left:22px">
      {% for o in opportunities %}<li>{{ o }}</li>{% endfor %}
    </ul>
  </div>
  {% endif %}
</div>
{% endif %}

{% if snapshot_rows %}
<!-- ===== CURRENT VS FORECAST ===== -->
<div class="card" id="snapshot">
  <h2><span class="ic"></span>Current vs Forecast — Key Metrics &amp; Probabilities</h2>
  <p class="section-note">Each row pairs the <b>latest reported actual</b> with the model's
    <b>next-period point forecast</b> and the Monte Carlo <b>probability distribution</b> at
    1 and 4 quarters out. P(growth) is the chance the metric ends <em>above</em> today's value
    (for Total Debt, lower is better — its bar is inverted). Data is quarterly, so "latest" is the
    most recent quarter.</p>
  <div class="table-scroll">
  <table class="data-table snap">
    <thead><tr>
      <th>Metric</th><th>Latest actual</th><th>Next q (median)</th>
      <th>Exp. Δ (next q)</th><th>P(growth) next q</th>
      <th>1-yr median</th><th>P(growth) 1-yr</th><th>1-yr range (P10–P90)</th>
    </tr></thead>
    <tbody>
    {% for r in snapshot_rows %}
      <tr>
        <td>{{ r.metric }}</td>
        <td class="mono">{{ r.current }}</td>
        <td class="mono">{{ r.next_forecast }}</td>
        <td><span class="tag" style="background:{{ pct_bg(r.exp_change, r.debt_like) }};color:#0d0f15">{{ signed(r.exp_change) }}</span></td>
        <td>{{ prob_bar(r.p_growth_1q, r.debt_like) }}</td>
        <td class="mono">{{ r.median_1y }}</td>
        <td>{{ prob_bar(r.p_growth_1y, r.debt_like) }}</td>
        <td class="mono" style="font-size:.92em">{{ r.range_1y }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  </div>
</div>
{% endif %}

<!-- ===== HEALTH SCORE ===== -->
<div class="card" id="health">
  <h2><span class="ic"></span>Financial Health Score</h2>
  <div class="grid" style="margin-top:10px">
    {% for label, val, col in health_boxes %}
    <div class="box"><div class="val" style="color:{{ col }}">{{ val }}</div><div class="lbl">{{ label }}</div></div>
    {% endfor %}
  </div>
</div>

{% if distress_row %}
<!-- ===== DISTRESS ===== -->
<div class="card" id="distress">
  <h2><span class="ic"></span>Distress Probability (Latest Period)</h2>
  <div class="grid" style="margin-top:10px">
    <div class="box"><div class="val" style="color:{{ p_color(distress_row.p_distress) }}">{{ pct(distress_row.p_distress) }}</div><div class="lbl">Ensemble P(distress)</div></div>
    <div class="box"><div class="val" style="color:{{ p_color(distress_row.p_distress_ohlson) }}">{{ pct(distress_row.p_distress_ohlson) }}</div><div class="lbl">Ohlson (1980)</div></div>
    <div class="box"><div class="val" style="color:{{ p_color(distress_row.p_distress_zmijewski) }}">{{ pct(distress_row.p_distress_zmijewski) }}</div><div class="lbl">Zmijewski (1984)</div></div>
    <div class="box"><div class="val" style="color:{{ p_color(distress_row.p_distress_altman) }}">{{ pct(distress_row.p_distress_altman) }}</div><div class="lbl">Altman-mapped</div></div>
    <div class="box"><div class="val" style="color:#81d4fa">{{ distress_row.distress_zone }}</div><div class="lbl">Zone</div></div>
  </div>
  <h3>History</h3>
  <div class="table-scroll">{{ distress_table | safe }}</div>
</div>
{% endif %}

<!-- ===== RISK SIGNALS ===== -->
<div class="card" id="risks">
  <h2><span class="ic"></span>Risk Signals</h2>
  {% for s in signals %}
  <div class="risk-item {{ s.severity }}">
    <span class="sev" style="color:{{ sev_color(s.severity) }}">{{ s.severity }}</span>
    &nbsp;&nbsp;<strong>{{ s.name }}</strong>
    <span class="muted" style="font-size:.85em"> · {{ s.category }}</span><br>
    <span style="color:#ccd2df;font-size:.88em">{{ s.explanation }}</span>
    {% if s.evidence %}
    <div class="mono muted" style="font-size:.82em;margin-top:4px">
      {% for k, v in s.evidence.items() %}{{ k }}: {{ v }}  {% endfor %}
    </div>
    {% endif %}
  </div>
  {% else %}
  <p style="color:var(--good)">No risk signals detected.</p>
  {% endfor %}
</div>

{% if factors %}
<!-- ===== FACTOR ANALYSIS ===== -->
<div class="card" id="factors">
  <h2><span class="ic"></span>Factor Analysis</h2>
  {% if factors.predictive_correlations is defined and factors.predictive_correlations is not none %}
  <h3>Predictive Correlations with Future Net Margin (lag=1)</h3>
  <p class="muted">Pearson r · p-value · 95% CI via Fisher z-transform. Significant means p&lt;0.05.</p>
  <div class="table-scroll">{{ pred_corr_table | safe }}</div>
  {% endif %}
  {% if factors.trend_statistics is defined and factors.trend_statistics is not none %}
  <h3>Trend Statistics (OLS)</h3>
  <p class="muted">Slope per period, R², p-value, t-statistic. RISING/FALLING only if p&lt;0.05.</p>
  <div class="table-scroll">{{ trend_table | safe }}</div>
  {% endif %}
  {% if factors.income_decomposition is defined and factors.income_decomposition is not none %}
  <h3>Income Change Decomposition</h3>
  <p class="muted">How much of ΔNet Income is explained by revenue growth vs margin improvement?</p>
  <div class="table-scroll">{{ income_decomp_table | safe }}</div>
  {% endif %}
  {% if pca_variance %}
  <h3>PCA Explained Variance</h3>
  <p class="muted">Ratio factor decomposition. PC1 = {{ pca_variance[0] }}%{% if pca_variance|length > 1 %}, PC2 = {{ pca_variance[1] }}%{% endif %}{% if pca_variance|length > 2 %}, PC3 = {{ pca_variance[2] }}%{% endif %}</p>
  <div class="table-scroll">{{ pca_loadings_table | safe }}</div>
  {% endif %}
</div>
{% endif %}

{% if scenarios %}
<!-- ===== MONTE CARLO ===== -->
<div class="card" id="montecarlo">
  <h2><span class="ic"></span>Monte Carlo Scenario Analysis (10,000 simulations, GBM)</h2>
  <p class="muted">P(growth) = probability of exceeding current value. VaR 5% = worst-case value at 5th percentile. Median = 50th percentile expected value.</p>
  <div class="table-scroll">{{ scenario_table | safe }}</div>
</div>
{% endif %}

{% if accuracy_table_html %}
<!-- ===== PREDICTION ACCURACY ===== -->
<div class="card" id="accuracy">
  <h2><span class="ic"></span>Real Data vs Past Predictions</h2>
  <p class="muted">Forecasts from previous runs, graded against the official figures once reported. These errors also calibrate today's forecasts.</p>
  <div class="table-scroll">{{ accuracy_table_html | safe }}</div>
</div>
{% endif %}

{% if forecast_rows %}
<!-- ===== FORECASTS ===== -->
<div class="card" id="forecasts">
  <h2><span class="ic"></span>Forecasts with Statistical Evidence</h2>
  <p class="muted">Slope p-value tests whether the recent trend is statistically non-zero. R² measures how well the trend explains variance. "Trend" is the fit space (<em>log-linear</em> for exponential growth, <em>linear</em> otherwise), chosen automatically over the most recent ~5 years. Fit err = WAPE (Σ|actual−fitted| / Σ|actual|) on that window.</p>
  <div class="table-scroll">{{ forecast_table | safe }}</div>
</div>
{% endif %}

{% if charts %}
<!-- ===== CHARTS ===== -->
<div class="card" id="charts">
  <h2><span class="ic"></span>Charts</h2>
  {% for title, chart_div in charts %}
  <h3>{{ title }}</h3>
  <div class="chart">{{ chart_div | safe }}</div>
  {% endfor %}
</div>
{% endif %}

<footer>Generated by AllianceAI &nbsp;·&nbsp; Data: yfinance / Yahoo Finance + SEC EDGAR &nbsp;·&nbsp; Quantitative summary only — not financial advice.</footer>
</div>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Jinja helpers for the snapshot table
# --------------------------------------------------------------------------- #

def _signed(v) -> str:
    try:
        return f"{float(v):+.1f}%"
    except (TypeError, ValueError):
        return "—"


def _pct_bg(v, debt_like: bool = False) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "#3a3f50"
    good = (v < 0) if debt_like else (v >= 0)
    return "#66bb6a" if good else "#ef5350"


def _prob_bar(p, debt_like: bool = False) -> str:
    """A small inline probability bar + label, coloured by favourability."""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "—"
    favourable = (1 - v) if debt_like else v
    color = "#66bb6a" if favourable >= 0.55 else "#ffee58" if favourable >= 0.4 else "#ef5350"
    width = max(3, min(100, round(v * 100)))
    return (f"<div style='display:flex;align-items:center;gap:7px;justify-content:center'>"
            f"<div class='bar'><i style='width:{width}%;background:{color}'></i></div>"
            f"<span style='font-size:.9em;color:{color};min-width:34px'>{v*100:.0f}%</span></div>")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def generate_html_report(
    ticker: str,
    info: dict,
    health: pd.DataFrame,
    signals: list[RiskSignal],
    altman: pd.DataFrame | None = None,
    distress: pd.DataFrame | None = None,
    factors: dict | None = None,
    scenarios: dict | None = None,
    forecasts: dict | None = None,
    highlights: list[str] | None = None,
    opportunities: list[str] | None = None,
    decision: dict | None = None,
    prediction_accuracy: pd.DataFrame | None = None,
    data_confidence: dict | None = None,
    income: pd.DataFrame | None = None,
    cashflow: pd.DataFrame | None = None,
    balance: pd.DataFrame | None = None,
    price_metrics: dict | None = None,
    prices: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
) -> str:
    logger.info("Generating HTML report for '%s'.", ticker)

    verdict_color = {"FAVORABLE": "#4caf50", "SELECTIVE": "#8bc34a", "WATCH": "#ffeb3b",
                     "CAUTION": "#ff9800", "UNFAVORABLE": "#f44336"}.get(
        (decision or {}).get("verdict"), "#4fc3f7")

    # --- Health boxes ---
    health_boxes: list[tuple[str, str, str]] = []
    if not health.empty:
        row = health.iloc[-1]
        for label, key in [
            ("Overall", "health_score"),
            ("Liquidity", "liquidity_subscore"),
            ("Leverage", "leverage_subscore"),
            ("Profitability", "profitability_subscore"),
            ("Cash Flow", "cashflow_subscore"),
        ]:
            val = row.get(key)
            try:
                display = f"{float(val):.0f}"
            except (TypeError, ValueError):
                display = "—"
            health_boxes.append((label, display, _color_score(val)))

    # --- Snapshot (current vs forecast) ---
    snapshot_rows = _snapshot_rows(scenarios, forecasts)

    # --- Hero KPIs ---
    hero_kpis: list[dict] = []
    snap_by_metric = {r["metric"]: r for r in snapshot_rows}
    for metric, label in [("Total Revenue", "Revenue (latest q)"),
                          ("Net Income", "Net income (latest q)"),
                          ("Total Debt", "Total debt")]:
        r = snap_by_metric.get(metric)
        if r:
            sub = (f"next q {r['next_forecast']}" if r["next_forecast"] != "—"
                   else "")
            col = "#4fc3f7"
            if r.get("exp_change") is not None:
                col = _pct_color(r["exp_change"], good_high=not r["debt_like"])
            hero_kpis.append({"label": label, "value": r["current"], "sub": sub, "color": col})
    if not health.empty:
        hv = health.iloc[-1].get("health_score")
        hero_kpis.append({"label": "Health score", "value": _fmt(hv, 0),
                          "sub": "0–100 composite", "color": _color_score(hv)})
    if distress is not None and not distress.empty:
        dv = distress.iloc[-1].to_dict().get("p_distress")
        hero_kpis.append({"label": "Distress prob.", "value": _pct(dv),
                          "sub": str(distress.iloc[-1].to_dict().get("distress_zone", "")),
                          "color": _color_prob(dv)})

    # --- Distress ---
    distress_row: dict | None = None
    distress_table = "<p class='muted'>No data.</p>"
    if distress is not None and not distress.empty:
        distress_row = distress.iloc[-1].to_dict()
        distress_table = _df_table(distress.select_dtypes(include="number").round(4))

    # --- Factor tables ---
    pred_corr_table = "<p class='muted'>No data.</p>"
    trend_table = "<p class='muted'>No data.</p>"
    income_decomp_table = "<p class='muted'>No data.</p>"
    pca_variance: list[float] = []
    pca_loadings_table = ""

    if factors:
        pc = factors.get("predictive_correlations")
        if pc is not None and not pc.empty:
            pred_corr_table = _df_table(pc)
        ts = factors.get("trend_statistics")
        if ts is not None and not ts.empty:
            trend_table = _df_table(ts)
        id_ = factors.get("income_decomposition")
        if id_ is not None and not id_.empty:
            income_decomp_table = _df_table(id_)
        pca = factors.get("pca") or {}
        pca_variance = pca.get("explained_variance_pct", [])
        loadings = pca.get("loadings")
        if loadings is not None and not loadings.empty:
            pca_loadings_table = _df_table(loadings)

    # --- Scenario table ---
    scenario_table = "<p class='muted'>No data.</p>"
    if scenarios:
        rows = []
        for metric, sim in scenarios.items():
            last_val = sim.get("last_value", "")
            for h, hdata in (sim.get("horizons") or {}).items():
                rows.append({
                    "Metric":            metric,
                    "Horizon (periods)": h,
                    "Last Value":        last_val,
                    "P(growth)":         f"{hdata.get('p_growth', 0)*100:.1f}%",
                    "P10":               _fmt(hdata.get("p10"), 0),
                    "P50 Median":        _fmt(hdata.get("p50_median"), 0),
                    "P90":               _fmt(hdata.get("p90"), 0),
                    "VaR 5%":            _fmt(hdata.get("var_5pct"), 0),
                    "Expected Δ%":       f"{hdata.get('expected_change_pct', 0):+.1f}%",
                })
        if rows:
            scenario_table = pd.DataFrame(rows).to_html(
                classes="data-table", border=0, index=False, na_rep="—")

    # --- Forecast table ---
    forecast_rows: list[dict] = []
    if forecasts:
        for stmt, metrics in forecasts.items():
            for metric, result in metrics.items():
                if not isinstance(result, dict):
                    continue
                st = result.get("stats") or {}
                fc = result.get("forecast")
                first_yhat = first_lo = first_hi = "—"
                if fc is not None and not fc.empty:
                    first_yhat = _fmt(fc["yhat"].iloc[0], 0)
                    first_lo   = _fmt(fc["yhat_lower"].iloc[0], 0)
                    first_hi   = _fmt(fc["yhat_upper"].iloc[0], 0)
                forecast_rows.append({
                    "Statement": stmt,
                    "Metric":    metric,
                    "Method":    result.get("method", "—"),
                    "Next yhat": first_yhat,
                    "Lower 80%": first_lo,
                    "Upper 80%": first_hi,
                    "Slope/period": _fmt(st.get("slope"), 4),
                    "Slope %/period": f"{st.get('slope_pct_per_period', float('nan')):.2f}%",
                    "R²":         _fmt(st.get("r_squared"), 4),
                    "p-value":    _fmt(st.get("p_value"), 4),
                    "Trend":      st.get("trend_model", "—"),
                    "Fit err %":  _fmt(st.get("mape_pct"), 2),
                })
    forecast_table = (
        pd.DataFrame(forecast_rows).to_html(classes="data-table", border=0, index=False, na_rep="—")
        if forecast_rows else "<p class='muted'>No data.</p>"
    )

    # --- Charts ---
    charts: list[tuple[str, str]] = []
    if income is not None and not income.empty:
        div = _chart_html([("Total Revenue", income.get("Total Revenue")),
                           ("Net Income", income.get("Net Income"))],
                          "Revenue & Net Income")
        if div:
            charts.append(("Revenue & Net Income", div))

    # New combined growth chart: Revenue, Net Income, Total Debt over time.
    growth_traces = []
    if income is not None and not income.empty:
        growth_traces.append(("Total Revenue", income.get("Total Revenue")))
        growth_traces.append(("Net Income", income.get("Net Income")))
    if balance is not None and not balance.empty:
        growth_traces.append(("Total Debt", balance.get("Total Debt")))
    if growth_traces:
        div = _chart_html(growth_traces, "Revenue, Net Income & Total Debt — Growth Over Time")
        if div:
            charts.append(("Revenue, Net Income & Total Debt — Growth Over Time", div))

    if cashflow is not None and not cashflow.empty:
        div = _chart_html([("Operating Cash Flow", cashflow.get("Operating Cash Flow")),
                           ("Free Cash Flow", cashflow.get("Free Cash Flow"))],
                          "Operating & Free Cash Flow")
        if div:
            charts.append(("Operating & Free Cash Flow", div))
    if not health.empty:
        div = _chart_html([("health_score", health.get("health_score"))],
                          "Health Score Over Time")
        if div:
            charts.append(("Health Score Over Time", div))

    # --- Prediction accuracy table (real vs predicted) ---
    accuracy_table_html = ""
    if prediction_accuracy is not None and not prediction_accuracy.empty:
        acc = prediction_accuracy.copy()
        for col in ("Predicted", "Actual (official)"):
            if col in acc.columns:
                acc[col] = acc[col].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
        for col in ("Error", "Abs Error"):
            if col in acc.columns:
                acc[col] = acc[col].map(lambda v: f"{v * 100:+.1f}%" if pd.notna(v) else "—")
        accuracy_table_html = acc.to_html(
            classes="data-table", border=0, index=False, na_rep="—")

    # --- Fund / price-only metrics ---
    fund_kpis: list[dict] = []
    fund_facts: list[tuple[str, str]] = []
    perf_rows: list[tuple[str, str, str]] = []   # (label, value, color)
    risk_rows: list[tuple[str, str]] = []
    level_rows: list[tuple[str, str, str]] = []
    if price_metrics:
        perf = price_metrics.get("performance", {})
        risk = price_metrics.get("risk", {})
        lev = price_metrics.get("levels", {})
        fund = price_metrics.get("fund", {})
        as_of = price_metrics.get("as_of", "")

        r1y = perf.get("1Y")
        if r1y is not None:
            fund_kpis.append({"label": "1-yr return", "value": f"{r1y:+.1f}%",
                              "sub": f"as of {as_of}", "color": _pct_color(r1y)})
        if risk.get("volatility") is not None:
            fund_kpis.append({"label": "Volatility (ann.)", "value": f"{risk['volatility']:.1f}%",
                              "sub": "daily σ × √252", "color": "#4fc3f7"})
        if risk.get("max_drawdown") is not None:
            fund_kpis.append({"label": "Max drawdown", "value": f"{risk['max_drawdown']:.1f}%",
                              "sub": "peak-to-trough", "color": "#ef5350"})
        if fund.get("expense_ratio") is not None:
            fund_kpis.append({"label": "Expense ratio", "value": f"{_fmt(fund['expense_ratio'], 2)}%",
                              "sub": "net", "color": "#4fc3f7"})

        for k in ("1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"):
            v = perf.get(k)
            if v is not None:
                perf_rows.append((k, f"{v:+.2f}%", _pct_color(v)))
        if price_metrics.get("cagr") is not None:
            c = price_metrics["cagr"]
            perf_rows.append(("CAGR (full history)", f"{c:+.2f}%", _pct_color(c)))

        if risk.get("volatility") is not None:
            risk_rows.append(("Annualised volatility", f"{risk['volatility']:.2f}%"))
        if risk.get("max_drawdown") is not None:
            risk_rows.append(("Max drawdown", f"{risk['max_drawdown']:.2f}%"))
        if risk.get("sharpe") is not None:
            risk_rows.append(("Sharpe ratio (rf=0)", f"{risk['sharpe']:.2f}"))
        if fund.get("beta") is not None:
            risk_rows.append(("Beta", _fmt(fund["beta"], 2)))

        if lev.get("price") is not None:
            level_rows.append(("Last price", _fmt(lev["price"], 2), "#e6e9f0"))
        if lev.get("high_52w") is not None:
            level_rows.append(("52-week high", _fmt(lev["high_52w"], 2), "#e6e9f0"))
        if lev.get("low_52w") is not None:
            level_rows.append(("52-week low", _fmt(lev["low_52w"], 2), "#e6e9f0"))
        if lev.get("pct_from_high") is not None:
            level_rows.append(("From 52-week high", f"{lev['pct_from_high']:+.1f}%",
                               _pct_color(lev["pct_from_high"])))
        if lev.get("above_200dma") is not None:
            level_rows.append(("Trend vs 200-day MA",
                               "Above ▲" if lev["above_200dma"] else "Below ▼",
                               "#66bb6a" if lev["above_200dma"] else "#ef5350"))

        for label, key, kind in [
            ("Category", "category", "raw"), ("Fund family", "family", "raw"),
            ("Type", "legal_type", "raw"), ("AUM", "aum", "money"),
            ("Yield", "yield", "pct_frac"), ("Inception", "inception", "raw"),
        ]:
            val = fund.get(key)
            if val in (None, ""):
                continue
            disp = _money(val) if kind == "money" else _pct(val) if kind == "pct_frac" else str(val)
            fund_facts.append((label, disp))

        hero_kpis = fund_kpis + hero_kpis

    # Price-history chart (funds have no statement charts).
    if prices is not None and not prices.empty and "Close" in prices.columns:
        div = _chart_html([("Close", prices["Close"])], "Price History")
        if div:
            charts.insert(0, ("Price History", div))

    # --- Render ---
    env = Environment(loader=BaseLoader())
    env.globals.update(pct=_pct, p_color=_color_prob, sev_color=_sev_color, fmt=_fmt,
                       signed=_signed, pct_bg=_pct_bg, prob_bar=_prob_bar)
    tmpl = env.from_string(_TEMPLATE)
    html = tmpl.render(
        ticker=ticker,
        info=info,
        verdict_color=verdict_color,
        hero_kpis=hero_kpis,
        health_boxes=health_boxes,
        snapshot_rows=snapshot_rows,
        distress_row=distress_row,
        distress_table=distress_table,
        signals=signals,
        factors=factors or {},
        pred_corr_table=pred_corr_table,
        trend_table=trend_table,
        income_decomp_table=income_decomp_table,
        pca_variance=pca_variance,
        pca_loadings_table=pca_loadings_table,
        scenarios=scenarios,
        scenario_table=scenario_table,
        forecast_rows=forecast_rows,
        forecast_table=forecast_table,
        charts=charts,
        highlights=highlights or [],
        opportunities=opportunities or [],
        decision=decision,
        accuracy_table_html=accuracy_table_html,
        data_confidence=data_confidence,
        price_metrics=price_metrics,
        fund_facts=fund_facts,
        perf_rows=perf_rows,
        risk_rows=risk_rows,
        level_rows=level_rows,
    )

    path = Path(output_path or f"{ticker}_report.html")
    path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved to '%s'.", path.resolve())
    return html
