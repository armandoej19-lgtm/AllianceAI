"""
AllianceAI CLI entry point.

Usage examples:
    allianceai AAPL
    allianceai AAPL --annual --no-forecast
    allianceai VNQ SPY MSFT --output-dir reports/
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from allianceai.core.logging_config import get_logger
from allianceai.orchestrator import analyse

console = Console()
logger = get_logger(__name__)


def _print_summary(result: dict) -> None:
    ticker = result["ticker"]
    info   = result.get("info", {})
    health = result.get("health")
    signals = result.get("signals", [])

    dc = result.get("data_confidence") or {}
    dc_color = {"HIGH": "green", "MODERATE": "yellow", "LOW": "red"}.get(dc.get("level"), "white")
    dc_line = (f"\n[dim]Analysis confidence:[/] [bold {dc_color}]{dc['level']}[/] "
               f"[dim]({dc['score']}/100 · {' '.join(dc.get('notes', []))})[/]") if dc else ""
    console.print(Panel(
        f"[bold cyan]{info.get('longName', ticker)}[/] ([yellow]{ticker}[/])\n"
        f"[dim]{info.get('sector', '')} · {info.get('industry', '')}[/]"
        f"{dc_line}",
        title="AllianceAI Financial Analysis",
    ))

    if health is not None and not health.empty:
        latest = health.iloc[-1]
        t = Table(show_header=True, header_style="bold blue")
        t.add_column("Metric", style="bold")
        t.add_column("Score", justify="right")
        for col in ["health_score", "liquidity_subscore", "leverage_subscore",
                    "profitability_subscore", "cashflow_subscore"]:
            val = latest.get(col, float("nan"))
            label = col.replace("_subscore", "").replace("_", " ").title()
            color = "green" if val >= 70 else ("yellow" if val >= 50 else "red")
            t.add_row(label, f"[{color}]{val:.0f}/100[/]")
        console.print(t)

    highlights = result.get("highlights") or []
    if highlights:
        console.print("\n[bold]Highlights:[/]")
        for h in highlights:
            console.print(f"  [cyan]*[/] {h}")

    opportunities = result.get("opportunities") or []
    if opportunities:
        console.print("\n[bold green]Opportunity Outlook:[/]")
        for o in opportunities:
            console.print(f"  [green]+[/] {o}")

    if signals:
        console.print("\n[bold]Risk Signals:[/]")
        sev_color = {"CRITICAL": "red", "HIGH": "orange3", "MEDIUM": "yellow", "LOW": "green"}
        for s in signals:
            c = sev_color.get(s.severity, "white")
            console.print(f"  [{c}][{s.severity}][/] {s.name} — {s.explanation[:100]}…")

    decision = result.get("decision")
    if decision:
        v_color = {"FAVORABLE": "green", "SELECTIVE": "green3", "WATCH": "yellow",
                   "CAUTION": "orange3", "UNFAVORABLE": "red"}.get(decision["verdict"], "white")
        conf = decision.get("confidence") or {}
        c_color = {"HIGH": "green", "MODERATE": "yellow", "LOW": "red"}.get(conf.get("level"), "white")
        console.print(Panel(
            f"[bold {v_color}]{decision['verdict']}[/] — {decision['score']}/100"
            f"   [dim]·[/]   Confidence: [bold {c_color}]{conf.get('level', 'N/A')}[/]"
            f" ({conf.get('score', 0)}/100)\n"
            f"{decision['summary']}\n"
            f"[cyan]{decision['timing']}[/]\n"
            f"[dim]{decision['disclaimer']}[/]",
            title="Investment Stance",
        ))

    acc = result.get("prediction_accuracy")
    if acc is not None and not acc.empty:
        console.print("\n[bold]Real Data vs Past Predictions:[/]")
        t = Table(show_header=True, header_style="bold blue")
        for col in ["Metric", "Period", "Predicted", "Actual (official)", "Error"]:
            t.add_column(col, justify="right")
        for _, r in acc.tail(10).iterrows():
            err = r["Error"]
            err_str = f"{err * 100:+.1f}%" if err == err else "-"
            err_color = "green" if abs(err or 1) < 0.10 else "yellow" if abs(err or 1) < 0.25 else "red"
            t.add_row(str(r["Metric"]), str(r["Period"]),
                      f"{r['Predicted']:,.0f}", f"{r['Actual (official)']:,.0f}",
                      f"[{err_color}]{err_str}[/]")
        console.print(t)

    path = result.get("report_path")
    if path:
        console.print(f"\n[dim]Report saved to:[/] [link={path}]{path}[/link]")


def _run_backtest(args) -> None:
    """Walk-forward backtest: seed graded predictions from historical data."""
    import time
    from allianceai.data.fetcher import fetch_all
    from allianceai.models.backtest import backtest_ticker, backtest_summary

    method = None if args.bt_prophet else "holt_winters"
    for ticker in args.tickers:
        console.rule(f"[bold magenta]Backtest · {ticker}")
        t0 = time.perf_counter()
        try:
            data = fetch_all(ticker, quarterly=not args.annual)
            graded = backtest_ticker(
                ticker, data["income"], data["balance"], data["cashflow"],
                min_train=args.bt_min_train, step=args.bt_step,
                max_steps=args.bt_max_steps, horizon=args.bt_horizon, method=method,
            )
        except Exception as exc:
            logger.exception("Backtest failed for '%s'.", ticker)
            console.print(f"[red]Backtest error for {ticker}: {exc}[/]")
            continue

        elapsed = time.perf_counter() - t0
        if graded.empty:
            console.print(f"[yellow]{ticker}: not enough history to backtest.[/]")
            continue

        summary = backtest_summary(graded)
        console.print(f"[green]Generated {len(graded)} graded predictions[/] "
                      f"in [bold]{elapsed:.1f}s[/] "
                      f"(method: {'Prophet' if args.bt_prophet else 'Holt-Winters'}).")
        t = Table(show_header=True, header_style="bold magenta")
        for col in ["Statement", "Metric", "Samples", "Calib. bias", "Reliability"]:
            t.add_column(col, justify="right")
        rel_color = {"good": "green", "fair": "yellow", "noisy": "red"}
        for _, r in summary.iterrows():
            rc = rel_color.get(r["reliability"], "white")
            t.add_row(r["statement"], r["metric"], str(int(r["samples"])),
                      f"{r['calibration_bias_pct']:+.1f}%",
                      f"[{rc}]{r['reliability']}[/]")
        console.print(t)
        console.print("[dim]'Calib. bias' is the outlier-filtered correction calibration will apply. "
                      "'noisy' metrics (mixed annual/quarterly history, seasonality) contribute "
                      "little after filtering.[/]\n")


def main() -> None:
    # Windows consoles default to cp1252, which can't encode characters like the
    # Unicode minus (−, U+2212) that appear in some summary strings.  Force UTF-8
    # so the rich summary never crashes the run after the report is already saved.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="allianceai",
        description="Open-source AI financial analysis engine.",
    )
    parser.add_argument("tickers", nargs="+", help="One or more ticker symbols (e.g. AAPL VNQ SPY).")
    parser.add_argument("--annual",      action="store_true", help="Use annual statements instead of quarterly.")
    parser.add_argument("--no-forecast", action="store_true", help="Skip time-series forecasting.")
    parser.add_argument("--no-report",   action="store_true", help="Skip HTML report generation.")
    parser.add_argument("--output-dir",  default=".", help="Directory for HTML reports (default: current dir).")

    bt = parser.add_argument_group("backtest (walk-forward training of the prediction loop)")
    bt.add_argument("--backtest", action="store_true",
                    help="Replay the forecaster through history to seed graded predictions.")
    bt.add_argument("--bt-min-train", type=int, default=8,
                    help="Min periods of history before the first forecast (default 8 = 2y).")
    bt.add_argument("--bt-step", type=int, default=1,
                    help="Quarters to advance per step (1 = every quarter; 2+ = faster/sparser).")
    bt.add_argument("--bt-max-steps", type=int, default=None,
                    help="Cap forecasts to the most recent N windows (default: full history).")
    bt.add_argument("--bt-horizon", type=int, default=1,
                    help="Periods ahead to predict and grade (default 1 = next quarter).")
    bt.add_argument("--bt-prophet", action="store_true",
                    help="Use Prophet during backtest (slower, more precise). Default: Holt-Winters.")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.backtest:
        _run_backtest(args)
        return

    for ticker in args.tickers:
        console.rule(f"[bold blue]{ticker}")
        try:
            result = analyse(
                ticker=ticker,
                quarterly=not args.annual,
                output_dir=args.output_dir,
                skip_forecast=args.no_forecast,
                skip_report=args.no_report,
            )
            if "error" in result:
                console.print(f"[red]Error:[/] {result['error']}")
            else:
                _print_summary(result)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
            sys.exit(0)
        except Exception as exc:
            logger.exception("Unhandled error for ticker '%s'.", ticker)
            console.print(f"[red]Unhandled error for {ticker}: {exc}[/]")


if __name__ == "__main__":
    main()

