# AllianceAI

Open-source AI financial analysis engine. Give it a ticker and it pulls the
company's financials, scores its health, flags risks, forecasts key metrics,
and renders a full HTML report — all from the command line.

> **Disclaimer:** AllianceAI is a research and educational tool. Nothing it
> produces is investment advice.

## Features

Running `analyse(ticker)` executes an end-to-end pipeline:

1. Fetch financial data from Yahoo Finance + SEC EDGAR (cached in DuckDB)
2. Compute financial ratios
3. Exploratory data analysis (trends, outliers, completeness)
4. Health scoring (AllianceAI composite + Altman Z-score)
5. Risk-signal detection
6. Anomaly detection (IsolationForest)
7. Distress-probability ensemble (Ohlson, Zmijewski, Altman)
8. Factor analysis (predictive correlations, PCA, income decomposition)
9. Monte Carlo scenario analysis
10. Forecasting of key metrics (Prophet / Holt-Winters)
11. HTML report generation

It also has a **walk-forward backtest** mode that replays the forecaster
through history to grade past predictions against the actuals.

## Requirements

- Python 3.10+
- See [`requirements.txt`](requirements.txt) / [`pyproject.toml`](pyproject.toml)

## Installation

```bash
# clone, then from the project root:
python -m venv alliance-venv
source alliance-venv/Scripts/activate   # Windows (Git Bash)
# source alliance-venv/bin/activate     # macOS/Linux

pip install -e .
```

## Configuration

Copy the example env file and fill in your own keys:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `anthropic`, `openrouter`, `auto`, or `none` (rule-based highlights only) |
| `ANTHROPIC_API_KEY` | Claude API key (if using Anthropic directly) |
| `OPENROUTER_API_KEY` | OpenRouter key (alternative LLM provider) |
| `EDGAR_USER_AGENT` | Real contact string required by SEC fair-access rules |

The LLM is used only for the natural-language *highlights*; the quantitative
analysis runs without any API key.

## Usage

```bash
# single ticker
allianceai AAPL

# multiple tickers, annual statements, skip forecasting
allianceai AAPL MSFT --annual --no-forecast

# write reports to a folder
allianceai VNQ SPY --output-dir reports/

# walk-forward backtest
allianceai AAPL --backtest
```

Or without installing the console script:

```bash
python main.py AAPL
```

### Key options

| Flag | Description |
|---|---|
| `--annual` | Use annual statements instead of quarterly |
| `--no-forecast` | Skip Prophet/Holt-Winters forecasting (faster) |
| `--no-report` | Skip HTML report generation |
| `--output-dir DIR` | Directory for HTML reports (default: current dir) |
| `--backtest` | Replay the forecaster through history to grade predictions |

## Project structure

```
allianceai/
  analysis/    ratios, health score, risk signals, EDA, decision, highlights
  models/      forecaster, anomaly, distress, factors, scenarios, backtest
  data/        Yahoo Finance + SEC EDGAR fetchers
  reports/     HTML report generation
  core/        config, storage (DuckDB), logging
  cli.py       command-line entry point
  orchestrator.py   end-to-end pipeline
tests/         pytest unit tests
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

See repository for license details.
