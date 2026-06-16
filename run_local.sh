#!/usr/bin/env bash
# ==============================================================================
# AllianceAI — local analysis runner
#
# Usage:
#   bash run_local.sh AAPL
#   bash run_local.sh AAPL MSFT TSLA
#   bash run_local.sh AAPL --annual
#   bash run_local.sh SPY --no-forecast
#
# Walk-forward backtest (train the prediction loop from history):
#   bash run_local.sh AAPL --backtest                  # full deep history
#   bash run_local.sh AAPL --backtest --bt-max-steps 12   # recent windows only
#   bash run_local.sh AAPL MSFT --backtest --bt-step 2
#
# Uses the bundled alliance-venv automatically — no need to activate it first.
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$PROJECT_DIR/logs/allianceai.log"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ------------------------------------------------------------------
# Resolve the interpreter: prefer the bundled venv, fall back to PATH.
# (The venv's activate scripts had stale absolute paths after the project
#  was moved, so we call its python directly rather than relying on `source`.)
# ------------------------------------------------------------------
if [ -x "$PROJECT_DIR/alliance-venv/Scripts/python.exe" ]; then
    PYTHON="$PROJECT_DIR/alliance-venv/Scripts/python.exe"   # Windows / Git Bash
elif [ -x "$PROJECT_DIR/alliance-venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/alliance-venv/bin/python"           # Linux / macOS
elif command -v python &>/dev/null; then
    PYTHON="python"
    warn "Bundled alliance-venv not found — using system python (deps may be missing)."
else
    error "No Python interpreter found. Create the venv and install the package:"
    error "  python -m venv alliance-venv"
    error "  alliance-venv/Scripts/python.exe -m pip install -e ."
    exit 1
fi

PYTHON_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PYTHON_VER  ($PYTHON)"

# Ensure the package is importable; install editable if not.
if ! "$PYTHON" -c "import allianceai" &>/dev/null; then
    warn "allianceai package not importable — installing in editable mode…"
    "$PYTHON" -m pip install -e "$PROJECT_DIR" --quiet
fi

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/reports"

if [ "$#" -lt 1 ]; then
    error "Usage: bash run_local.sh TICKER [TICKER ...] [--annual] [--no-forecast] [--backtest ...]"
    exit 1
fi

info "Logs → $LOG_FILE"
info "Reports → $PROJECT_DIR/reports/"
echo ""

"$PYTHON" "$PROJECT_DIR/main.py" --output-dir "$PROJECT_DIR/reports" "$@"

# Backtest mode writes to the DB, not HTML — skip the report tally for it.
case " $* " in
    *" --backtest "*) echo ""; info "Backtest complete — graded predictions stored to the prediction DB."; exit 0 ;;
esac

echo ""
REPORTS=$(ls "$PROJECT_DIR/reports/"*.html 2>/dev/null | wc -l)
if [ "$REPORTS" -gt 0 ]; then
    info "Analysis complete.  Reports saved:"
    ls "$PROJECT_DIR/reports/"*.html | while read f; do
        info "  $(basename "$f")"
    done
else
    warn "No reports generated — check the log:"
    echo "    tail -50 $LOG_FILE"
fi
