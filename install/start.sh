#!/usr/bin/env sh
# Algo Agent launcher for macOS / Linux / Android (Termux).
# First run installs into ./.venv, later runs just start the app.
#   ./install/start.sh                            # this device only
#   ./install/start.sh --host 0.0.0.0 --pin 4821  # also reachable from your tablet
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || { echo "Python 3.10+ is required"; exit 1; }
if [ -n "$TERMUX_VERSION" ]; then
  # Termux: numpy/pandas come prebuilt from Termux packages; no venv needed.
  python -c "import pandas, numpy" 2>/dev/null || {
    echo "Installing Python packages from Termux (one time)..."
    pkg install -y python-numpy python-pandas 2>/dev/null || {
      pkg install -y tur-repo && pkg install -y python-pandas; } ; }
  python -c "import requests" 2>/dev/null || python -m pip install requests
  python -m pip install --no-deps -e . >/dev/null
  exec python -m algo_agent app --no-browser "$@"
fi
if [ ! -d .venv ]; then
  echo "First run: creating environment..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e .
fi
exec .venv/bin/python -m algo_agent app "$@"
