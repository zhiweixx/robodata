#!/usr/bin/env bash
set -euo pipefail
export PYTHONEXE=/work/scripts/venv_python_exec.sh
exec /isaac-sim/python.sh "$@"
