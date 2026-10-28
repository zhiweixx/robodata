#!/usr/bin/env bash
set -euo pipefail
export PYTHONEXE=/opt/robodata/deploy/sim_venv_python.sh
exec /isaac-sim/python.sh "$@"
