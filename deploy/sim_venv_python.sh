#!/usr/bin/env bash
set -euo pipefail
: "${ROBODATA_SIM_VENV:=/opt/robodata-env/sim}"
export PYTHONPATH="$ROBODATA_SIM_VENV/lib/python3.12/site-packages:/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROBODATA_SIM_VENV/bin/python" "$@"
