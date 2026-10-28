#!/usr/bin/env bash
set -euo pipefail
# CPU/network preparation only. Models/simulation are never executed here.
: "${ROBODATA_DATA:?set ROBODATA_DATA to an external data/install directory}"
command -v uv >/dev/null || { echo 'Install uv in your own account first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 2; }
uv python install 3.12.13
uv venv --python 3.12.13 "$ROBODATA_DATA/envs/tools"
uv pip install --python "$ROBODATA_DATA/envs/tools/bin/python" huggingface-hub==1.22.0
printf 'Deployment Python: %s\n' "$ROBODATA_DATA/envs/tools/bin/python"
