#!/usr/bin/env bash
set -euo pipefail
: "${ACCEPT_EULA:?explicit Isaac Sim license acceptance is required}"
[[ "$ACCEPT_EULA" == Y ]] || exit 2
/isaac-sim/license.sh
: "${ROBODATA_SIM_VENV:=/opt/robodata-env/sim}"
if [[ ! -x "$ROBODATA_SIM_VENV/bin/python" ]]; then
  /isaac-sim/python.sh -m venv --system-site-packages "$ROBODATA_SIM_VENV"
fi
unset CUROBO_USE_PYBIND CUDA_HOME CUDA_PATH TRITON_PTXAS_PATH
export PATH="$ROBODATA_SIM_VENV/bin:$PATH" VIRTUAL_ENV="$ROBODATA_SIM_VENV"
export PIP_CACHE_DIR=/tmp/robodata-setup/pip PYTHONNOUSERSITE=1 PIP_NO_COMPILE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONPATH=/work/RoboDojo:/work/RoboDojo/XPolicyLab:/opt/robodata/src:/opt/robodata/third_party/aspire/src
py() { /opt/robodata/deploy/sim_python.sh "$@"; }
cd /work/RoboDojo
if [[ ! -e third_party/IsaacLab/_isaac_sim ]]; then ln -s /isaac-sim third_party/IsaacLab/_isaac_sim; fi
py -m pip install --constraint /opt/robodata/deploy/locks/simulator-bundled-constraints.txt --requirement /opt/robodata/deploy/locks/simulator-overlay.txt
for name in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_contrib; do
  py -m pip install --no-deps --no-build-isolation -e "third_party/IsaacLab/source/$name"
done
py -m pip install --constraint /opt/robodata/deploy/locks/simulator-bundled-constraints.txt --no-build-isolation -e 'third_party/curobo[cu12]'
py - <<'PY'
import json,os,sys
from pathlib import Path
import imageio_ffmpeg
venv=Path(os.environ['ROBODATA_SIM_VENV'])
assert sys.version_info[:2]==(3,12),'Isaac6 overlay is pinned to Python3.12'
ffmpeg=venv/'bin/ffmpeg'
if not ffmpeg.exists():ffmpeg.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
root=venv/'lib/python3.12/site-packages/nvidia';root.mkdir(parents=True,exist_ok=True)
bundle=Path('/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia')
for name in ('cuda_nvrtc','cuda_runtime'):
    target=bundle/name
    if not target.is_dir():raise RuntimeError(f'missing pinned Isaac CUDA headers: {name}')
    link=root/name
    if not link.exists():link.symlink_to(target,target_is_directory=True)
import importlib.metadata as m
(venv/'installation.json').write_text(json.dumps({'python':sys.version,'packages':{d.metadata['Name']:d.version for d in m.distributions()},'cuda_arch_list':os.environ.get('TORCH_CUDA_ARCH_LIST'),'scope':'dependency installation only, not rendering/evaluation validation'},indent=2)+'\n')
PY
printf 'Simulator dependencies installed; an actual RGB-D/control smoke is still required.\n'
