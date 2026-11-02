#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
campaign=$1
worker=$2
group=$3
task=$4
eval_seed=$5
export ASPIRE_VULKAN_INDEX=$6 ASPIRE_SIM_GPU_UUID=$7
target=$8
layouts=$9
model=${10}
case "$campaign" in qwen_paired_*) ;; *) exit 2;; esac
[[ "$worker" =~ ^w[0-9][0-9]$ ]] || exit 2
[[ "$task" =~ ^[A-Za-z][A-Za-z0-9_]*$ ]] || exit 2
[[ "$group" =~ ^${task}_s0_l[0-9]+$ && "$eval_seed" == 0 ]] || exit 2
[[ "$layouts" =~ ^[0-9]+$ && "$group" == "${task}_s0_l${layouts}" ]] || exit 2
case "$target" in 1) ;; *) exit 2;; esac
[[ "$model" == qwen38-27b ]] || exit 2
: "${ASPIRE_PERCEPTION_CONFIG:?provider configuration required}"
code=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
profile="/work/aspire_validate/harness_profiles/$campaign"
export ASPIRE_TERRA_MANIFEST="/work/aspire_validate/full_results/$campaign/manifest.json"
export ASPIRE_TERRA_GROUP="$group" ASPIRE_TERRA_WORKER="$worker"
cd /work/RoboDojo
export PATH="/work/.venv/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$profile/qwen_full:$profile/ASPIRE/src:$profile/qwen_integration:/work/RoboDojo:/work/RoboDojo/XPolicyLab"
export ROBODOJO_ROOT=/work/RoboDojo ASPIRE_FULL_MODE=evaluation
export EVAL_NUM="$target" EVAL_LAYOUT_IDS="$layouts" EVAL_SEED="$eval_seed"
export ROBODOJO_MAX_INPROC_RESTARTS=0 ROBODOJO_NUM_ENVS=1
export ROBODOJO_UNTILED_CAMERAS=0 ROBODOJO_PATH_TRACING=1
export ROBODOJO_ENABLE_METRIC_DEPTH=1 ROBODOJO_ACTION_TYPE=joint
export ENV_CFG=arx_x5 ROBODOJO_ENV_CFG=arx_x5
export ROBODOJO_RUN_ID="${campaign}_${group}"
export ASPIRE_MODEL_NAME="$model" ASPIRE_RELAY_DIR=/aspire-relay
export ASPIRE_OUTPUT_DIR="/work/aspire_validate/full_results/$campaign/items/$group"
export ASPIRE_SIM_LOG="${ASPIRE_SIM_LOG:-$ASPIRE_OUTPUT_DIR/simulator.log}"
export ASPIRE_CUDA_DEVICE_INDEX=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
# Disable optional Hub/telemetry helpers; keep rendering and robot APIs unchanged.
export OMNICLIENT_HUB_MODE=disabled OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA=1
unset VLLM_API_KEY ASPIRE_MAX_ITERATIONS ASPIRE_EPISODE_ID OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN ARK_API_KEY ARK_API_KEY_BACKUP SAFEHARNESS_KEY_FILE TS_AUTHKEY TS_AUTH_KEY TAILSCALE_AUTHKEY
mkdir -p "$ASPIRE_OUTPUT_DIR"
# Pin the inspected NGX library before loading the narrowly scoped child reaper.
# This leaves NGX feature detection/rendering enabled. It only collects the
# known missing-updater child, whose original caller never waits for it.
export NGX_REAPER_LIBRARY=/opt/robodojo-driver-libs/libnvidia-ngx.so.1
/work/.venv/bin/python - "$code" "$NGX_REAPER_LIBRARY" "$ASPIRE_OUTPUT_DIR/ngx_reaper_preflight.json" <<'PY'
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from terra_platform import verify_ngx_library
pathlib.Path(sys.argv[3]).write_text(json.dumps(verify_ngx_library(sys.argv[2]), sort_keys=True, indent=2) + '\n')
PY
[[ -f "$code/ngx_child_reaper.so" ]] || exit 2
export NGX_REAPER_LOG="$ASPIRE_OUTPUT_DIR/ngx_child_reaper.jsonl"
# The supervisor archives any earlier unscored attempt before this fresh launch.
: > "$NGX_REAPER_LOG"
export NGX_REAPER_SHIM="$code/ngx_child_reaper.so"
# NVIDIA's launcher sets libcarb.so as LD_PRELOAD. Add our pinned shim only
# after that setup, preserving the vendor library and all Python environment.
cat > /tmp/terra-ngx-python-exec.sh <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
: "${NGX_REAPER_SHIM:?pinned NGX child collector required}"
export LD_PRELOAD="$NGX_REAPER_SHIM${LD_PRELOAD:+:$LD_PRELOAD}"
exec /work/scripts/venv_python_exec.sh "$@"
WRAPPER
chmod 700 /tmp/terra-ngx-python-exec.sh
export PYTHONEXE=/tmp/terra-ngx-python-exec.sh
exec /isaac-sim/python.sh "$code/terra_entry.py" \
  --task_name "$task" --env_cfg_type arx_x5 --num_envs 1 --enable_cameras \
  --device_id 0 --policy_name ASPIRE_Pilot --port 0 --protocol ws \
  --additional_info aspire_terra_paired --seed "$eval_seed" --host 127.0.0.1 --headless \
  --experience /isaac-sim/apps/isaacsim.exp.base.python.kit \
  --kit_args="--/isaaclab/cameras_enabled=true --/crashreporter/enabled=false --/renderer/multiGpu/enabled=false --/renderer/activeGpu=$ASPIRE_VULKAN_INDEX --/structuredLog/enable=false --/telemetry/enableAnonymousData=false"
