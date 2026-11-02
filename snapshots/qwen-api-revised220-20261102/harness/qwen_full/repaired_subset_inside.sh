#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
task=$1
campaign=$2
eval_seed=$3
export ASPIRE_VULKAN_INDEX=$4 ASPIRE_SIM_GPU_UUID=$5
subset_target=$6
subset_layout_ids=$7
case "$campaign" in qwen38_subset20_repaired_*) ;; *) exit 2;; esac
case "$eval_seed" in 0) ;; *) exit 2;; esac
case "$subset_target" in 1|6|10|20) ;; *) exit 2;; esac
: "${ASPIRE_PERCEPTION_CONFIG:?provider configuration required}"
ASPIRE_BOOTSTRAP_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export ASPIRE_BOOTSTRAP_INSIDE="$ASPIRE_BOOTSTRAP_DIRECTORY/repaired_subset_inside.sh"
export ASPIRE_REPAIRED_CAMPAIGN="$campaign"
export ASPIRE_REPAIRED_PROFILE="/work/aspire_validate/harness_profiles/$campaign"
[[ "$ASPIRE_BOOTSTRAP_DIRECTORY" == "$ASPIRE_REPAIRED_PROFILE/qwen_full" ]] || exit 2
cd /work/RoboDojo
export PATH="/work/.venv/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ASPIRE_REPAIRED_PROFILE/qwen_full:$ASPIRE_REPAIRED_PROFILE/ASPIRE/src:$ASPIRE_REPAIRED_PROFILE/qwen_integration:/work/RoboDojo:/work/RoboDojo/XPolicyLab"
export ROBODOJO_ROOT=/work/RoboDojo ASPIRE_FULL_MODE=evaluation
export EVAL_NUM="$subset_target" EVAL_LAYOUT_IDS="$subset_layout_ids" EVAL_SEED="$eval_seed"
export ROBODOJO_MAX_INPROC_RESTARTS=3 ROBODOJO_NUM_ENVS=1
export ROBODOJO_UNTILED_CAMERAS=0 ROBODOJO_PATH_TRACING=1
export ROBODOJO_ENABLE_METRIC_DEPTH=1 ROBODOJO_ACTION_TYPE=joint
export ENV_CFG=arx_x5 ROBODOJO_ENV_CFG=arx_x5
export ROBODOJO_RUN_ID="${campaign}_${task}_s${eval_seed}"
export ASPIRE_MODEL_NAME=Qwen/Qwen3.8-27B ASPIRE_RELAY_DIR=/aspire-relay
export ASPIRE_OUTPUT_DIR="/work/aspire_validate/full_results/$campaign/groups/${task}_s${eval_seed}"
export ASPIRE_SIM_LOG="${ASPIRE_SIM_LOG:-$ASPIRE_OUTPUT_DIR/simulator.log}"
export ASPIRE_CUDA_DEVICE_INDEX="${ASPIRE_CUDA_DEVICE_INDEX:-0}"
unset ASPIRE_MAX_ITERATIONS ASPIRE_EPISODE_ID OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN ARK_API_KEY ARK_API_KEY_BACKUP SAFEHARNESS_KEY_FILE
mkdir -p "$ASPIRE_OUTPUT_DIR"
exec /work/scripts/python_exec.sh "$ASPIRE_BOOTSTRAP_DIRECTORY/repaired_subset_entry.py" \
  --task_name "$task" --env_cfg_type arx_x5 --num_envs 1 --enable_cameras \
  --device_id "$ASPIRE_CUDA_DEVICE_INDEX" --policy_name ASPIRE_Pilot --port 0 --protocol ws \
  --additional_info aspire_qwen_full --seed "$eval_seed" --host 127.0.0.1 --headless \
  --experience /isaac-sim/apps/isaacsim.exp.base.python.kit \
  --kit_args="--/isaaclab/cameras_enabled=true --/crashreporter/enabled=false --/renderer/multiGpu/enabled=false --/renderer/activeGpu=$ASPIRE_VULKAN_INDEX"
