#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
task=$1
campaign=$2
eval_seed=$3
export ASPIRE_VULKAN_INDEX=$4
export ASPIRE_SIM_GPU_UUID=$5
subset_target=$6
subset_layout_ids=$7
case "$eval_seed" in 0) ;; *) echo 'subset eval_seed must be0' >&2; exit 2;; esac
case "$subset_target" in 1|6|10|20) ;; *) echo 'invalid subset target' >&2; exit 2;; esac
case "$campaign" in qwen38_subset20_*) ;; *) echo 'invalid subset campaign' >&2; exit 2;; esac
cd /work/RoboDojo
export PATH="/work/.venv/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/work/aspire_validate/qwen_full:/work/aspire_validate/ASPIRE/src:/work/aspire_validate/qwen_integration:/work/RoboDojo:/work/RoboDojo/XPolicyLab
export ROBODOJO_ROOT=/work/RoboDojo
export ASPIRE_FULL_MODE=evaluation EVAL_NUM="$subset_target"
export EVAL_LAYOUT_IDS="$subset_layout_ids"
export ASPIRE_SUBSET_MANIFEST="/work/aspire_validate/full_results/$campaign/manifest.json"
export EVAL_SEED="$eval_seed"
export ROBODOJO_MAX_INPROC_RESTARTS=3 ROBODOJO_NUM_ENVS=1
export ROBODOJO_UNTILED_CAMERAS=0 ROBODOJO_PATH_TRACING=1
export ROBODOJO_ENABLE_METRIC_DEPTH=1 ROBODOJO_ACTION_TYPE=joint
export ENV_CFG=arx_x5 ROBODOJO_ENV_CFG=arx_x5
export ROBODOJO_RUN_ID="${campaign}_${task}_s${eval_seed}"
export ASPIRE_MODEL_NAME=Qwen/Qwen3.8-27B
export ASPIRE_RELAY_DIR=/aspire-relay
export ASPIRE_OUTPUT_DIR="/work/aspire_validate/full_results/$campaign/groups/${task}_s${eval_seed}"
export ASPIRE_SIM_LOG="${ASPIRE_SIM_LOG:-$ASPIRE_OUTPUT_DIR/simulator.log}"
export ASPIRE_CUDA_DEVICE_INDEX="${ASPIRE_CUDA_DEVICE_INDEX:-0}"
unset ASPIRE_MAX_ITERATIONS ASPIRE_EPISODE_ID OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN ARK_API_KEY ARK_API_KEY_BACKUP SAFEHARNESS_KEY_FILE
mkdir -p "$ASPIRE_OUTPUT_DIR"
exec /work/scripts/python_exec.sh /work/aspire_validate/qwen_full/subset_entry.py \
  --task_name "$task" --env_cfg_type arx_x5 --num_envs 1 --enable_cameras \
  --device_id "$ASPIRE_CUDA_DEVICE_INDEX" \
  --policy_name ASPIRE_Pilot --port 0 --protocol ws \
  --additional_info aspire_qwen_full --seed "$eval_seed" --host 127.0.0.1 --headless \
  --experience /isaac-sim/apps/isaacsim.exp.base.python.kit \
  --kit_args="--/isaaclab/cameras_enabled=true --/crashreporter/enabled=false --/renderer/multiGpu/enabled=false --/renderer/activeGpu=$ASPIRE_VULKAN_INDEX"
