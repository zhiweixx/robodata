#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
task=$1
comparison=$2
arm=$3
export ASPIRE_VULKAN_INDEX=$4
export ASPIRE_SIM_GPU_UUID=$5
paired_target=$6
paired_layout_ids=$7
case "$comparison" in qwen38_paired_*) ;; *) exit 2;; esac
case "$arm" in baseline|fixed) ;; *) exit 2;; esac
case "$paired_target" in 1|2) ;; *) exit 2;; esac
campaign="${comparison}_${arm}"
export ASPIRE_PAIRED_ARM="$arm"
export ASPIRE_PAIRED_PROFILE="/work/aspire_validate/paired_profiles/$comparison/$arm"
export ASPIRE_PAIRED_MANIFEST="/work/aspire_validate/full_results/$comparison/manifest.json"
cd /work/RoboDojo
export PATH="/work/.venv/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ASPIRE_PAIRED_PROFILE/qwen_full:$ASPIRE_PAIRED_PROFILE/ASPIRE/src:$ASPIRE_PAIRED_PROFILE/qwen_integration:/work/RoboDojo:/work/RoboDojo/XPolicyLab"
export ROBODOJO_ROOT=/work/RoboDojo ASPIRE_FULL_MODE=evaluation
export EVAL_NUM="$paired_target" EVAL_LAYOUT_IDS="$paired_layout_ids" EVAL_SEED=0
export ROBODOJO_MAX_INPROC_RESTARTS=3 ROBODOJO_NUM_ENVS=1
export ROBODOJO_UNTILED_CAMERAS=0 ROBODOJO_PATH_TRACING=1
export ROBODOJO_ENABLE_METRIC_DEPTH=1 ROBODOJO_ACTION_TYPE=joint
export ENV_CFG=arx_x5 ROBODOJO_ENV_CFG=arx_x5
export ROBODOJO_RUN_ID="${campaign}_${task}_s0"
export ASPIRE_MODEL_NAME=Qwen/Qwen3.8-27B ASPIRE_RELAY_DIR=/aspire-relay
export ASPIRE_OUTPUT_DIR="/work/aspire_validate/full_results/$campaign/groups/${task}_s0"
export ASPIRE_SIM_LOG="${ASPIRE_SIM_LOG:-$ASPIRE_OUTPUT_DIR/simulator.log}"
export ASPIRE_CUDA_DEVICE_INDEX="${ASPIRE_CUDA_DEVICE_INDEX:-0}"
unset ASPIRE_MAX_ITERATIONS ASPIRE_EPISODE_ID OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN ARK_API_KEY ARK_API_KEY_BACKUP SAFEHARNESS_KEY_FILE
mkdir -p "$ASPIRE_OUTPUT_DIR"
exec /work/scripts/python_exec.sh /work/aspire_validate/qwen_full/paired_entry.py \
  --task_name "$task" --env_cfg_type arx_x5 --num_envs 1 --enable_cameras \
  --device_id "$ASPIRE_CUDA_DEVICE_INDEX" --policy_name ASPIRE_Pilot --port 0 --protocol ws \
  --additional_info aspire_qwen_full --seed 0 --host 127.0.0.1 --headless \
  --experience /isaac-sim/apps/isaacsim.exp.base.python.kit \
  --kit_args="--/isaaclab/cameras_enabled=true --/crashreporter/enabled=false --/renderer/multiGpu/enabled=false --/renderer/activeGpu=$ASPIRE_VULKAN_INDEX"
