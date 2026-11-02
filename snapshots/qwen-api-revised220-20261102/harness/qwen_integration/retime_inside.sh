#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /work/RoboDojo
export PATH="/work/.venv/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/work/aspire_validate/ASPIRE/src:/work/aspire_validate/integration:/work/RoboDojo:/work/RoboDojo/XPolicyLab
export ROBODOJO_ROOT=/work/RoboDojo
export EVAL_NUM=1 EVAL_LAYOUT_IDS=0 EVAL_SEED=0
export ROBODOJO_MAX_INPROC_RESTARTS=0 ROBODOJO_NUM_ENVS=1
export ROBODOJO_UNTILED_CAMERAS=0 ROBODOJO_PATH_TRACING=1
export ROBODOJO_ENABLE_METRIC_DEPTH=1 ROBODOJO_ACTION_TYPE=joint
export ENV_CFG=arx_x5 ROBODOJO_ENV_CFG=arx_x5
export ROBODOJO_RUN_ID="aspire_retime_${1:?Slurm job ID required}"
export ASPIRE_MODE=retime_diagnostic
export ASPIRE_OUTPUT_DIR="/work/aspire_validate/results/$ROBODOJO_RUN_ID"
unset OPENAI_API_KEY HF_TOKEN ARK_API_KEY ARK_API_KEY_BACKUP
mkdir -p "$ASPIRE_OUTPUT_DIR"
exec /work/scripts/python_exec.sh /work/aspire_validate/integration/robodojo_entry.py \
  --task_name general_pickup --env_cfg_type arx_x5 --num_envs 1 --enable_cameras \
  --device_id 0 --policy_name ASPIRE_Pilot --port 0 --protocol ws \
  --additional_info aspire_retime --seed 0 --host 127.0.0.1 --headless \
  --experience /isaac-sim/apps/isaacsim.exp.base.python.kit \
  --kit_args="--/isaaclab/cameras_enabled=true --/crashreporter/enabled=false --/renderer/multiGpu/enabled=false --/renderer/activeGpu=0"
