#!/bin/bash
#SBATCH --job-name=qwen-bf16-mtp-trial
#SBATCH --account=vvh
#SBATCH --partition=vvh-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:l40s:4
#SBATCH --gpu_cmode=shared
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen/logs/mtp-trial-%j.log
set -euo pipefail
umask 077
QWEN_MTP_ROOT=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate
: "${QWEN_MTP_TRIAL:?trial ID required}"
: "${QWEN_FULL_CAMPAIGN:?campaign ID required}"
cd "$QWEN_MTP_ROOT"
unset OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN CUDA_HOME TRITON_PTXAS_PATH
export PYTHONUNBUFFERED=1
QWEN_MTP_STEP_PID=''
QWEN_MTP_WAIT_INTERRUPTED=0
forward_mtp_warning() {
  QWEN_MTP_WAIT_INTERRUPTED=1
  if [[ -n "$QWEN_MTP_STEP_PID" ]]; then
    kill -USR1 "$QWEN_MTP_STEP_PID" 2>/dev/null || true
  fi
}
trap forward_mtp_warning USR1
srun --gpu_cmode=shared /usr/bin/python3.12 qwen_full/mtp_trial_supervisor.py \
  --root "$QWEN_MTP_ROOT" --campaign "$QWEN_FULL_CAMPAIGN" \
  --trial "$QWEN_MTP_TRIAL" --cases "$QWEN_MTP_ROOT/mtp_trials/$QWEN_MTP_TRIAL/cases" &
QWEN_MTP_STEP_PID=$!
while :; do
  QWEN_MTP_WAIT_INTERRUPTED=0
  QWEN_MTP_WAIT_STATUS=0
  wait "$QWEN_MTP_STEP_PID" || QWEN_MTP_WAIT_STATUS=$?
  if [[ "$QWEN_MTP_WAIT_INTERRUPTED" -eq 0 ]]; then
    break
  fi
done
exit "$QWEN_MTP_WAIT_STATUS"
