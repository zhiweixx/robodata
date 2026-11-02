#!/bin/bash
#SBATCH --job-name=qwen-robodojo-paired12
#SBATCH --account=vvh
#SBATCH --partition=vvh-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:l40s:4
#SBATCH --gpu_cmode=shared
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen/logs/paired12-%j.log
set -euo pipefail
umask 077
QWEN_PAIRED_ROOT=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate
: "${QWEN_PAIRED_CAMPAIGN:?paired comparison ID required}"
: "${QWEN_PAIRED_MODE:=full}"
case "$QWEN_PAIRED_CAMPAIGN" in qwen38_paired_*) ;; *) exit 2;; esac
cd "$QWEN_PAIRED_ROOT"
export ACCEPT_EULA=Y PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN CUDA_HOME TRITON_PTXAS_PATH
QWEN_PAIRED_STEP_PID=''
QWEN_PAIRED_WAIT_INTERRUPTED=0
forward_allocation_warning() {
  QWEN_PAIRED_WAIT_INTERRUPTED=1
  if [[ -n "$QWEN_PAIRED_STEP_PID" ]]; then kill -USR1 "$QWEN_PAIRED_STEP_PID" 2>/dev/null || true; fi
}
trap forward_allocation_warning USR1
srun --gpu_cmode=shared /usr/bin/python3.12 -B "$QWEN_PAIRED_ROOT/qwen_full/paired_supervisor.py" \
  --root "$QWEN_PAIRED_ROOT" --campaign "$QWEN_PAIRED_CAMPAIGN" \
  --manifest "$QWEN_PAIRED_ROOT/full_results/$QWEN_PAIRED_CAMPAIGN/manifest.json" \
  --mode "$QWEN_PAIRED_MODE" &
QWEN_PAIRED_STEP_PID=$!
while :; do
  QWEN_PAIRED_WAIT_INTERRUPTED=0
  QWEN_PAIRED_WAIT_STATUS=0
  wait "$QWEN_PAIRED_STEP_PID" || QWEN_PAIRED_WAIT_STATUS=$?
  if [[ "$QWEN_PAIRED_WAIT_INTERRUPTED" -eq 0 ]]; then break; fi
done
exit "$QWEN_PAIRED_WAIT_STATUS"
