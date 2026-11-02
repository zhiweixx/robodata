#!/bin/bash
#SBATCH --job-name=qwen-robodojo-subset20
#SBATCH --account=vvh
#SBATCH --partition=vvh-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:l40s:4
#SBATCH --gpu_cmode=shared
#SBATCH --time=14-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen/logs/subset20-%j.log
set -euo pipefail
umask 077
QWEN_SUBSET_ROOT=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate
: "${QWEN_SUBSET_CAMPAIGN:?subset campaign ID required}"
: "${QWEN_SUBSET_MODE:=full}"
: "${QWEN_SUBSET_MTP_TOKENS:=3}"
case "$QWEN_SUBSET_CAMPAIGN" in qwen38_subset20_*) ;; *) echo 'invalid subset campaign' >&2; exit 2;; esac
case "$QWEN_SUBSET_MTP_TOKENS" in 0|1|3) ;; *) echo 'MTP must be0,1,or3' >&2; exit 2;; esac
cd "$QWEN_SUBSET_ROOT"
export ACCEPT_EULA=Y PYTHONUNBUFFERED=1
unset OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN CUDA_HOME TRITON_PTXAS_PATH
QWEN_SUBSET_STEP_PID=''
QWEN_SUBSET_WAIT_INTERRUPTED=0
forward_allocation_warning() {
  QWEN_SUBSET_WAIT_INTERRUPTED=1
  if [[ -n "$QWEN_SUBSET_STEP_PID" ]]; then
    kill -USR1 "$QWEN_SUBSET_STEP_PID" 2>/dev/null || true
  fi
}
trap forward_allocation_warning USR1
srun --gpu_cmode=shared /usr/bin/python3.12 "$QWEN_SUBSET_ROOT/qwen_full/subset_supervisor.py" \
  --root "$QWEN_SUBSET_ROOT" --campaign "$QWEN_SUBSET_CAMPAIGN" \
  --manifest "$QWEN_SUBSET_ROOT/qwen_full/subset_manifest.json" --mode "$QWEN_SUBSET_MODE" \
  --mtp-tokens "$QWEN_SUBSET_MTP_TOKENS" &
QWEN_SUBSET_STEP_PID=$!
while :; do
  QWEN_SUBSET_WAIT_INTERRUPTED=0
  QWEN_SUBSET_WAIT_STATUS=0
  wait "$QWEN_SUBSET_STEP_PID" || QWEN_SUBSET_WAIT_STATUS=$?
  if [[ "$QWEN_SUBSET_WAIT_INTERRUPTED" -eq 0 ]]; then
    break
  fi
done
exit "$QWEN_SUBSET_WAIT_STATUS"
