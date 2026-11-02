#!/bin/bash
#SBATCH --job-name=qwen-robodojo-full
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
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen/logs/full-%j.log
set -euo pipefail
umask 077
QWEN_FULL_ROOT=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate
: "${QWEN_FULL_CAMPAIGN:?campaign ID required}"
: "${QWEN_FULL_MODE:=full}"
: "${QWEN_FULL_MTP_TOKENS:=0}"
case "$QWEN_FULL_MTP_TOKENS" in
  0|1|3) ;;
  *) echo 'QWEN_FULL_MTP_TOKENS must be 0, 1, or 3' >&2; exit 2 ;;
esac
cd "$QWEN_FULL_ROOT"
export ACCEPT_EULA=Y PYTHONUNBUFFERED=1
unset OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN CUDA_HOME TRITON_PTXAS_PATH
# Signal the srun job step before Slurm's deadline so native checkpoints survive.
QWEN_STEP_PID=''
QWEN_WAIT_INTERRUPTED=0
forward_allocation_warning() {
  QWEN_WAIT_INTERRUPTED=1
  if [[ -n "$QWEN_STEP_PID" ]]; then
    kill -USR1 "$QWEN_STEP_PID" 2>/dev/null || true
  fi
}
trap forward_allocation_warning USR1
srun --gpu_cmode=shared /usr/bin/python3.12 "$QWEN_FULL_ROOT/qwen_full/full_supervisor.py" \
  --root "$QWEN_FULL_ROOT" --campaign "$QWEN_FULL_CAMPAIGN" \
  --manifest "$QWEN_FULL_ROOT/qwen_full/full_manifest.json" --mode "$QWEN_FULL_MODE" \
  --mtp-tokens "$QWEN_FULL_MTP_TOKENS" &
QWEN_STEP_PID=$!
# A trapped signal interrupts Bash's wait even while srun is still alive.
# Reap the step after forwarding the warning; do not let set -e end the batch
# allocation before the supervisor has checkpointed and cleaned up children.
while :; do
  QWEN_WAIT_INTERRUPTED=0
  QWEN_WAIT_STATUS=0
  wait "$QWEN_STEP_PID" || QWEN_WAIT_STATUS=$?
  if [[ "$QWEN_WAIT_INTERRUPTED" -eq 0 ]]; then
    break
  fi
done
exit "$QWEN_WAIT_STATUS"
