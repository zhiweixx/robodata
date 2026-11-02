#!/bin/bash
#SBATCH --job-name=qwen-robodojo-repaired
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
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen/logs/repaired-%j.log
set -euo pipefail
umask 077
QWEN_REPAIRED_ROOT=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate
: "${QWEN_REPAIRED_CAMPAIGN:?repaired campaign ID required}"
: "${QWEN_REPAIRED_MODE:=full}"
case "$QWEN_REPAIRED_CAMPAIGN" in qwen38_subset20_repaired_*) ;; *) exit 2;; esac
case "$QWEN_REPAIRED_MODE" in full|resume) ;; *) exit 2;; esac
cd "$QWEN_REPAIRED_ROOT"
export ACCEPT_EULA=Y PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
QWEN_REPAIRED_BATCH_SHA256=$(sha256sum -- "${BASH_SOURCE[0]}")
export QWEN_REPAIRED_BATCH_SHA256="${QWEN_REPAIRED_BATCH_SHA256%% *}"
unset OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN CUDA_HOME TRITON_PTXAS_PATH
QWEN_REPAIRED_STEP_PID=''
QWEN_REPAIRED_WAIT_INTERRUPTED=0
forward_allocation_warning() {
  QWEN_REPAIRED_WAIT_INTERRUPTED=1
  if [[ -n "$QWEN_REPAIRED_STEP_PID" ]]; then kill -USR1 "$QWEN_REPAIRED_STEP_PID" 2>/dev/null || true; fi
}
trap forward_allocation_warning USR1
QWEN_REPAIRED_BOOTSTRAP="$QWEN_REPAIRED_ROOT/harness_profiles/$QWEN_REPAIRED_CAMPAIGN/qwen_full"
srun --gpu_cmode=shared /usr/bin/python3.12 -B "$QWEN_REPAIRED_BOOTSTRAP/repaired_subset_supervisor.py" \
  --root "$QWEN_REPAIRED_ROOT" --campaign "$QWEN_REPAIRED_CAMPAIGN" --mode "$QWEN_REPAIRED_MODE" &
QWEN_REPAIRED_STEP_PID=$!
while :; do
  QWEN_REPAIRED_WAIT_INTERRUPTED=0
  QWEN_REPAIRED_WAIT_STATUS=0
  wait "$QWEN_REPAIRED_STEP_PID" || QWEN_REPAIRED_WAIT_STATUS=$?
  if [[ "$QWEN_REPAIRED_WAIT_INTERRUPTED" -eq 0 ]]; then break; fi
done
exit "$QWEN_REPAIRED_WAIT_STATUS"
