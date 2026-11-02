#!/bin/bash
#SBATCH --job-name=qwen-robodojo-perception6
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
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen/logs/perception6-%j.log
set -euo pipefail
umask 077
QWEN_PERCEPTION_ROOT=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate
: "${QWEN_PERCEPTION_CAMPAIGN:?paired comparison ID required}"
: "${QWEN_PERCEPTION_MODE:=full}"
case "$QWEN_PERCEPTION_CAMPAIGN" in qwen38_perception_*) ;; *) exit 2;; esac
cd "$QWEN_PERCEPTION_ROOT"
export ACCEPT_EULA=Y PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
QWEN_PERCEPTION_BATCH_SHA256=$(sha256sum -- "${BASH_SOURCE[0]}")
export QWEN_PERCEPTION_BATCH_SHA256="${QWEN_PERCEPTION_BATCH_SHA256%% *}"
unset OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN CUDA_HOME TRITON_PTXAS_PATH
QWEN_PERCEPTION_STEP_PID=''
QWEN_PERCEPTION_WAIT_INTERRUPTED=0
forward_allocation_warning() {
  QWEN_PERCEPTION_WAIT_INTERRUPTED=1
  if [[ -n "$QWEN_PERCEPTION_STEP_PID" ]]; then kill -USR1 "$QWEN_PERCEPTION_STEP_PID" 2>/dev/null || true; fi
}
trap forward_allocation_warning USR1
# The initial batch wrapper selects the fixed shared profile. The supervisor
# validates its own path/hash and all bootstrap bytes before starting any model.
QWEN_PERCEPTION_BOOTSTRAP="$QWEN_PERCEPTION_ROOT/perception_profiles/$QWEN_PERCEPTION_CAMPAIGN/perception/qwen_full"
srun --gpu_cmode=shared /usr/bin/python3.12 -B "$QWEN_PERCEPTION_BOOTSTRAP/perception_supervisor.py" \
  --root "$QWEN_PERCEPTION_ROOT" --campaign "$QWEN_PERCEPTION_CAMPAIGN" \
  --manifest "$QWEN_PERCEPTION_ROOT/full_results/$QWEN_PERCEPTION_CAMPAIGN/manifest.json" \
  --mode "$QWEN_PERCEPTION_MODE" &
QWEN_PERCEPTION_STEP_PID=$!
while :; do
  QWEN_PERCEPTION_WAIT_INTERRUPTED=0
  QWEN_PERCEPTION_WAIT_STATUS=0
  wait "$QWEN_PERCEPTION_STEP_PID" || QWEN_PERCEPTION_WAIT_STATUS=$?
  if [[ "$QWEN_PERCEPTION_WAIT_INTERRUPTED" -eq 0 ]]; then break; fi
done
exit "$QWEN_PERCEPTION_WAIT_STATUS"
