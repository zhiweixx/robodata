#!/bin/bash
#SBATCH --job-name=aspire-qwen3
#SBATCH --account=vvh
#SBATCH --partition=vvh-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:l40s:4
#SBATCH --gpu_cmode=shared
#SBATCH --time=03:30:00
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen/logs/eval-%j.log
set -euo pipefail
umask 077
QWEN_PILOT_ROOT=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate
cd "$QWEN_PILOT_ROOT"
export ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1
# The supervisor sanitizes all child environments; no cloud API key is needed.
unset OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN
date -u
hostname
srun --gpu_cmode=shared /usr/bin/python3.12 \
  "$QWEN_PILOT_ROOT/qwen_integration/qwen_server_suite.py" \
  --root "$QWEN_PILOT_ROOT" --run-id "qwen3_${SLURM_JOB_ID}"
