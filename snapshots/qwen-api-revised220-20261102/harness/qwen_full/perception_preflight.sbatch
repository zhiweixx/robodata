#!/bin/bash
#SBATCH --job-name=qwen-perception-smoke
#SBATCH --account=vvh
#SBATCH --partition=vvh-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:l40s:4
#SBATCH --gpu_cmode=shared
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen/logs/perception-smoke-%j.log
set -euo pipefail
umask 077
cd /srv/ROBODATA_RUNTIME/isaac6/aspire_validate
export ACCEPT_EULA=Y PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN CUDA_HOME TRITON_PTXAS_PATH
: "${QWEN_PERCEPTION_PREFLIGHT_PROFILE:?explicit frozen smoke profile required}"
exec srun --gpu_cmode=shared /usr/bin/python3.12 -B "$QWEN_PERCEPTION_PREFLIGHT_PROFILE/qwen_full/perception_preflight.py"
