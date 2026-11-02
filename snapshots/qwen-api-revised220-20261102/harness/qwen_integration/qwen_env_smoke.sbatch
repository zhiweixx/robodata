#!/bin/bash
#SBATCH --job-name=qwen-env-smoke
#SBATCH --account=vvh
#SBATCH --partition=vvh-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:l40s:4
#SBATCH --gpu_cmode=shared
#SBATCH --time=00:08:00
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen/logs/env-smoke-%j.log
set -euo pipefail
umask 077
export ACCEPT_EULA=Y
unset OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN
srun --gpu_cmode=shared /usr/bin/python3.12 /srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen_integration/qwen_env_smoke.py
