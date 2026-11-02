#!/bin/bash
#SBATCH --job-name=qwen-prepare
#SBATCH --account=vvh
#SBATCH --partition=vvh-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:l40s:4
#SBATCH --gpu_cmode=shared
#SBATCH --time=02:00:00
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen/logs/prepare-%j.log
set -euo pipefail
umask 077
source /etc/profile.d/http_proxy.sh
ROOT=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate
export UV_CACHE_DIR="$ROOT/qwen/uv-cache"
export HF_HOME="$ROOT/qwen/hf-home"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
unset HF_TOKEN OPENAI_API_KEY OPENAI_API_KEY_FILE
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv > "$ROOT/qwen/gpus-prepare-$SLURM_JOB_ID.csv"
getconf GNU_LIBC_VERSION
if [ ! -x "$ROOT/qwen/.venv/bin/python" ]; then
  /home/ROBODATA_USER/.local/bin/uv venv --python /usr/bin/python3.12 "$ROOT/qwen/.venv"
fi
/home/ROBODATA_USER/.local/bin/uv pip install --python "$ROOT/qwen/.venv/bin/python" uv
"$ROOT/qwen/.venv/bin/uv" pip install --python "$ROOT/qwen/.venv/bin/python" 'vllm==0.32.0' 'transformers==5.16.1' --torch-backend=cu130
"$ROOT/qwen/.venv/bin/uv" pip freeze --python "$ROOT/qwen/.venv/bin/python" > "$ROOT/qwen/environment.freeze.txt"
"$ROOT/qwen/.venv/bin/python" "$ROOT/qwen_integration/qwen_prepare.py"
"$ROOT/qwen/.venv/bin/python" -c 'import torch, transformers, vllm; print({"torch":torch.__version__,"cuda":torch.version.cuda,"vllm":vllm.__version__,"transformers":transformers.__version__,"gpu_count":torch.cuda.device_count()}); x=torch.ones(8,device="cuda"); print({"cuda_sum":x.sum().item()})'
