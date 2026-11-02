#!/bin/bash
#SBATCH --job-name=qwen-kernels
#SBATCH --account=vvh
#SBATCH --partition=vvh-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:l40s:4
#SBATCH --gpu_cmode=shared
#SBATCH --time=01:00:00
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen/logs/kernels-%j.log
set -euo pipefail
umask 077
source /etc/profile.d/http_proxy.sh
ROOT=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate
export UV_CACHE_DIR="$ROOT/qwen/uv-cache"
export PATH="$ROOT/qwen/.venv-runtime/bin:$ROOT/qwen/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
unset HF_TOKEN OPENAI_API_KEY OPENAI_API_KEY_FILE CUDA_HOME CUDA_PATH TRITON_PTXAS_PATH
"$ROOT/qwen/.venv-runtime/bin/vllm" download-kernels
"$ROOT/qwen/.venv/bin/uv" pip freeze --python "$ROOT/qwen/.venv-runtime/bin/python" > "$ROOT/qwen/environment-runtime.freeze.txt"
"$ROOT/qwen/.venv-runtime/bin/python" -c 'import torch,flashinfer; x=torch.zeros((1,248320),device="cuda"); y=flashinfer.sampling.top_k_top_p_sampling_from_logits(x,top_k=20,top_p=0.95); torch.cuda.synchronize(); print({"sampling_kernel_passed":True,"output_shape":list(y.shape),"flashinfer":flashinfer.__version__})'
