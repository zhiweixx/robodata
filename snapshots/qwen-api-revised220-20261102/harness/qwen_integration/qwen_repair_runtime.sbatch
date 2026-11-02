#!/bin/bash
#SBATCH --job-name=qwen-runtime
#SBATCH --account=vvh
#SBATCH --partition=vvh-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:l40s:4
#SBATCH --gpu_cmode=shared
#SBATCH --time=01:00:00
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen/logs/runtime-%j.log
set -euo pipefail
umask 077
source /etc/profile.d/http_proxy.sh
ROOT=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate
export UV_CACHE_DIR="$ROOT/qwen/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/qwen/python"
export PYTHONUNBUFFERED=1
unset HF_TOKEN OPENAI_API_KEY OPENAI_API_KEY_FILE
"$ROOT/qwen/.venv/bin/uv" python install 3.12.13
QWEN_PYTHON=$("$ROOT/qwen/.venv/bin/uv" python find --managed-python 3.12.13)
"$ROOT/qwen/.venv/bin/uv" venv --python "$QWEN_PYTHON" "$ROOT/qwen/.venv-runtime"
"$ROOT/qwen/.venv/bin/uv" pip install --python "$ROOT/qwen/.venv-runtime/bin/python" 'vllm==0.32.0' 'transformers==5.16.1' --torch-backend=cu130
"$ROOT/qwen/.venv/bin/uv" pip freeze --python "$ROOT/qwen/.venv-runtime/bin/python" > "$ROOT/qwen/environment-runtime.freeze.txt"
export PATH="$ROOT/qwen/.venv-runtime/bin:$PATH"
"$ROOT/qwen/.venv-runtime/bin/python" -c 'import sysconfig,pathlib,torch; p=pathlib.Path(sysconfig.get_path("include"))/"Python.h"; assert p.is_file(); print({"python_header":str(p),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu_count":torch.cuda.device_count()}); from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration; print("Qwen architecture import passed")'
