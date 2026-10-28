#!/usr/bin/env bash
# Resource/account/array parameters are supplied by the frozen submit plan.
set -euo pipefail
: "${SLURM_JOB_ID:?must run through Slurm}"
: "${SLURM_ARRAY_TASK_ID:?one array element per manifest assignment}"
# Slurm copies this script; BASH_SOURCE cannot locate the source checkout.
repo=$1
campaign_dir=$2
mode=${3:-full}
export PYTHONPATH="$repo/src:$repo/third_party/aspire/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN CUDA_HOME CUDA_PATH TRITON_PTXAS_PATH
ROBODATA_BATCH_SHA256=$(sha256sum -- "${BASH_SOURCE[0]}")
export ROBODATA_BATCH_SHA256="${ROBODATA_BATCH_SHA256%% *}"
command=("${ROBODATA_PYTHON:-python3}" -B -m robodata.eval.robodojo.runner --campaign-dir "$campaign_dir" --shard-index "$SLURM_ARRAY_TASK_ID" --frozen)
if [[ "$mode" == resume ]]; then command+=(--resume); fi
exec srun "${command[@]}"
