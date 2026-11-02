#!/bin/bash
#SBATCH --job-name=qwen-api-revised220
#SBATCH --account=vvh
#SBATCH --partition=vvh-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:l40s:4
#SBATCH --gpu_cmode=shared
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen/logs/qwen-revised220-%j.log
set -euo pipefail
umask 077
ulimit -c 0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset VLLM_API_KEY OPENAI_API_KEY OPENAI_API_KEY_FILE HF_TOKEN HUGGING_FACE_HUB_TOKEN TS_AUTHKEY TS_AUTH_KEY TAILSCALE_AUTHKEY CUDA_HOME TRITON_PTXAS_PATH
: "${TERRA_CAMPAIGN:?freeze an isolated paired campaign before submission}"
[[ "$TERRA_CAMPAIGN" =~ ^qwen_paired_[A-Za-z0-9_]+$ ]] || exit 2
TERRA_MODE="${TERRA_MODE:-full}"
case "$TERRA_MODE" in full|resume) ;; *) exit 2;; esac
terra_root=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate
cd "$terra_root"
# Slurm runs a spool copy of this script, so BASH_SOURCE is not the deployed directory.
terra_code_relative=$(/usr/bin/python3.12 -B - "$terra_root/full_results/$TERRA_CAMPAIGN/manifest.json" <<'PYCODE'
import json, re, sys
with open(sys.argv[1]) as stream:
    manifest=json.load(stream)
value=manifest.get('bootstrap_relative_root','terra_robodojo')
pattern=(r'qwen_api_revised220_[A-Za-z0-9_]+' if manifest.get('protocol')=='qwen_paired220_task_reasoning_v2'
         else r'terra_robodojo')
if not isinstance(value,str) or re.fullmatch(pattern,value) is None:
    raise ValueError('unsafe or historical bootstrap path')
print(value)
PYCODE
)
terra_code="$terra_root/$terra_code_relative"
# Supervisor owns a private rootless Tailscale socket; no local model server or nested srun.
# Supervisor verifies exact baseline audit, frozen sources and nonfinancial startup gates.
exec /usr/bin/python3.12 -B "$terra_code/terra_supervisor.py" \
  --root "$terra_root" --campaign "$TERRA_CAMPAIGN" \
  --manifest "$terra_root/full_results/$TERRA_CAMPAIGN/manifest.json" \
  --profile "$terra_root/harness_profiles/$TERRA_CAMPAIGN" --mode "$TERRA_MODE"
