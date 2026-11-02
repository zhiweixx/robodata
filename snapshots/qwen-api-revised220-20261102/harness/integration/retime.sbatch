#!/bin/bash
#SBATCH --account=vvh
#SBATCH --partition=vvh-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:4
#SBATCH --gpu_cmode=shared
#SBATCH --time=00:20:00
#SBATCH --job-name=aspire-retime
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/logs/retime-%j.log
set -euo pipefail
cd /srv/ROBODATA_RUNTIME/isaac6
export ACCEPT_EULA=Y
date -u
hostname
printenv SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES
srun --gpu_cmode=shared timeout --signal=TERM --kill-after=30 900 \
  bash scripts/container_exec.sh /bin/bash /work/aspire_validate/integration/retime_inside.sh "$SLURM_JOB_ID"
/usr/bin/python3.12 - "$SLURM_JOB_ID" <<'PY'
import json, pathlib, sys
p = pathlib.Path('aspire_validate/results') / ('aspire_retime_' + sys.argv[1]) / 'retime_report.json'
r = json.loads(p.read_text())
assert r.get('ok') is True, r
print('VERIFIED_ASPIRE_ROBODOJO_RETIMING', str(p), flush=True)
PY
