#!/bin/bash
#SBATCH --account=vvh
#SBATCH --partition=vvh-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:4
#SBATCH --gpu_cmode=shared
#SBATCH --time=01:00:00
#SBATCH --job-name=aspire-astra-remaining
#SBATCH --output=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/logs/remaining-%j.log
set -euo pipefail
source /etc/profile.d/http_proxy.sh
cd /srv/ROBODATA_RUNTIME/isaac6
export ACCEPT_EULA=Y
export SAFEHARNESS_PROXY="${https_proxy:-${http_proxy:-}}"
export SAFEHARNESS_CA_FILE=/etc/pki/tls/certs/ca-bundle.crt
date -u
hostname
printenv SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES
srun --gpu_cmode=shared /usr/bin/python3.12 aspire_validate/integration/run_suite.py \
  --run-id "astra3_remaining_${SLURM_JOB_ID}" \
  --tasks stack_bowls align_blocks \
  --prior-ledger /srv/ROBODATA_RUNTIME/isaac6/aspire_validate/results/astra3_27940548/usage.jsonl
