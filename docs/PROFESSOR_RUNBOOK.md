# Run Qwen3.8-27B on 30 RoboDojo states per task

This guide is for a collaborator installing and running an **independent evaluation
on their own cluster**. It does not connect to, stop, or modify the existing
Lighthouse evaluation. Source: <https://github.com/zhiweixx/robodata>.

## Requested experiment

| Item | Setting |
|---|---|
| Benchmark scope | All 42 RoboDojo task families, represented by 54 runnable configurations |
| Initial states | Up to 30 distinct official physical states per family, one scored trial per state |
| Generalization tasks | 15 standard + 15 random states per family, **not 30 per variant** |
| Finite-layout exceptions | `cover_blocks`: 6 distinct layouts; `play_tic_tac_toe`: 1 |
| Total target | **1,207 new scored episodes**: 40 × 30 + 6 + 1 |
| Seed/layout selection | Official seed-0 layout directory; original layout IDs and deterministic unique-state order |
| Model | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Inference | BF16, TP2, MTP-3, native 262,144-token context; pinned sampling |
| Harness | ASPIRE API/Python executor + custom X5/native CuRobo bridge, with all six reviewed repairs |
| Perception | SAM3 and Contact-GraspNet enabled |
| Learning | No offline skill discovery, prompt tuning from test outcomes, or cross-episode learning |
| Budgets | Unchanged native action budgets; no added turn/output-token/episode-time caps |

Seed 0 contains many different initial layouts. This experiment does **not** repeat
one identical state 30 times. The manifest authorizes 2,565 unique candidate states
so native invalid initializations can be skipped while filling the 1,207 targets.
Those extra candidates are replacements, not additional scored trials. If the
unique candidates are exhausted, retain and report the shortfall.

**All 42 includes `general_pickup` and `stack_bowls`.** Excluding those two would
mean 40 families and 1,147 target episodes. The commands below include them and
start a new campaign; they do not merge any historical scores. This is a custom
subset evaluation, not the official full-benchmark average.

## Compute and access needed

The implemented deployment uses **one four-GPU allocation per worker**, with all
four GPUs on the same node:

- GPU slot 0: Isaac Sim rendering and simulation; requires a supported RT-capable GPU.
- GPU slots 1–2: Qwen inference with tensor parallelism 2.
- GPU slot 3: SAM3 and Contact-GraspNet.

Four L40S cards are the reference topology. **H100/H200 have no RT cores and cannot
fill the renderer role in this Isaac Sim pipeline.** This release does not provide
cross-node RPC between H100/H200 inference servers and separate RTX render nodes.
To use the current launcher, provision suitable four-GPU nodes. A different
heterogeneous deployment needs additional implementation and validation.

The cluster needs Slurm, compatible NVIDIA drivers, Apptainer/Singularity with GPU
passthrough, Git, `uv`, and preferably `skopeo`. A Docker daemon is not required.
Use shared installation/output paths visible identically on every worker, plus
private node-local scratch. Obtain the operator's own access to SAM3 weights and
accept applicable NVIDIA/model licenses. No OpenAI API key is used and there are
no paid API calls. Contact-GraspNet retains its non-commercial research/evaluation
license; see [third-party notices](third-party.md).

## 1. Clone and check the CPU package

```bash
git clone https://github.com/zhiweixx/robodata.git
cd robodata
# For an existing clean checkout, use: git pull --ff-only
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e third_party/aspire -e '.[test]'
python -m pytest -q
robodata make-manifest --help
```

The first public release only supported the old 767-episode manifest. Use an
updated checkout containing `make-manifest`; do not edit the old manifest or
remove its integrity checks. A coding agent should read [AGENTS.md](../AGENTS.md)
before making changes.

## 2. Prepare the cluster installation

Use the [installation guide](installation.md) for driver/container details. The
main preparation commands are:

```bash
export ROBODATA_DATA=/shared/YOUR_PROJECT/robodata-runtime
cp configs/clusters/rtx-slurm.example.json configs/clusters/local.json
bash deploy/bootstrap.sh
export ROBODATA_PYTHON="$ROBODATA_DATA/envs/tools/bin/python"
uv pip install --python "$ROBODATA_PYTHON" -e third_party/aspire -e .
export PATH="$ROBODATA_DATA/envs/tools/bin:$PATH"
```

Edit `configs/clusters/local.json` before continuing:

- Replace scheduler account, partition and GPU resource name with the site's values.
- Set `scheduler.max_concurrent_shards` to the number of simultaneous four-GPU workers.
  Use **1 for four GPUs total**; 24 workers require 96 GPUs.
- Set the Slurm walltime within site limits and long enough for the slowest group.
- Verify storage paths and GPU roles. Set `container.accept_eula=true` only after
  the installing operator has accepted the NVIDIA license.

Save a SAM3-authorized HF token in a private user-owned file outside Git, with mode
600. Pass its path; never place the token in a command, document, log, or commit.

```bash
robodata fetch --config configs/clusters/local.json \
  --components sources assets model providers \
  --hf-token-file /PRIVATE/PATH/hf.token
robodata build-container --config configs/clusters/local.json
```

Install the separate model, provider and simulator environments on a compute node:

```bash
salloc --account=YOUR_ACCOUNT --partition=YOUR_RTX_PARTITION \
  --nodes=1 --ntasks=1 --gres=gpu:l40s:4 --cpus-per-task=32 \
  --mem=240G --time=02:00:00
srun robodata setup --config configs/clusters/local.json
srun robodata doctor --config configs/clusters/local.json --gpu
```

Adapt `--gres` and the other resource options to the actual site. Installation and
kernel compilation must not run on a developer laptop/login node. Release this
preparation allocation when finished. `doctor` checks prerequisites; it does not
certify a rendered observation or a successful grasp. The destination installation
still requires the acceptance step below.

## 3. Generate the 30-state manifest and freeze a new campaign

These are CPU-only commands. The generator uses a pinned catalog of official
source hashes, layout identities and task budgets. It does not generate new scenes
or inspect model outcomes.

```bash
export CAMPAIGN_ROOT=/shared/YOUR_PROJECT/robodata-runs
mkdir -p "$CAMPAIGN_ROOT/manifests"

robodata make-manifest --states-per-task 30 \
  --output "$CAMPAIGN_ROOT/manifests/qwen30_all42.json"

robodata plan \
  --manifest "$CAMPAIGN_ROOT/manifests/qwen30_all42.json" \
  --config configs/clusters/local.json \
  --output "$CAMPAIGN_ROOT/qwen30_all42"

robodata submit --campaign-dir "$CAMPAIGN_ROOT/qwen30_all42" --dry-run
```

The generator should report **42 families, 54 groups, 1,207 target episodes and
2,565 candidate states**, with finite exceptions of 6 and 1. With no `--shards`
override, planning creates 54 disjoint group assignments. Array concurrency comes
from the cluster configuration; 54 assignments do not require 54 simultaneous
allocations. Each assignment owns a whole native group and its replacement queue.

The output manifest and campaign directory must be new. Planning copies executable
code into the campaign and freezes configuration, source hashes and group ownership.
Later changes to the development checkout do not modify that campaign.

## 4. Validate one counted group, then submit the remainder

On a fresh cluster, first request a four-GPU compute allocation with enough time
for one whole group. For example, adapt this to the site's limits:

```bash
salloc --account=YOUR_ACCOUNT --partition=YOUR_RTX_PARTITION \
  --nodes=1 --ntasks=1 --gres=gpu:l40s:4 --cpus-per-task=32 \
  --mem=240G --time=24:00:00
srun --ntasks=1 robodata run \
  --campaign-dir "$CAMPAIGN_ROOT/qwen30_all42" --group-id align_blocks_s0
robodata audit --campaign-dir "$CAMPAIGN_ROOT/qwen30_all42" \
  --output "$CAMPAIGN_ROOT/qwen30_all42/acceptance_audit.json"
```

This is a real **30-episode evaluation group**, counted within the 1,207 total.
It is not training or a disposable prompt-tuning run. Review the native scores,
three-camera videos, action counts, actual provider/weight evidence and audit
errors. The overall campaign will still be incomplete. The worker's initial
model-interface smoke is unscored and its usage is recorded separately.

After this group finishes and its allocation has stopped, launch the remaining
work with `--resume` so completed scores are preserved:

```bash
robodata submit --campaign-dir "$CAMPAIGN_ROOT/qwen30_all42" --resume
```

For a runtime already accepted on this cluster, a fresh unstarted campaign may be
submitted without `--resume`. Do not modify the frozen harness after looking at
failures. If infrastructure repair changes the harness/protocol, retain the old
results and create a distinctly labeled new campaign.

## 5. Monitor and return the evidence

```bash
robodata status --campaign-dir "$CAMPAIGN_ROOT/qwen30_all42"
robodata audit --campaign-dir "$CAMPAIGN_ROOT/qwen30_all42" \
  --output "$CAMPAIGN_ROOT/qwen30_all42/final_audit.json"
```

Completion requires `complete: true`, all 1,207 native scores, and no evidence
errors. Keep failed policy attempts as failures; resume only native unscored states
after an infrastructure interruption. Never rerun a scored failure to improve the
success rate. If official candidate states prove invalid and the pool is exhausted,
report a partial campaign rather than replacing states synthetically.

Return the final JSON audit, per-family success rates, elapsed time and worker/GPU
resource time, token totals, exceptions, source/configuration/container/model
inventories, and paths to raw trajectories and native videos. Keep the frozen
campaign and checkpoints. Share large artifacts separately; do not commit them or
credentials to the public Git repository.

## Capacity planning for a one-day run

Install and accept the runtime before using a production-day estimate. For 1,207
episodes, the optimistic episode-work time is `1207 × mean_minutes / (60 × workers)`.
The following are planning examples, not measured performance:

| Mean minutes per episode | Concurrent workers | GPUs | Episode-work estimate |
|---:|---:|---:|---:|
| 10 | 12 | 48 | 16.8 hours |
| 20 | 24 | 96 | 16.8 hours |
| 30 | 36 | 144 | 16.8 hours |

Add queueing, repeated model/provider startup, invalid scenes and whole-group
stragglers. Even with 54 workers, the slowest group remains a lower bound. The
uncapped episode protocol means a one-day finish is not guaranteed. Measure
representative groups before selecting concurrency; do not shorten budgets or
history to meet a deadline.
