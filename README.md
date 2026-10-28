# RoboData

Infrastructure for robot data collection and evaluation. The current pipeline runs
**Qwen3.8-27B code policies on RoboDojo**, using the ASPIRE API/program executor,
a custom ARX X5/native CuRobo bridge, SAM3 segmentation, and Contact-GraspNet grasp
proposals. There is no VLA, offline skill discovery, or cross-episode learning.
This integration is not a complete reproduction of ASPIRE.

The repository separates the reusable agent, execution, environment and perception
code from benchmark rules. Model weights, simulator assets, videos, credentials
and raw trajectories live outside Git.

For the collaborator’s **30 states per task / all 42 tasks** run, use the
[professor runbook](docs/PROFESSOR_RUNBOOK.md). It creates a separate 1,207-episode
campaign with the same model, perception harness and native action budgets.

## Quickstart

Use Python **3.12**. The following commands install only the CPU package and tests;
they do not start a model, simulator, or paid API call.

```bash
git clone https://github.com/zhiweixx/robodata.git
cd robodata
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e third_party/aspire -e '.[test]'
python -m pytest -q
robodata --help
```

For GPU evaluation, start with [installation](docs/installation.md), then
[cluster execution](docs/cluster.md). Copy the example cluster configuration to an
ignored `configs/clusters/local.json`, set installation paths and scheduler
resources, and use `robodata doctor --help` to check the machine. Preparation has
separate environments for the simulator, Qwen, SAM3 and Contact-GraspNet. The
installer requires the collaborator's own access to gated artifacts and acceptance
of applicable simulator/model licenses; no credentials are shipped here.

The initial deployment uses a **four-GPU, single-node stack per worker**: two GPUs
for BF16/TP2/MTP-3 Qwen, one RTX-capable GPU for Isaac Sim, and one GPU for SAM3/CGN.
Scale by launching independent workers across nodes. H100/H200 are useful for
inference but do not supply the RT cores required by this Isaac Sim rendering path.
See the deployment guide for hardware and mixed-node limitations.

After preparing and checking the runtime, create a new campaign:

```bash
robodata plan \
  --manifest experiments/manifests/robodojo_subset767.json \
  --config configs/clusters/local.json \
  --output /shared/robodata-runs/my_eval \
  --shards 51

# Preview the exact Slurm array, then submit it.
robodata submit --campaign-dir /shared/robodata-runs/my_eval --dry-run
robodata submit --campaign-dir /shared/robodata-runs/my_eval

# Read-only progress and final evidence checks:
robodata status --campaign-dir /shared/robodata-runs/my_eval
robodata audit --campaign-dir /shared/robodata-runs/my_eval \
  --output /shared/robodata-runs/my_eval/audit.json
```

Planning copies an execution source snapshot into the new campaign and records its
hashes. Each shard owns whole task/configuration/seed groups. It preserves the
native candidate order, checkpoint and invalid-initialization replacement rules.
Scored failures are never rerun to improve results. Use a new campaign for a new
harness or protocol.

## What is evaluated?

The supplied **custom 767-episode subset** contains 40 task families, 51 task
configurations, and seed 0. Thirty-eight families have 20 target episodes each;
`cover_blocks` has six unique official layouts and `play_tic_tac_toe` has one.
`general_pickup` and `stack_bowls` are excluded. No historical scores are merged.
Invalid initializations may advance only within each frozen unique-state queue;
exhausting a queue is reported rather than silently inventing replacements.

Qwen uses its pinned revision, BF16, tensor parallelism 2, MTP-3, native 262,144-token
context, unchanged sampling and full public history. The native task action budgets
remain in force. There is no added turn limit, independent output-token cap, or
per-episode wall-clock deadline. These are subset results, **not the official
full-benchmark average**. More nodes improve throughput but do not guarantee a
one-day finish; unusually long episodes and uneven task groups remain possible.

See [evaluation contracts](docs/evaluation.md) and [migration notes](docs/migration.md).

## Code map

| Location | Responsibility |
|---|---|
| `src/robodata/data/` | Portable provenance and data contracts |
| `src/robodata/agents/` | Model client, full-history loop communication, relay |
| `src/robodata/execution/` | Python program execution, receipts, public observations |
| `src/robodata/environments/robodojo/` | X5 robot adapter, native planning and control |
| `src/robodata/perception/` | SAM3/CGN service and grasp-frame conversion |
| `src/robodata/eval/robodojo/` | State selection, native lifecycle, budgets, scoring, audits |
| `configs/` | Model/protocol choices and site configuration examples |
| `deploy/` | Pinned dependencies, installers, container and Slurm wrappers |
| `third_party/aspire/` | Modified ASPIRE runtime, original license and notices |
| `patches/` | Reproducible upstream compatibility changes |
| `experiments/` | Small portable manifests and provenance, not runtime artifacts |
| `tests/` | CPU regression and protocol tests |

For a coding agent, begin with [AGENTS.md](AGENTS.md). For the trajectory format and
its current limits, read [data and artifacts](docs/data.md).

## Validation and licenses

This public migration has a distinct execution identity from the existing frozen
Lighthouse run. The existing run is unchanged. CPU validation in this repository
covers the migrated runtime and planning contracts; a clean-room GPU acceptance run of this
new deployment must be performed on the destination cluster before broad
submission. A previous runtime's successful rendering does not certify a new driver,
GPU, container or installer.

Original RoboData code is Apache-2.0. ASPIRE retains its Apache-2.0 license and
NVIDIA notice. External simulators, assets, providers and weights keep their own
licenses. Contact-GraspNet retains its non-commercial research/evaluation terms;
SAM3 requires the operator’s own gated access. See
[third-party notices](docs/third-party.md). No weights or simulator images are
redistributed by this repository.
