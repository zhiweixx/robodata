# Working in RoboData

## Read first

1. `README.md`: current pipeline, supported scope and CPU quickstart.
2. `docs/installation.md` and `docs/cluster.md`: environment setup and GPU execution.
3. `docs/evaluation.md`: state identity, native scoring, resume and audit rules.
4. `docs/PROFESSOR_RUNBOOK.md`: independent 30-state/all-42-task handoff.
5. `docs/migration.md`: mapping from the older experiment repository.

This is a robot data project. RoboDojo evaluation is one consumer of shared
execution/perception code. Keep benchmark rules in `src/robodata/eval/robodojo`;
do not put general robot, model or data code there merely because it is first used
by this evaluator. Use explicit `robodata.*` imports. Do not resurrect the old
`qwen_full`/`qwen_integration` PYTHONPATH layout.

## Commands and working environment

- Use Python 3.12 and an editable install of both `third_party/aspire` and this repo.
- CPU checks: `python -m pytest -q`; syntax: `python -m compileall -q src`.
- Lightweight entry point: `robodata --help`.
- Set site-specific paths in an ignored `configs/clusters/local*.json` file.
- Real model inference, perception inference and simulator execution belong on
  GPU compute nodes in a scheduler allocation. Do not run them on a developer's
  laptop or a cluster login node. CPU tests/planning/log inspection are fine there.
- `doctor`, `plan`, and `status` do not certify rendering. Validate the intended driver/container/device on a small scheduled run
  before broad submission. See docs/cluster.md for what is counted and which
  standalone preflight checks have not yet been ported.
- Read the actual installed version locks; don't upgrade dependencies to latest
  while diagnosing an experiment. Model/provider/simulator environments differ.

## Preserve experiments

Never edit a running campaign's frozen source, manifest, model settings, prompt,
provider thresholds, candidate queue, or action interface. Make changes in a new
checkout/campaign and preserve the old evidence. This repository migration does
not authorize changing any independently running Lighthouse experiment.

The current protocol pins Qwen revision/sampling, BF16/TP2/MTP-3, 262144 context,
full history, native action budgets and real SAM3/Contact-GraspNet. It contains no
VLA, offline skill discovery or cross-episode learning. Do not add turn limits,
output-token caps, history compaction or episode deadlines as a performance fix.
Any protocol change needs an explicit new experiment identity and user direction.

- Count an episode only after native scoring and evidence commit.
- Distinguish an interruption from a scored failure. Preserve all interrupted
  attempts and raw usage, and resume only native unscored states.
- Never retry a scored failure, relabel a layout ID, duplicate a physical initial
  state, select a newer result opportunistically, or fabricate replacement states.
- Invalid initialization replacement uses only the frozen group's remaining
  unique candidate queue. Exhaustion is a partial result requiring diagnosis.
- Group workers must own disjoint groups and maintain locks/checkpoints. Keep
  simulator/controller/provider processes and node-local relays isolated.
- Keep evaluator-only labels and simulator ground truth out of model observations.
- Audit native scores, state hashes, budgets, attempts, all usage and videos before
  claiming completion. Subset averages are not an official RoboDojo full score.
- Multi-node scheduling/CGN random streams and hardware kernels can change outputs;
  matching model settings and budgets is not a promise of bitwise replay.

## Implementation boundaries

The six reviewed repairs are intentional: inactive-arm commanded-target hold;
CGN empty/singleton indexing; separate native tick stdout archive; accepted-command
gripper cache synchronization after partial failure; strict waypoint-count
receipts; and frozen bootstrap/source verification. Preserve their tests.

ASPIRE's vendored source retains its license and notice. Document modifications
and update the patch/provenance records. Do not silently replace it with upstream
HEAD. Package-facing APIs should not import heavyweight CUDA/simulator libraries
at module import time; keep planning, CLI help, data inspection and tests CPU-only.

Generated Python execution is not a security sandbox. Keep credentials out of its
environment and artifacts, use private node-local relay directories, and bind model
and perception services to loopback. Cross-node RPC requires an explicit secure
transport design; do not expose an unauthenticated service on 0.0.0.0.

## Publishing and documentation

Never commit credentials, local machine configurations, absolute private storage
paths, model checkpoints, simulator assets/images, compiled kernels, videos or raw
trajectories. Store artifact references/checksums instead. Preserve third-party
license conditions; gated downloads use the operator's own credentials and access.

Document only commands that exist. Label CPU-tested behavior separately from
GPU-validated behavior. Don't claim a one-day runtime, improved success, complete
ASPIRE reproduction, dense training trajectories, or official benchmark scores
without corresponding evidence. Add meaningful regression tests for behavioral
changes; do not mass-rewrite frozen experiment outputs to make tests pass.
