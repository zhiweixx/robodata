# Portable RoboDojo evaluation

The executable reference run is a custom **767 new scored episodes**, across **40 task families and 51 configuration groups**, using seed 0. It preserves the reviewed subset's native layout IDs, state content hashes, candidate order and action budgets. Generalization pairs contribute 10 standard plus 10 random states. `cover_blocks` has 6 distinct official layouts and `play_tic_tac_toe` has 1. `general_pickup` and `stack_bowls` are excluded. Historical 40 scores are not merged; this is not an official full RoboDojo Avg.

The harness uses ASPIRE's public API and Python executor, the custom ARX X5/native CuRobo bridge, SAM3 and Contact-GraspNet. There is no VLA, offline skill discovery or cross-episode learning. Qwen3.8-27B remains the pinned revision with BF16, TP2, MTP3, native 262144 context and unchanged sampling. There are no independent model-turn, output-token or episode wall-time caps. The 8192 server setting limits a chunk of prefill work, not completion length.

## Hardware boundary

**H100/H200-only nodes cannot run this reference Isaac Sim RTX rendering pipeline.** They lack RT cores. The implemented worker uses four co-located Slurm GPUs: slot 0 renders/simulates, slots 1/2 serve Qwen, and slot 3 runs SAM3/CGN. Use an RTX-compatible node such as four L40S with a supported driver. Models/providers use loopback endpoints and the model relay uses node-local files. Separating Hopper inference from remote RTX render nodes requires a new authenticated transport/topology and validation; it is not implemented here.

Independent workers scale across nodes. One worker exclusively owns a whole native configuration group and its replacement queue. 51 groups can run concurrently if 51 suitable allocations are available; fewer shards assign several entire groups sequentially. Each group starts fresh local model/provider services. No Tensor Parallel collective crosses nodes in this reference topology.

## Freeze and stage the run

Complete [installation](installation.md) and configure site paths/resources first. Then freeze a fresh campaign from this checkout:

```bash
robodata plan \
  --manifest experiments/manifests/robodojo_subset767.json \
  --config /shared/robodata/local.cluster.json \
  --output /shared/robodata/runs/my_eval \
  --shards 51
```

Planning performs no GPU work. It copies executable source into `my_eval/source`, validates the semantic manifest and saves resolved machine configuration, source hashes and disjoint group assignments. The original checkout may change later; each worker executes the campaign's frozen source. Do not edit campaign snapshots. A new source/configuration requires a new campaign.

Before launching the array, validate the new site in a Slurm allocation with the configured four compatible GPUs. Inside that allocation, run **one real planned group**, for example:

```bash
srun --ntasks=1 robodata run --campaign-dir /shared/robodata/runs/my_eval --group-id align_blocks_s0
robodata audit --campaign-dir /shared/robodata/runs/my_eval --output /shared/robodata/runs/my_eval/staging_audit.json
```

This group is actual evaluation and its 20 native scores count toward 767. A worker first checks model bytes and an unscored three-image model-interface request, then loads actual SAM3/CGN, verifies their checkpoints and the CGN repair, and launches native rendering. Native startup verifies Vulkan/CUDA GPU identity and camera calibration. Inspect the staging group's video/commit/provider evidence before scaling up. The portable package has CPU contract tests; it has **not yet passed a clean-room GPU deployment on the recipient cluster**. A separate unscored end-to-end RGB-D/SAM3/CGN/control preflight command has not been ported.

Use `robodata submit --help` for the Slurm wrapper. Submit the frozen campaign with `--resume` after staging; completed native groups are skipped, not replayed. Worker array elements accept:

```bash
python -m robodata.eval.robodojo.runner \
  --campaign-dir /shared/robodata/runs/my_eval --shard-index 0 --resume
```

The runner requires a Slurm allocation. GPU/model/simulator execution on a laptop or login node is rejected.

## Ownership, checkpoints and invalid layouts

`plan.json` covers each of the 51 groups exactly once. A filesystem lock enforces one live owner per group across nodes. The native task evaluator remains responsible for success, partial scores, videos, invalid initialization and its checkpoint. The manifest contains 2425 unique candidate states from which 767 scored episodes are requested. It is not a request to run 2425 scored episodes.

If an official initialization is invalid, native logic skips it and continues to the next unused manifest candidate. If a finite candidate queue is exhausted, the worker stops with partial evidence; it does not generate new layouts, repeat a state or rerun a scored failure. Resume checks the native scored IDs against the completed/abandoned checkpoint IDs before continuing. Completed scores with missing evidence cause an error requiring inspection, never replay.

Slurm preemption, OOM or infrastructure faults preserve native checkpoints and unscored usage. After diagnosing the fault, use the same campaign with `--resume`. A scored model failure is a benchmark result and is not a reason to retry. Per-attempt logs and relay archives remain separate; services stop before the private node-local relay is archived.

## Evidence and aggregation

```bash
robodata status --campaign-dir /shared/robodata/runs/my_eval
robodata audit --campaign-dir /shared/robodata/runs/my_eval --output /shared/robodata/runs/my_eval/audit.json
```

`status` reads native score snapshots without loading a model or traversing large ledgers. `audit` also verifies the frozen imports, group identities, native action limits, actual perception providers, three nonempty correctly labeled camera videos per score, one committed episode report, model finish receipts and raw usage from every worker attempt. Model-interface smoke usage is reported separately from measured evaluation usage. The summary reports native success rates by family/category; it does not relabel the raw episode-weighted rate as official Avg. Source/config drift, duplicate owners, inconsistent checkpoints or orphan ledgers prevent a verified-complete report.

Artifacts are organized as follows:

```text
my_eval/
  source/                     frozen executed code
  manifest.json               machine-independent task/state selection
  cluster_config.json         resolved site locations and resource choices
  source_inventory.json
  plan.json                   disjoint group ownership
  allocations/<job>/<group>/attempt_NNN/
    worker.json               actual services/GPU identities and timestamps
    providers.json            actual checkpoint/repair/environment evidence
    usage.jsonl               all requests, including unscored interruptions
    relay/                    archived after endpoints stop
  groups/<group>/episodes/<episode>/attempt_NNN/
    public_input_*.json
    program_*.py
    episode_report.json
  native/RoboDojo/...          native results, videos and resume checkpoints
```

Perception RNG is seeded 0 at service startup and then advances per request. Services restart at group/attempt boundaries. Consequently, this portable multi-replica schedule does not promise bit-for-bit identity with the earlier long-lived Lighthouse service or across interruptions. State/model/action-budget identity is preserved and the scheduling policy is recorded. Do not silently implement per-call RNG reseeding to hide this distinction.

## One-day capacity planning

Measure a few complete representative groups on the target RTX topology first. If average episode time is `t` hours and usable evaluation time after setup is `H` hours, `767*t/H` is only a lower bound on concurrent simulation workers. Add headroom for startup, invalid scenes, long-context episodes and whole-group stragglers. Independent replicas avoid shared inference queues, but a single slow group can still determine the finish time. Neither a large H100/H200 pool nor a planned concurrency value guarantees a one-day finish for this rendering pipeline.
