# Execute the subset across nodes

Each worker owns whole native task/configuration/seed groups. This avoids sharing
mutable checkpoints or accidentally repeating candidate layouts after official
invalid-initialization skips. The supplied custom subset has 51 groups, 40 task
families and 767 target scored episodes. No old campaign or historical scores are
merged into this plan. Read [the evaluation contract](evaluation.md) before running.

## Freeze a new campaign

Install the runtime using [installation.md](installation.md), then:

```bash
robodata plan \
  --manifest experiments/manifests/robodojo_subset767.json \
  --config configs/clusters/local.json \
  --output /shared/robodata-runs/my_eval \
  --shards 51
robodata submit --campaign-dir /shared/robodata-runs/my_eval --dry-run
```

The output directory must be new. Planning expands environment variables and
copies the execution sources into `my_eval/source/`; it records source, config,
manifest and assignment digests. Each group appears exactly once. A 51-shard plan
assigns one group per array element. Fewer shards assign multiple whole groups to
each worker sequentially. Do not split a group's candidate queue across workers.

Before the first broad submission on a new cluster, inspect the manifest's first
group ID and run only that planned group in a four-GPU Slurm allocation:

```bash
srun --ntasks=1 robodata run --campaign-dir /shared/robodata-runs/my_eval \
  --group-id align_blocks_s0
robodata audit --campaign-dir /shared/robodata-runs/my_eval \
  --output /shared/robodata-runs/my_eval/initial_audit.json
```

The command refuses execution outside Slurm. An incomplete campaign audit is
expected at this point; inspect its evidence errors separately from unfinished
groups. Follow the acceptance checks in the installation guide. Once the initial
group is finished and no worker is still active, submit the remaining work with
`--resume` so it preserves that group's native checkpoint:

```bash
robodata submit --campaign-dir /shared/robodata-runs/my_eval --resume
```

For a previously accepted runtime with an entirely new, unstarted campaign,
submit without `--resume`. Both forms submit the **frozen campaign's** batch script,
not the current working copy. The worker hashes Slurm's actual copied script and
compares it against the frozen inventory. Later edits to your development checkout
do not change this campaign.

## Resource and topology choices

`scheduler.max_concurrent_shards` sets the array concurrency. For example, eight
simultaneous elements require eight four-GPU allocations, or 32 GPUs total:
16 for Qwen, eight for rendering and eight for SAM3/CGN. Slurm may place those
allocations on different nodes. Each element requests one node and four GPUs;
all services communicate through loopback and a private node-local relay. GPU
slots refer to the devices allocated to that job; the runner resolves their UUIDs.
Do not hardcode physical host device IDs in the configuration.

H100/H200 can accelerate inference, but the Isaac renderer used here needs an
RTX-capable GPU. This release does not implement cross-node RPC, network service
authentication or queueing between Hopper model servers and separate RTX rendering
nodes. Having thousands of H100s alone therefore does not make this exact stack
runnable. Secure enough compatible RTX allocations, or implement and separately
validate that heterogeneous deployment before scheduling it as the same experiment.

There is no NCCL communication between array elements; TP2 is confined to each
node. Each group starts its own model and perception services. This favors failure
isolation and simple checkpoint ownership over maximum model-server utilization.
It also introduces startup overhead. Perception-service RNG is initialized at
service startup and then advances with requests; changing group order or restart
boundaries can change sampled proposals. The runner records this policy. Do not
claim bit-for-bit equivalence to a serial historical campaign.

## Estimate the one-day target

Measure **end-to-end** episode times across several task families on the target
cluster, including long contexts, provider calls and simulator execution. With
`P` concurrent workers and mean episode time `t` minutes, the optimistic processing
term is `767 * t / (60 * P)` hours. Add installation/model startup, queue delay,
uneven groups and long episodes. At eight workers, a measured 10-minute mean gives
about 16 hours of episode work; a 20-minute mean gives about 32 hours. These are
capacity examples, not measurements or promised finish times.

The slowest whole group is another lower bound even with 51 workers. There is no
per-episode timeout, artificial turn limit, independent output-token cap, or
history compaction. Preserving that protocol means a one-day deadline cannot be
guaranteed. Increase compatible worker concurrency after measuring representative
runtime; do not silently reduce budgets to meet the deadline. Set a Slurm walltime
large enough for startup plus the slowest group, within your site's limit.

## Monitor, audit, resume

```bash
robodata status --campaign-dir /shared/robodata-runs/my_eval
robodata audit --campaign-dir /shared/robodata-runs/my_eval \
  --output /shared/robodata-runs/my_eval/audit.json
```

`status` reads native progress. A final successful audit requires all expected
native scores, state identities, action budgets, three-camera videos, committed
reports, provider evidence and raw usage. This custom subset's score is not the
official full RoboDojo average.

After an infrastructure interruption, preserve the allocation's logs, unscored
usage and checkpoints. Confirm no array element from the campaign remains active
(`submit` also checks the campaign job name). Diagnose the failure, then use
`robodata submit --campaign-dir ... --resume` with the unchanged frozen plan.
Committed failures remain failures; a resume advances from native checkpoint state.
Exhausted unique-state queues are reported rather than replaced with new layouts.
A protocol, prompt, model or harness change belongs in a different campaign.

The process may need first-use CUDA compilation again on a new node. Do not copy
credentials or mutable live relays onto shared storage. Endpoints are stopped
before their relay data are archived. Collect the final JSON audit, frozen source,
configuration, inventories and scheduler records alongside the native outputs for
reproduction. Share reports and code deliberately; do not add raw videos, weights,
containers or credentials to Git.
