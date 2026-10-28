# Migration from the experiment repository

This is a new public repository with a fresh history. It was extracted from source
revision `0730f9da3f6bdce7b79afe444df96a525f78f464`; the existing repaired767 run
continues using its original frozen `871bbc0` source independently. No remote
runtime directory, checkpoint, job or existing private Git history was changed.

| Old location | Current location |
|---|---|
| `qwen_integration/robodojo_adapter.py`, `native_retime.py`, `qwen_full/full_adapter.py` | `src/robodata/environments/robodojo/` |
| `aspire_glue.py`, `receipt_executor.py`, `native_stdout.py` | `src/robodata/execution/` |
| Public serialization from `model_episode.py` | `src/robodata/execution/model_episode.py` |
| Full model driver/relay and shared request helpers | `src/robodata/agents/` |
| `perception_bridge.py`, `perception_service.py`, `cgn_singleton_fix.py` | `src/robodata/perception/` |
| Native episode/lifecycle, task manifests and audits | `src/robodata/eval/robodojo/` |
| Lighthouse-specific scripts and paths | Portable `deploy/` plus `configs/clusters/` |
| Modified `ASPIRE` runtime | `third_party/aspire/` plus patch provenance |

Historical three-task GPT pilots, superseded capped Qwen launchers, one-off timing
experiments, private paths and raw results are not active public entry points.
They remain in the original repository for provenance. Runtime helpers still used
by the current policy were extracted before removing those pilot entry points.
The public observation serializer is retained without the old pilot's turn/output
caps or truncated feedback loop.

The original subset manifest's task names, native layout IDs, content hashes,
candidate order and budgets are preserved. Installation paths are portable path
bindings. Moving Python modules necessarily changes source hashes: public
campaigns receive new manifests/source inventories and never reuse old source
hashes as if their bytes were unchanged.

All six fixes are retained. The new group-level deployment adds execution
orchestration and explicit imports; this is a new deployment, not an assertion of
bitwise equivalence to the old single-worker order. CGN seeds its process RNG on
startup; allocating groups to fresh provider processes changes random streams.
Qwen batching and GPU kernels can also affect generated outputs.

Future data collection, filtering and training modules should consume shared
execution records. Add those modules when implemented rather than filling this
repository with empty subsystems. Native scoring remains evaluator-owned.
