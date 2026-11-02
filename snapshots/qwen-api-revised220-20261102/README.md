# Revised Qwen API evaluation source — November 2, 2026

This archive captures the source used by the revised 220-state RoboDojo evaluation
after inference switched back to the professor's API, using alias `qwen38-27b`.
It is based on operational source revision `0f84773`. The restored provider's
underlying checkpoint, precision and server configuration have not been verified.

The deployed configuration uses nine continuous episode workers on four L40S
GPUs, shared SAM3 and Contact-GraspNet, and remote model inference. It preserves
the revised official task/return-home instructions, failure-only `give_up`, full
public Qwen history and native robot action budgets. No client output-token,
model-turn or episode-time cap is added. Spending controls are disabled.

The complete deployed inventories contain 582 harness files and 19 bootstrap
files. This public archive contains **554 source/configuration/license files**:
519 are byte-identical to the deployed files and 35 have private site bindings
replaced by placeholders. Seventeen historical experiment manifests, results,
research notes or deployment receipts, plus 30 mesh/media/binary files, are
omitted. Model weights, simulator images, credentials, VPN state, requests,
camera images, trajectories and private evaluation datasets are not included.

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records every included file's original
and published SHA256, redaction categories, and each omitted inventory entry.
The deployed manifest SHA256 is
`27ba8b6cf1bcd57f3953f1b4c5d280d899f74e99766bc9a72848e87281bbfeeb`;
the unchanged 220-state pairing SHA256 is
`8aca18851088e2fc6fa658d6b75ec00391e7227748db2fb1dd16cdda7a600107`.

## Entry points for review

| Source | Responsibility |
|---|---|
| [bootstrap/qwen_driver.py](bootstrap/qwen_driver.py) | API request/stream handling and model response validation |
| [bootstrap/qwen_network.py](bootstrap/qwen_network.py) | Isolated rootless Tailscale transport |
| [harness/qwen_full/full_model_driver.py](harness/qwen_full/full_model_driver.py) | Qwen prompts, sampling and public history |
| [harness/qwen_full/task_prompt_contract.py](harness/qwen_full/task_prompt_contract.py) | Official task instructions and return-home requirements |
| [harness/qwen_full/full_episode.py](harness/qwen_full/full_episode.py) | Native episode lifecycle and action budgets |
| [harness/qwen_integration/aspire_glue.py](harness/qwen_integration/aspire_glue.py) | ASPIRE program execution and robot integration |
| [harness/qwen_integration/perception_bridge.py](harness/qwen_integration/perception_bridge.py) | Shared perception integration |
| [bootstrap/terra_supervisor.py](bootstrap/terra_supervisor.py) | Continuous episode queue and worker lifecycle |
| [bootstrap/audit_qwen_results.py](bootstrap/audit_qwen_results.py) | Native result and provenance audit |

Some reused bootstrap filenames retain `terra`; the current API adapter uses
`qwen38-27b`. Older integration scripts and financial modules remain in the source
inventory for historical provenance; their presence does not make them active
entry points or enable financial admission controls.

## Archive boundaries

This is a source archive for inspection and comparison, **not a directly runnable
resume bundle**. Private hostnames, usernames and storage paths are placeholders.
Original source hashes describe the private deployed bytes; published-file hashes
describe this archive. Missing external assets and historical manifests are not
replaced by fabricated files. The child-reaper C source is included; its compiled
binary is omitted. No code in the deployed evaluation was edited, no job was
started or resumed, and no model call was made to publish this snapshot.

The root package outside this directory is the separately migrated portable
runtime. It has a different execution identity. Neither this archive nor the
portable quickstart authorizes resuming historical or cancelled campaigns.

ASPIRE's [license](harness/ASPIRE/LICENSE) and [notice](harness/ASPIRE/NOTICE) and the
official task recipes' [license](harness/qwen_full/official_task_recipes/LICENSE)
and [provenance](harness/qwen_full/official_task_recipes/PROVENANCE.json) are retained.
See also the repository's [third-party notices](../../docs/third-party.md).

The related [API failure report](../../diagnostics/qwen-api-oom-20261102/README.md)
contains one saved HTTP 500 response and a small request-context record.
