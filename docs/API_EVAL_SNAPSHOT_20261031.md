# API evaluation source snapshot — 2026-10-31

The exact Lighthouse Qwen/GPT API code is preserved in the private
[robo-posttrain snapshot](https://github.com/zhiweixx/robo-posttrain/tree/api-eval-pre-revision-20261031/snapshots/api-eval-20261031),
with the annotated tag `api-eval-pre-revision-20261031`. The same tag in this
repository marks RoboData's state before the planned major revision.

| Runtime | Source directory in snapshot | Recorded scope |
|---|---|---|
| Professor endpoint `qwen38-27b` | `code/professor_api_full807/` | Closed, 721 scored states |
| `gpt-6.1-terra`, medium | `code/terra_robodojo/` | Stopped, 220 scored states |
| `gpt-6-astra`, medium | `code/astra_robodojo/` | 220-state target; owner controls stopping and recovery |

The snapshot preserves each deployed bootstrap and the exact campaign manifests,
plus references to the shared 536-file harness at Git commit
`871bbc0d0e510e9b3a20298b3ddaf3ed6d76d2f9`. It includes source verification and
offline restoration helpers, native dependency revisions/patches, provider package
locks, site launchers, and available CPU tests. Runtime source bytes were not
refactored. Raw results and credentials are excluded from Git.

The API runs use four L40S GPUs, nine simulation slots, shared SAM3/Contact-GraspNet,
and remote language-model inference. Qwen schedules whole configuration groups;
the GPT runs use an episode-level queue. Native action budgets and full public
history are retained. OpenAI provider settings differ from Qwen's supported
sampling fields. Current GPT billing is user-managed; archived financial modules
remain only because historical evidence validation imports them.

This repository's `robodata` package and local-Qwen deployment guides remain a
separate portable implementation. The site-specific API drivers have not been
moved into its package namespace. Use the snapshot to review those drivers and
their exact contracts before designing the requested revision.

## Private research data

- [Qwen archive](https://huggingface.co/datasets/robodata-lab/robodojo-qwen38-27b-20261029),
  revision `9fd3029bb3e396335d9c794d5d5f94af04e1eb60`.
- [Terra archive](https://huggingface.co/datasets/robodata-lab/robodojo-gpt61-terra-20261031),
  revision `b83b164666df22f9755edbaf8f1500149f57fa03`.

The archives retain videos, observations, programs, all attempts, native scoring,
raw API responses, and known/unknown token usage. Their dataset cards describe
hash-verified extraction. Compare Terra with Qwen on the same 220 physical states.
The original 807-state plan remains incomplete; unscored states are not failures.

Snapshot creation did not submit, resume, cancel, or modify any evaluation job.
Astra cleanup/recovery is explicitly left to the owner. Restoration writes only
to a new offline directory and is not an operational resume procedure.
