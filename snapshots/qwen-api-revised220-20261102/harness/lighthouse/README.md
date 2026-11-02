# Installed native runtime snapshot

These patches and launcher scripts were read from the existing Lighthouse
installation on 2026-10-27, after stopping subset job 27981498 and before the
paired harness comparison. They are the **shared runtime for both arms**, not
part of the harness treatment. `snapshot.json` records file SHA256 values;
`*-revision.txt` records the corresponding upstream Git revisions.

The patches capture tracked changes in RoboDojo, XPolicyLab, and IsaacLab. They
include earlier compatibility and evaluation-lifecycle work. Apply each patch
only to its matching upstream revision and review installation paths before
running the scripts. The small untracked Python package initializer is retained
under `untracked/`. Downloaded simulator binaries, assets, Python environments,
model weights, and the alternate IsaacLab download are external dependencies.

These files are an archival snapshot. Creating this directory did not apply any
patch or modify the running benchmark, evaluator, planner, or simulator.
