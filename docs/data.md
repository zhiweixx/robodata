# Data and artifacts

The current evaluator saves **model-turn evidence** and native simulator outputs.
It does not yet export a dense action-aligned training dataset.

A campaign owns a portable state manifest, machine configuration, source inventory,
frozen source tree, a disjoint group plan and per-group attempts. Native RoboDojo
owns its score and resume checkpoints. The integration records each model turn's
public observations (camera images/calibration and robot state), generated Python
program, public execution receipts, token usage, timing and termination reason.
Three native camera videos accompany scored episodes. Interrupted attempts remain
separate from committed scores; all request usage must remain available.

The public model observation does not contain evaluator-only success labels or
privileged object state. Results used for analysis may contain native evaluation
labels, but those labels must not be fed back as extra policy observations.

For a downstream data pipeline, preserve these identity fields when exporting:

- campaign, source/model/provider revisions and environment identity;
- task/configuration, seed, native layout ID and initial-state content hash;
- attempt ID, model-turn index and timestamp;
- observation references, generated program and tool receipts;
- native action count, termination reason and score commit identity;
- request usage including interrupted requests;
- exposure/split identity independently from success/failure labels.

Dense per-control-tick action/state/camera alignment requires additional recording
and a separately validated schema. Do not infer such alignment from a turn-level
program or a video frame index alone. `data/provenance.py` currently implements
source identities; it is not a training-data exporter.

Keep videos, images, raw trajectories, logs, weights, kernels and simulator assets
in external artifact storage. Git should contain small manifests, schemas,
checksums, fixtures and documented artifact references only. Redact credentials
and local private paths before sharing any runtime record.
