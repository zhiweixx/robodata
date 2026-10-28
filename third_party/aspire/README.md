# Vendored ASPIRE runtime

Runtime source from NVlabs/ASPIRE at `fa00f0ef2bc1d91d594b9aa502c22133c9f538f7`, with the two reviewed patches in `../../patches/aspire`. LICENSE and NOTICE are preserved. `SOURCE_MANIFEST.json` records every copied file hash and each omitted source-tree artifact.

This is the code-as-policy API/executor and provider-client dependency of RoboData. It is **not** a complete ASPIRE experiment or offline skill-discovery pipeline. RoboDojo uses its own environment/robot bridge and native planner.

All Python runtime source and small packaged configuration are preserved. Real-YAM STL meshes, caches, upstream datasets/weights, demos, development tooling, and external dependency checkouts are omitted. Consequently this distribution does not support real-YAM deployment. SAM3 and Contact-GraspNet model code and weights are prepared separately at pinned revisions, subject to their own licenses and access requirements. Optional upstream simulator backends need their separate dependencies; the supported entrypoint here is RoboData's RoboDojo evaluator.

Install with `python -m pip install -e third_party/aspire` from the RoboData root. This intentionally minimal package does not install GPU providers or simulator dependencies into the CPU/client environment.
