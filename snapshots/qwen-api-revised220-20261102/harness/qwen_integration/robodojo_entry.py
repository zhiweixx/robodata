"""Run existing RoboDojo lifecycle with an in-process ASPIRE smoke policy.

Accepts the SAME CLI flags as src/eval_client/main.py. Set ASPIRE_OUTPUT_DIR.
Use --policy_name ASPIRE_Pilot --port 0; no WebSocket/model server is started.
ASPIRE src and this integration directory must already be on PYTHONPATH.
"""
from __future__ import annotations
import json
import os
from pathlib import Path


def main():
    # Import launches Isaac via RoboDojo's normal AppLauncher before any env import.
    from src.eval_client import main as native
    from src.eval_client import eval_env
    from omegaconf import OmegaConf
    from smoke import run_smoke

    class InProcessPolicyBoundary:
        def __init__(self, **kwargs):
            pass
        def call(self, *, func_name, **kwargs):
            if func_name != "reset":
                raise RuntimeError(f"unexpected remote policy request {func_name}")
            return None
        def close(self):
            pass

    # This explicitly replaces unused transport only. All actions go via EvalEnv.
    eval_env.WsModelClient = InProcessPolicyBoundary
    original_create = native.create_eval_env
    reports = []

    def create(config, app, **kwargs):
        for key, value in {"depth":True,"approximate_depth":False,
                           "intrinsic_matrix":True,"extrinsic_matrix":True}.items():
            OmegaConf.update(config, f"eval_cfg.observation.vision.{key}", value, force_add=True)
        for name in list(config.camera.annotator):
            OmegaConf.update(config, f"camera.annotator.{name}.distance_to_image_plane_capture",
                {"type":"distance_to_image_plane","device":"cpu"}, force_add=True)
        env = original_create(config, app, **kwargs)
        def evaluate():
            mode=os.environ.get("ASPIRE_MODE","smoke")
            if mode == "smoke":
                reports.append(run_smoke(env))
            elif mode == "model":
                from model_episode import run_model_episode
                reports.append(run_model_episode(env))
            elif mode == "retime_diagnostic":
                from retime_diagnostic import run_retime_diagnostic
                reports.append(run_retime_diagnostic(env))
            else:
                raise ValueError(f"unknown ASPIRE_MODE {mode!r}")
        env.eval_one_episode = evaluate
        env.eval_one_episode_batch = evaluate
        return env

    native.create_eval_env = create
    native.main()
    if len(reports) != 1 or not reports[0].get("ok"):
        raise SystemExit("ASPIRE episode runner failed or no complete episode was run; inspect output report")
    print("ASPIRE_ROBODOJO_RUN_COMPLETED", json.dumps({"mode":os.environ.get("ASPIRE_MODE","smoke"),"output":str(Path(os.environ["ASPIRE_OUTPUT_DIR"]))}))


if __name__ == "__main__":
    main()
