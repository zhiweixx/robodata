"""Runtime owner for the deterministic native-ASPIRE smoke program."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image

from cap_harness.artifacts import RunRecorder
from aspire_glue import build_api, execute_program, render_api_docs, implementation_provenance
from robodojo_adapter import RoboDojoAdapter


def run_smoke(task_env):
    root = Path(os.environ["ASPIRE_OUTPUT_DIR"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    recorder = RunRecorder(output_root=root / "artifacts", benchmark="RoboDojo",
        suite="Open", task_id=0, task_name=task_env.task_name, seed=0,
        max_steps=min(int(task_env.step_lim), 128), camera_width=640, camera_height=480,
        control_frequency_hz=25, capture_videos=False, step_video=False,
        providers={"planner":"RoboDojo-native-CuRobo", "mode":"deterministic-smoke"})
    adapter = RoboDojoAdapter(task_env, recorder=recorder)
    started = time.monotonic()
    report = {"scope":"ASPIRE integration smoke; not a task-solving evaluation",
        "model_requests":0,"artifact_dir":str(recorder.root),"ok":False}
    report["implementation_provenance"] = implementation_provenance()
    try:
        calibration = adapter.calibration_sanity()
        report["camera_calibration"] = calibration
        report["controller_metadata"] = adapter.get_controller_metadata()
        observation = adapter.get_observation()
        recorder.on_reset(observation, {"task":task_env.task_name,"layout_id":int(task_env.env_seeds[0])})
        # Only calibrated public sensor data, no semantic/object-position oracle.
        arrays = {}
        for name, camera in observation.cameras.items():
            arrays[f"{name}_rgb"] = camera.rgb
            arrays[f"{name}_depth_m"] = camera.depth_m
            arrays[f"{name}_K"] = camera.intrinsics
            arrays[f"{name}_world_from_camera"] = camera.camera_pose.as_matrix()
            Image.fromarray(camera.rgb).save(root / f"{name}_initial.png")
        np.savez_compressed(root / "public_initial_rgbd.npz", **arrays)
        report["task_language"] = observation.task_context.language
        api = build_api(adapter, recorder=recorder)
        (root / "public_api.txt").write_text(render_api_docs(api), encoding="utf-8")
        program = Path(__file__).with_name("smoke_program.py")
        outcome = execute_program(api, program, recorder)
        official = adapter.official_status()
        report.update(ok=bool(outcome.ok), program_result=outcome.result,
            program_error=None if outcome.error is None else str(outcome.error),
            program_stdout=outcome.stdout, public_tool_calls=len(outcome.calls), official=official)
        recorder.finalize(program_ok=outcome.ok, task_success=official["official_success"],
            protocol_success=outcome.ok, termination_reason="program_completed" if outcome.ok else "program_error",
            program_result=outcome, error=outcome.error)
        return report
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        recorder.close_incomplete(exc)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        (root / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        # EvalEnv starts success=True to mean scene-valid. It is NOT task success.
        official = adapter.official_status()
        task_env.success[0] = bool(official["official_success"])
        task_env.end_flag[0] = True
        adapter.close()
