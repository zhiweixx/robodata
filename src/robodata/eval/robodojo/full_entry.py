"""Run the pinned native RoboDojo lifecycle with a full-budget ASPIRE policy."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time

from robodata.eval.robodojo.full_episode import FullEvaluationInterrupted, run_full_episode, write_json


def effective_eval_count(native_count, *, mode, override):
    """Mirror the native outer loop's count without adding full-run caps."""
    native_count = int(native_count)
    if native_count <= 0:
        raise ValueError("native evaluation count must be positive")
    overridden = override is not None and str(override).lower() not in ("", "native")
    if mode == "evaluation":
        if overridden:
            raise ValueError("full evaluation requires EVAL_NUM=native or unset")
        return native_count
    if mode != "interface_smoke":
        raise ValueError("unknown ASPIRE_FULL_MODE")
    if not overridden:
        return native_count
    count = int(override)
    if count <= 0:
        raise ValueError("smoke evaluation count must be positive")
    return min(count, native_count)


def _interface_smoke(env, output):
    """One hold tick, calibrated public sensors and a native CuRobo planning check."""
    from cap_harness.contracts import ArmCommand, RobotAction
    from robodata.environments.robodojo.full_adapter import FullRoboDojoAdapter
    from robodata.eval.robodojo.full_episode import finalize_native_episode
    from robodata.execution.model_episode import public_observation
    from PIL import Image
    adapter = FullRoboDojoAdapter(env)
    started = time.monotonic()
    report = {"scope": "interface_smoke_only", "model_requests": 0, "ok": False}
    try:
        report["camera_calibration"] = adapter.calibration_sanity()
        observation = adapter.get_observation()
        for name, camera in observation.cameras.items():
            Image.fromarray(camera.rgb).save(output / f"{name}_initial.png")
        write_json(output / "public_initial_observation.json", public_observation(adapter))
        state = adapter.get_robot_state()
        planned = adapter.native_plan_to_joints(state, state.joint_positions["right"], arm="right")
        report["native_planning_ok"] = bool(planned.get("ok"))
        if not planned.get("ok"):
            raise RuntimeError(f"native CuRobo smoke failed: {planned.get('error')}")
        step = adapter.step(RobotAction({"right": ArmCommand(
            mode="joint_position", target=state.joint_positions["right"],
            embodiment="robodojo")}))
        if not step.ok or int(env.take_action_cnt[0]) != 1:
            raise RuntimeError("native hold action did not advance exactly one tick")
        report.update(ok=True, official=finalize_native_episode(env, adapter))
        return report
    finally:
        report["wall_seconds"] = time.monotonic() - started
        write_json(output / "smoke_report.json", report)
        adapter.close()


def main():
    # Native import launches Isaac before app-dependent robot modules are used.
    from src.eval_client import main as native
    from src.eval_client import eval_env
    from omegaconf import OmegaConf
    from utils.cluttered_generator import UnStableError
    import torch

    output = Path(os.environ["ASPIRE_OUTPUT_DIR"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("ASPIRE_FULL_MODE", "evaluation")
    if mode not in ("evaluation", "interface_smoke"):
        raise ValueError("unknown ASPIRE_FULL_MODE")
    cuda_index = int(os.environ.get("ASPIRE_CUDA_DEVICE_INDEX", "0"))
    torch.cuda.set_device(cuda_index)
    known_native = (native.PhysXFatalError, native.PhysXBrokenError, UnStableError)
    last_env = None
    reports = []
    gpu_report = None
    started = time.monotonic()
    group_error = None

    class InProcessPolicyBoundary:
        def __init__(self, **kwargs):
            pass
        def call(self, *, func_name, **kwargs):
            if func_name != "reset":
                raise RuntimeError(f"unexpected remote policy call {func_name}")
        def close(self):
            pass

    eval_env.WsModelClient = InProcessPolicyBoundary
    original_create = native.create_eval_env
    original_process_config = native.process_config

    def process_config(*args, **kwargs):
        # Native main copies eval_cfg into OmegaConf, then later updates only
        # its original dict with eval_num. Keep EvalEnv metadata in agreement
        # with the native outer loop, without processing the scene twice or
        # changing the original process_config return count.
        config, native_count = original_process_config(*args, **kwargs)
        effective_count = effective_eval_count(
            native_count, mode=mode, override=os.environ.get("EVAL_NUM"))
        OmegaConf.update(config, "eval_cfg.eval_num", effective_count, force_add=True)
        return config, native_count

    def create(config, app, **kwargs):
        nonlocal last_env
        for key, value in {"depth": True, "approximate_depth": False,
                           "intrinsic_matrix": True, "extrinsic_matrix": True}.items():
            OmegaConf.update(config, f"eval_cfg.observation.vision.{key}", value, force_add=True)
        for name in list(config.camera.annotator):
            OmegaConf.update(config, f"camera.annotator.{name}.distance_to_image_plane_capture",
                             {"type": "distance_to_image_plane", "device": "cpu"}, force_add=True)
        env = original_create(config, app, **kwargs)
        last_env = env
        pending = {"report": None, "client": None}
        original_reset, original_run = env.reset, env.run_eval

        def checkpoint():
            try:
                return env.persist_resume_manifest()
            except Exception as exc:
                write_json(output / "checkpoint_error.json", {"error": str(exc)})

        def reset(*args, **kw):
            try:
                return original_reset(*args, **kw)
            except known_native:
                raise
            except Exception as exc:
                checkpoint()
                raise FullEvaluationInterrupted(f"native reset interrupted: {type(exc).__name__}: {exc}") from exc

        def evaluate():
            nonlocal gpu_report
            from robodata.eval.robodojo.full_gpu_guard import verify_simulator_gpu
            gpu_report = verify_simulator_gpu(
                os.environ.get("ASPIRE_SIM_LOG", str(output / "simulator.log")),
                os.environ["ASPIRE_SIM_GPU_UUID"],
                vulkan_index=int(os.environ["ASPIRE_VULKAN_INDEX"]),
                cuda_index=cuda_index, env=env)
            write_json(output / "gpu_isolation.json", gpu_report)
            if mode == "interface_smoke":
                report = _interface_smoke(env, output)
            else:
                from robodata.agents.full_relay import FullRelayClient
                client = FullRelayClient(Path(os.environ["ASPIRE_RELAY_DIR"]))
                pending["client"] = client
                report = run_full_episode(env, client_factory=lambda path: client)
            pending["report"] = report
            reports.append(report)

        def run_eval():
            pending.update(report=None, client=None)
            # run_eval mutates details before MP4 finalization. Roll back partial
            # in-memory commit on an infrastructure exception before checkpoint.
            before = {name: deepcopy(getattr(env, name)) for name in
                      ("success_nums", "fail_nums", "unstable_nums", "total_score", "eval_result")}
            try:
                original_run()
            except known_native:
                raise
            except BaseException as exc:
                for name, value in before.items():
                    setattr(env, name, value)
                checkpoint()
                report, client = pending["report"], pending["client"]
                if client and report:
                    try:
                        client.abandon_episode(report["episode_id"], "infrastructure_interrupted")
                    except Exception:
                        pass
                if isinstance(exc, FullEvaluationInterrupted):
                    raise
                raise FullEvaluationInterrupted(f"native evaluation interrupted: {type(exc).__name__}: {exc}") from exc
            report, client = pending["report"], pending["client"]
            if report is not None and client is not None:
                scored = any(int(row["layout_id"]) == report["layout_id"]
                             for row in env.eval_result["details"].values())
                try:
                    if scored:
                        client.finish_episode(report["episode_id"], report["termination_reason"])
                    else:
                        client.abandon_episode(report["episode_id"], "native_unstable")
                except Exception as exc:
                    # Native score is already durable: preserve that checkpoint,
                    # never let main's generic handler silently consume a seed.
                    checkpoint()
                    raise FullEvaluationInterrupted(f"post-score relay interrupted: {exc}") from exc
                report["native_score_committed"] = scored
                write_json(Path(report["artifact_dir"]) / "episode_report.json", report)
            summary = publish_group_report("native_score_committed")
            write_json(output / "group_progress.json", summary)

        env.reset, env.run_eval = reset, run_eval
        env.eval_one_episode = evaluate
        env.eval_one_episode_batch = evaluate
        return env

    def group_summary(env):
        result_path = Path(env.save_dir).resolve() / "_result.json"
        details = env.eval_result.get("details", {})
        videos = sorted(Path(env.save_dir).resolve().glob("episode_*.mp4"))
        valid_videos = [p for p in videos if p.stat().st_size > 0]
        scored = int(env.success_nums + env.fail_nums)
        target = int(env.eval_num)
        video_ok = len(valid_videos) == 3 * scored
        smoke_report = reports[-1] if mode == "interface_smoke" and reports else None
        smoke_ok = mode != "interface_smoke" or (
            smoke_report is not None and smoke_report.get("ok") is True
            and smoke_report.get("native_planning_ok") is True
            and smoke_report.get("model_requests") == 0
            and gpu_report is not None and gpu_report.get("verified") is True)
        return {"mode": mode, "complete": scored == target and video_ok and smoke_ok,
                "scores_complete": scored == target, "target": target, "scored": scored,
                "success": int(env.success_nums), "fail": int(env.fail_nums),
                "unstable": int(env.unstable_nums), "task": env.task_name,
                "eval_seed": int(env.eval_seed), "native_run_id": env.run_id,
                "native_result_path": str(result_path), "score": env.eval_result.get("score"),
                "details": details, "videos_validated": video_ok,
                "video_validation": "three nonempty native camera MP4s per scored episode",
                "video_count": len(valid_videos), "video_paths": [str(p) for p in valid_videos],
                "gpu_isolation": gpu_report, "wall_seconds": time.monotonic()-started,
                "interface_smoke": smoke_report,
                "interface_smoke_ok": smoke_ok if mode == "interface_smoke" else None,
                "model_episode_reports_this_process": len(reports), "error": group_error}

    def publish_group_report(stage):
        """Write before native Kit shutdown, which can terminate the process."""
        if last_env is not None:
            summary = group_summary(last_env)
            if group_error:
                summary["complete"] = False
            if not summary["complete"]:
                # Native main clears its manifest even on candidate exhaustion.
                # Restore it whenever this bridge cannot certify completion so
                # scored layouts are not accidentally repeated on a later job.
                try:
                    checkpoint = last_env.persist_resume_manifest()
                    summary["native_resume_manifest"] = (
                        str(Path(checkpoint).resolve()) if checkpoint else None)
                except Exception as exc:
                    summary["checkpoint_error"] = f"{type(exc).__name__}: {exc}"
        else:
            summary = {"complete": False, "error": group_error, "mode": mode,
                       "scored": 0, "wall_seconds": time.monotonic()-started}
        summary["report_stage"] = stage
        write_json(output / "group_report.json", summary)
        return summary

    original_app_close = native.simulation_app.close

    def close_with_report(*args, **kwargs):
        # Isaac Sim may call os._exit inside close(). A Python finally after
        # native.main cannot provide completion evidence in that runtime.
        publish_group_report("before_simulation_app_close")
        return original_app_close(*args, **kwargs)

    native.simulation_app.close = close_with_report
    native.create_eval_env = create
    native.process_config = process_config
    try:
        native.main()
    except BaseException as exc:
        group_error = f"{type(exc).__name__}: {exc}"
        if last_env is not None:
            try:
                last_env.persist_resume_manifest()
            except Exception:
                pass
        raise
    finally:
        summary = publish_group_report("python_finally")
    if not summary["complete"]:
        raise SystemExit("native task group incomplete; inspect group_report.json")
    print("ASPIRE_FULL_GROUP_COMPLETED", json.dumps({key: summary[key] for key in
          ("mode", "task", "eval_seed", "scored", "target", "success", "native_result_path")}))


if __name__ == "__main__":
    main()
