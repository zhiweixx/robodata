"""One native layout with ASPIRE programs until official/model termination.

The native EvalEnv owns action limits, success/partial scoring and MP4 videos.
Only public per-model-turn observations, programs and receipts are archived here.
No paid API, turn cap, output cap, program alarm or per-episode deadline is used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from robodata.execution.aspire_glue import build_api, execute_program, implementation_provenance
from robodata.execution.model_episode import plain, public_observation
from robodata.environments.robodojo.full_adapter import FullRoboDojoAdapter
from robodata.eval.robodojo.full_deployment_docs import full_api_documentation
from robodata.execution.native_stdout import archive_native_stdout


class FullEvaluationInterrupted(BaseException):
    """Bypass native generic-Exception skip/replacement for bridge outages."""


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))
    temporary.replace(path)


def public_feedback(outcome):
    """Full program stdout/receipts; native tick progress is archived separately."""
    return {
        "program_ok": bool(outcome.ok), "result": plain(outcome.result),
        "stdout": outcome.stdout,
        "error": None if outcome.error is None else str(outcome.error),
        "tool_calls": [{"tool": c.tool, "ok": c.ok, "duration_s": c.duration_s,
                        "failure_code": c.failure_code, "error_message": c.error_message,
                        "inputs_summary": plain(c.inputs_summary),
                        "outputs_summary": plain(c.outputs_summary), "frame": c.frame,
                        "provider": c.provider, "result_type": c.result_type}
                       for c in outcome.calls],
    }


def finalize_native_episode(task_env, adapter):
    """Ask the native scorer for final checking when the model stops early."""
    if not task_env.is_episode_end():
        task_env.success[0] = False
        # Native final_check handles tasks whose goal is only checked at an end.
        task_env.is_episode_end()
    official = adapter.official_status()
    # Initial success=True means layout validity. Never score that as policy success.
    task_env.success[0] = bool(official["official_success"])
    task_env.end_flag[0] = True
    return official


def run_full_episode(task_env, *, client_factory=None, adapter_factory=FullRoboDojoAdapter):
    from robodata.agents.full_relay import FullRelayClient, ModelEpisodeFailure
    forbidden = sorted(k for k in os.environ if k in {
        "OPENAI_API_KEY", "OPENAI_API_KEY_FILE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
        "ARK_API_KEY", "ARK_API_KEY_BACKUP", "SAFEHARNESS_KEY_FILE"})
    if forbidden:
        raise RuntimeError("simulator inherited credential variables: " + ",".join(forbidden))
    eval_seed = int(task_env.eval_seed)
    if eval_seed not in (0, 1, 2) or task_env.num_envs != 1:
        raise ValueError("full evaluator requires one native environment and eval_seed0/1/2")
    layout_id = int(task_env.env_seeds[0])
    episode_id = f"{task_env.task_name}_s{eval_seed}_l{layout_id}"
    group_root = Path(os.environ["ASPIRE_OUTPUT_DIR"]).resolve()
    episode_root = group_root / "episodes" / episode_id
    # Preserve interrupted attempts: a native restart can revisit an unscored layout.
    episode_root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (episode_root / f"attempt_{attempt:03d}").exists():
        attempt += 1
    root = episode_root / f"attempt_{attempt:03d}"
    root.mkdir()
    progress_path = root / "execution_state.json"
    adapter = None
    client = None
    started = time.monotonic()
    reason = "interrupted"
    completed = False
    report = {
        "episode_id": episode_id, "task": task_env.task_name,
        "layout_id": layout_id, "eval_seed": eval_seed, "seed": eval_seed,
        "attempt": attempt, "model": os.environ.get("ASPIRE_MODEL_NAME", "Qwen/Qwen3.8-27B"),
        "ok": False, "status": "running", "iterations": [], "artifact_dir": str(root),
        "native_save_dir": str(Path(task_env.save_dir).resolve()),
        "step_limit": int(task_env.step_lim),
        "implementation_provenance": implementation_provenance(),
        "scope": "ASPIRE public API/program executor + custom ARX X5 bridge and native CuRobo; SAM3 and Contact-GraspNet enabled; no VLA or offline skill discovery",
        "limits": {"program_count": None, "program_wall_seconds": None,
                   "output_tokens": "remaining native model context", "action_budget": "native task config"},
    }

    def phase(name):
        write_json(progress_path, {"episode_id": episode_id, "attempt": attempt,
                   "phase": name, "unix_time": time.time(),
                   "environment_steps": int(task_env.take_action_cnt[0]),
                   "step_limit": int(task_env.step_lim)})

    try:
        phase("initializing_adapter")
        adapter = adapter_factory(task_env, progress_path=progress_path)
        report["camera_calibration"] = adapter.calibration_sanity()
        initial = adapter.get_observation()
        api = build_api(adapter, recorder=None)
        api_docs = full_api_documentation(api)
        (root / "public_api.txt").write_text(api_docs)
        client = (client_factory or FullRelayClient)(Path(os.environ["ASPIRE_RELAY_DIR"]))
        previous = None
        index = 0
        while True:
            if task_env.is_episode_end():
                reason = "environment_ended"
                break
            phase("observing")
            sensed = public_observation(adapter)
            write_json(root / f"public_input_{index:04d}.json",
                       {"task_instruction": initial.task_context.language,
                        "public_observation": sensed, "previous_result": previous})
            phase("waiting_for_model")
            try:
                proposal = client.next_program(
                    episode_id=episode_id, task_instruction=initial.task_context.language,
                    public_observation=sensed, previous_result=previous,
                    api_documentation=api_docs)
            except ModelEpisodeFailure as exc:
                reason = str(exc.code)
                report["model_failure"] = {"code": str(exc.code), "message": str(exc)}
                break
            entry = {"iteration": proposal.iteration, "summary": proposal.summary,
                     "done": proposal.done, "usage": plain(proposal.usage),
                     "budget": plain(proposal.budget)}
            report["iterations"].append(entry)
            if proposal.done:
                reason = "model_done"
                break
            entry["source"] = f"program_{index:04d}.py"
            program = root / entry["source"]
            program.write_text(proposal.code, encoding="utf-8")
            phase("executing_program")
            program_started = time.monotonic()
            entry["native_stdout_log"] = f"native_stdout_{index:04d}.log"
            with archive_native_stdout(root / entry["native_stdout_log"]):
                outcome = execute_program(api, program, recorder=None)
            entry["program_wall_seconds"] = time.monotonic() - program_started
            previous = public_feedback(outcome)
            entry["execution"] = previous
            entry["environment_steps_after"] = int(task_env.take_action_cnt[0])
            write_json(root / "episode_progress.json", report)
            # ProgramExecutor reports tool exceptions as receipts. Re-surface any
            # native PhysX monitor exception to the official recovery handler.
            if getattr(task_env, "physx_monitor_enabled", False):
                task_env._check_physx_broken_envs()
            index += 1
        phase("native_final_check")
        report["official"] = finalize_native_episode(task_env, adapter)
        report.update(ok=True, status="completed", termination_reason=reason,
                      total_programs_executed=sum(not i["done"] for i in report["iterations"]))
        completed = True
        return report
    except BaseException as exc:
        report.update(status="interrupted", error=f"{type(exc).__name__}: {exc}",
                      termination_reason="interrupted")
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["environment_steps"] = int(task_env.take_action_cnt[0])
        # Only native run_eval can commit a score/video. full_entry calls finish
        # after that commit; a return from this callback alone is not completion.
        if client is not None and not completed:
            try:
                client.abandon_episode(episode_id=episode_id, reason="infrastructure_interrupted")
            except BaseException as exc:
                report["relay_finish_error"] = f"{type(exc).__name__}: {exc}"
        write_json(root / "episode_report.json", report)
        write_json(episode_root / "latest_attempt.json",
                   {"attempt": attempt, "report": str(root / "episode_report.json"),
                    "status": report["status"]})
        if adapter is not None:
            adapter.close()
