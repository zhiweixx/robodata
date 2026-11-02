"""Run exactly the untouched alignment episode after the reviewed length abort."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

from qwen_remaining_audit import (
    EPISODE, PRIOR_RUN_ID, TASK, audit_prior_run, claim_continuation,
    require, verify_evidence, verify_prior_job_inactive,
)
from run_qwen_suite import clean_environment, terminate, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prior-run-id", required=True)
    args = parser.parse_args()
    require(args.run_id == "qwen3_" + os.environ.get("SLURM_JOB_ID", ""), "new run ID must match Slurm job")
    require(args.prior_run_id == PRIOR_RUN_ID and args.run_id != PRIOR_RUN_ID, "invalid prior/new run pair")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    integration = Path(__file__).resolve().parent
    root, simroot = integration.parent, integration.parent.parent
    verify_prior_job_inactive()
    audit = audit_prior_run(root, args.prior_run_id)
    output = root / "results" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    claim_continuation(audit, output)
    audit_path = output / "prior_run_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    config = {**audit["controller_config"], "usage_jsonl": str(output / "usage.jsonl"),
              "base_url": os.environ["ASPIRE_LOCAL_BASE_URL"]}
    (output / "controller_config.json").write_text(json.dumps(config, indent=2))
    snapshot = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in integration.iterdir() if p.is_file() and p.suffix in (".py", ".sh", ".sbatch")}
    (output / "source_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    experiment = {**audit["experiment_config"], "tasks": [TASK],
        "authorized_original_tasks": ["general_pickup", "stack_bowls", TASK],
        "prior_run_id": PRIOR_RUN_ID, "continuation_claim": audit["claim"],
        "no_retry_of_attempted_tasks": True, "native_action_budgets": {TASK: 200},
        "simulator_gpu": os.environ["ASPIRE_SIM_GPU"],
        "simulator_gpu_uuid": os.environ["ASPIRE_SIM_GPU_UUID"],
        "simulator_cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
    (output / "experiment_config.json").write_text(json.dumps(experiment, indent=2))
    private = Path("/home/ROBODATA_USER/.config/qwen-pilot")
    private.mkdir(mode=0o700, parents=True, exist_ok=True)
    relay = private / ("relay-" + args.run_id)
    relay.mkdir(mode=0o700, exist_ok=False)
    env = clean_environment(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    attempts, complete, final_budget, controller, sim = [], False, None, None, None
    began = time.monotonic()

    def record(event, **fields):
        row = {"event": event, "utc": dt.datetime.now(dt.timezone.utc).isoformat(), **fields}
        with (output / "attempts.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n"); handle.flush(); os.fsync(handle.fileno())
        print(json.dumps(row), flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        verify_evidence(audit)
        record("continuation_claim_verified", prior_run_id=PRIOR_RUN_ID, allowed_task=TASK, claim=audit["claim"])
        with (output / "controller.log").open("w") as log:
            controller = subprocess.Popen([sys.executable, str(integration / "qwen_remaining_driver.py"),
                "--config", str(output / "controller_config.json"), "--audit", str(audit_path),
                "--relay-dir", str(relay)], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 60
        while not (relay / "ready.json").exists():
            if controller.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("continuation controller not ready; no automatic retry")
            time.sleep(.2)
        require(json.loads((relay / "ready.json").read_text()).get("episode_ids") == [EPISODE],
                "controller is not restricted to the untouched alignment episode")
        record("controller_ready", api_cost_usd=0)
        verify_evidence(audit)
        taskout = output / TASK
        taskout.mkdir()
        simenv = clean_environment(env)
        simenv["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        simenv["SINGULARITY_BINDPATH"] = str(relay) + ":/aspire-relay"
        record("episode_process_start", task=TASK, seed=0, layout_id=0)
        started = time.monotonic()
        with (taskout / "simulator.log").open("w") as log:
            sim = subprocess.Popen(["bash", str(simroot / "scripts/container_exec.sh"), "/bin/bash",
                "/work/aspire_validate/qwen_integration/model_inside.sh", TASK, args.run_id, EPISODE,
                os.environ["ASPIRE_SIM_GPU_UUID"]], cwd=simroot, env=simenv,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            rc = sim.wait(timeout=3600)
        sim = None
        attempt = {"task": TASK, "exit_code": rc, "wall_seconds": time.monotonic() - started}
        attempts.append(attempt)
        record("episode_process_end", **attempt)
        require(rc == 0, "alignment simulator failed; preserve evidence without retry")
        report = json.loads((taskout / "episode_report.json").read_text())
        official = validate(report, TASK)
        attempt.update(validated_episode=True, model_calls=len(report["iterations"]),
                       programs_executed=report["total_programs_executed"], **official)
        record("episode_validated", **attempt)
        from model_relay import RelayModelClient
        final_budget = RelayModelClient(relay).stop()["budget"]
        require(controller.wait(timeout=10) == 0 and not final_budget["aborted"], "controller shutdown failed")
        require(final_budget["calls"][EPISODE] == attempt["model_calls"]
                and all(final_budget["calls"][e] == 0 for e in ("general_pickup_s0_l0", "stack_bowls_s0_l0")),
                "continuation call ledger differs or an attempted task was retried")
        verify_evidence(audit)
        complete = True
        record("remaining_episode_complete", prior_run_id=PRIOR_RUN_ID)
    finally:
        terminate(sim)
        terminate(controller)
        shutil.copytree(relay, output / "relay_audit")
        shutil.rmtree(relay)
        (output / "launcher_summary.json").write_text(json.dumps({
            "complete": complete, "scope": "untouched alignment only", "prior_run_id": PRIOR_RUN_ID,
            "continuation_claim": audit["claim"], "attempts": attempts,
            "wall_seconds": time.monotonic() - began, "budget": final_budget,
            "api_cost_usd": 0, "no_automatic_retry": True,
            "excluded_attempted_tasks": ["general_pickup", "stack_bowls"]}, indent=2))


if __name__ == "__main__":
    main()
