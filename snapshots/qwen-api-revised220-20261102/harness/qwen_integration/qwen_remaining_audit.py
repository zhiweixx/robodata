"""One-time authorization gate for the untouched alignment episode only."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

PRIOR_RUN_ID = "qwen3_27941342"
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TASK = "align_blocks"
EPISODE = TASK + "_s0_l0"
EPISODES = ["general_pickup_s0_l0", "stack_bowls_s0_l0", EPISODE]
FROZEN_FILES = (
    "local_model_driver.py", "model_driver.py", "model_episode.py", "model_relay.py",
    "robodojo_adapter.py", "aspire_glue.py", "native_retime.py", "deployment_docs.py",
    "model_inside.sh", "robodojo_entry.py", "qwen_gpu_guard.py", "run_qwen_suite.py",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "evidence must be a regular, non-symlink file")
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_rows(path):
    data = Path(path).read_bytes()
    require(bool(data) and data.endswith(b"\n"), "partial or empty prior ledger")
    return [json.loads(line) for line in data.splitlines()]


def verify_prior_job_inactive():
    result = subprocess.run(["squeue", "--noheader", "--jobs", PRIOR_RUN_ID.split("_")[1],
                             "--format=%T"], capture_output=True, text=True, check=True, timeout=20)
    require(not result.stdout.strip(), "prior Slurm job is still active")


def audit_prior_run(root, prior_run_id=PRIOR_RUN_ID, *, runtime_root=None):
    """Accept this known terminal length failure, never an arbitrary prior abort."""
    require(prior_run_id == PRIOR_RUN_ID, "only the reviewed original run may be continued")
    root = Path(root).resolve(strict=True)
    prior = root / "results" / prior_run_id
    runtime = Path(runtime_root).resolve(strict=True) if runtime_root else root / "qwen"
    require(not (prior / TASK).exists() and not (prior / TASK).is_symlink(), "alignment was already started")
    evidence = {}

    def read(path, lines=False):
        path = Path(path)
        evidence[str(path)] = digest(path)
        return json_rows(path) if lines else json.loads(path.read_text())

    server = read(runtime / "runs" / prior_run_id / "server_suite.json")
    require(server.get("run_id") == prior_run_id and server.get("finished_utc")
            and server.get("complete") is False, "prior supervisor has no terminal failure record")
    cleanup = {row.get("name"): row for row in server.get("cleanup", [])}
    require(set(cleanup) == {"server", "smoke", "suite"}
            and all(row.get("returncode") is not None and not row.get("error_type")
                    for row in cleanup.values()), "prior process cleanup is incomplete")
    require(server.get("model_revision") == REVISION, "prior model revision differs")
    require(server.get("package_versions", {}).get("vllm") == "0.32.0", "prior vLLM differs")
    manifest = read(runtime / "model_manifest.json")
    require(manifest.get("repo") == MODEL and manifest.get("revision") == REVISION,
            "current model manifest differs")
    require(evidence[str(runtime / "model_manifest.json")] == server.get("manifest_sha256"),
            "current manifest is not the prior pinned manifest")
    summary = read(prior / "launcher_summary.json")
    require(summary.get("complete") is False and summary.get("no_automatic_retry") is True
            and [a.get("task") for a in summary.get("attempts", [])] == ["general_pickup", "stack_bowls"],
            "prior launcher attempts are not the reviewed two attempts")
    attempts = read(prior / "attempts.jsonl", lines=True)
    starts = [row.get("task") for row in attempts if row.get("event") == "episode_process_start"]
    require(starts == ["general_pickup", "stack_bowls"], "alignment touched or duplicate prior attempt")
    require(all(row.get("task") in (None, "general_pickup", "stack_bowls") for row in attempts),
            "unexpected task in prior attempts")
    require([row.get("task") for row in attempts if row.get("event") == "episode_process_end"] == starts,
            "prior simulator attempt has no end record")
    config = read(prior / "controller_config.json")
    require(config.get("episode_ids") == EPISODES and config.get("model") == MODEL
            and all(config.get(k) == v for k, v in {"max_iterations": 8, "max_input_tokens": 128000,
                "max_output_tokens": 8192, "request_timeout_s": 400}.items()), "prior driver settings differ")
    experiment = read(prior / "experiment_config.json")
    require(experiment.get("tasks") == ["general_pickup", "stack_bowls", TASK]
            and experiment.get("seed") == 0 and experiment.get("layout") == 0
            and experiment.get("native_action_budgets") == {"general_pickup": 200, "stack_bowls": 800, TASK: 200}
            and experiment.get("process_timeout_s") == 3600 and experiment.get("sam3") is False
            and experiment.get("contact_graspnet") is False, "prior experiment protocol differs")
    source = read(prior / "source_snapshot.json")
    for filename in FROZEN_FILES:
        path = root / "qwen_integration" / filename
        evidence[str(path)] = digest(path)
        require(evidence[str(path)] == source.get(filename), "frozen policy/integration changed: " + filename)

    rows = read(prior / "usage.jsonl", lines=True)
    require(rows[0].get("event") == "suite_initialized", "prior ledger lacks initialization")
    ledger_config = rows[0].get("config", {})
    from local_model_driver import SAMPLING, THINKING
    require(ledger_config.get("episode_ids") == EPISODES and ledger_config.get("model") == MODEL
            and ledger_config.get("sampling") == SAMPLING and ledger_config.get("chat_template_kwargs") == THINKING
            and ledger_config.get("tokenize_timeout_s") == 60
            and all(ledger_config.get(k) == config[k] for k in
                    ("max_iterations", "max_input_tokens", "max_output_tokens", "request_timeout_s")),
            "prior ledger policy settings differ")
    expected_keys = [(EPISODES[0], i) for i in range(1, 8)] + [(EPISODES[1], i) for i in (1, 2)]
    events = ("program_input", "input_token_count", "generation_started", "raw_response", "generation_usage")
    require([r.get("event") for r in rows] == ["suite_initialized"] + list(events) * 9 + ["driver_stopped"],
            "prior ledger contains an uncertain, duplicate, or unexpected event")
    calls = {}
    for index, key in enumerate(expected_keys):
        group = rows[1 + index * 5: 6 + index * 5]
        require(all((row.get("episode_id"), row.get("iteration")) == key for row in group),
                "prior call identity differs or alignment was touched")
        inp, count, start, raw, usage = group
        response = raw.get("response", {})
        measured = usage.get("usage", {})
        require(response.get("model") == MODEL and len(response.get("choices", [])) == 1,
                "prior response model/choice differs")
        prompt = count.get("count")
        actual = response.get("usage", {})
        require(type(prompt) is int and 0 < prompt <= 128000 and count.get("approximate") is False
                and start.get("counted_input_tokens") == prompt
                and actual.get("prompt_tokens") == prompt and measured.get("input_tokens") == prompt
                and measured.get("preflight_matches_generation_usage") is True
                and measured.get("output_tokens") == actual.get("completion_tokens")
                and type(actual.get("completion_tokens")) is int and 0 <= actual["completion_tokens"] <= 8192,
                "prior token accounting is incomplete or inconsistent")
        choice = response["choices"][0]
        if key == (EPISODES[1], 2):
            require(choice.get("finish_reason") == "length" and not choice.get("message", {}).get("content")
                    and actual.get("completion_tokens") == 8192
                    and actual.get("completion_tokens_details", {}).get("reasoning_tokens") == 8192,
                    "stack failure is not the reviewed all-reasoning output limit")
        else:
            require(choice.get("finish_reason") == "stop", "unexpected earlier generation failure")
        calls[key] = group
    require(rows[-1].get("error_type") == "DriverStopped" and rows[-1].get("no_retry") is True
            and rows[-1].get("reason") == "generation incomplete or not a single stopped response",
            "prior driver stopped for an unreviewed reason")

    general = read(prior / "general_pickup" / "episode_report.json")
    from run_qwen_suite import validate
    validate(general, "general_pickup")
    require(len(general["iterations"]) == 7 and general["official"]["environment_steps"] == 200
            and general["official"]["official_success"] is False, "general pickup differs from completed failure")
    stack = read(prior / "stack_bowls" / "episode_report.json")
    require(stack.get("ok") is False and stack.get("episode_id") == EPISODES[1]
            and stack.get("model") == MODEL and stack.get("seed") == 0 and stack.get("layout_id") == 0
            and len(stack.get("iterations", [])) == 1
            and stack.get("error") == "DriverStopped: API controller stopped this suite; do not retry"
            and stack.get("official", {}).get("environment_steps") == 0
            and stack.get("official", {}).get("step_limit") == 800,
            "stack report is not the reviewed protocol failure")
    for task, count in (("general_pickup", 7), ("stack_bowls", 2)):
        inputs = sorted((prior / task).glob("public_input_*.json"))
        require([p.name for p in inputs] == [f"public_input_{i:02d}.json" for i in range(count)],
                "public input evidence is missing or duplicated")
        for p in inputs:
            read(p)
        for p in sorted((prior / task).glob("program_*.py")):
            evidence[str(p)] = digest(p)
        evidence[str(prior / task / "public_api.txt")] = digest(prior / task / "public_api.txt")
    require(evidence[str(prior / "general_pickup/public_api.txt")] == evidence[str(prior / "stack_bowls/public_api.txt")],
            "public API document changed between attempted tasks")

    relay = prior / "relay_audit"
    ready = read(relay / "ready.json")
    stopped = read(relay / "stopped.json")
    require(ready.get("episode_ids") == EPISODES and ready.get("max_iterations") == 8
            and stopped == {"reason": "controller_stopped", "no_retry": True}, "prior relay was not cleanly stopped")
    requests = sorted((relay / "requests").glob("*.json"))
    responses = sorted((relay / "responses").glob("*.json"))
    require(len(requests) == 9 and [p.name for p in requests] == [p.name for p in responses],
            "unmatched or extra prior relay request")
    seen = set()
    for request_path, response_path in zip(requests, responses):
        envelope, result = read(request_path), read(response_path)
        require(envelope.get("request_id") == request_path.stem == result.get("request_id"), "relay ID mismatch")
        payload = envelope.get("payload", {})
        episode = payload.get("episode_id")
        require(episode in EPISODES[:2], "alignment or unexpected task present in relay")
        if result.get("ok") is True:
            iteration = result.get("proposal", {}).get("iteration")
            require((episode, iteration) in calls, "relay proposal has no matching generation")
            require(result["proposal"].get("response_id") == calls[(episode, iteration)][3].get("response", {}).get("id"),
                    "relay proposal and generation differ")
        else:
            require(episode == EPISODES[1] and result.get("error_type") == "DriverStopped"
                    and result.get("no_retry") is True and result.get("stop_suite") is True,
                    "unexpected relay failure")
            iteration = 2
        key = (episode, iteration)
        require(key in calls and key not in seen, "duplicate or unmatched relay call")
        seen.add(key)
        task = episode.removesuffix("_s0_l0")
        public = json.loads((prior / task / f"public_input_{iteration - 1:02d}.json").read_text())
        require(all(payload.get(k) == public.get(k) for k in
                    ("task_instruction", "public_observation", "previous_result")), "relay/public input mismatch")
        require(payload.get("api_documentation") == (prior / task / "public_api.txt").read_text(),
                "relay API documentation mismatch")
    require(seen == set(expected_keys), "prior relay does not cover all measured calls")
    inventory_patterns = [(relay / "requests", "*.json"), (relay / "responses", "*.json")]
    for task in ("general_pickup", "stack_bowls"):
        inventory_patterns += [(prior / task, "public_input_*.json"), (prior / task, "program_*.py")]
    inventories = [{"directory": str(directory), "pattern": pattern,
                    "files": sorted(p.name for p in directory.glob(pattern))}
                   for directory, pattern in inventory_patterns]
    return {"schema_version": 1, "prior_run_id": prior_run_id, "prior_output": str(prior),
            "allowed_tasks": [TASK], "allowed_episode_ids": [EPISODE],
            "excluded_attempted_tasks": ["general_pickup", "stack_bowls"],
            "prior_calls": dict(zip(EPISODES, (7, 2, 0))),
            "prior_stack_failure": "8192 output tokens, all reasoning; no final content",
            "evidence_sha256": evidence, "evidence_inventories": inventories, "controller_config": config,
            "experiment_config": experiment, "no_retry_of_attempted_tasks": True}


def verify_evidence(audit):
    require(audit.get("prior_run_id") == PRIOR_RUN_ID and audit.get("allowed_episode_ids") == [EPISODE],
            "invalid continuation audit scope")
    prior = Path(audit["prior_output"])
    require(not (prior / TASK).exists() and not (prior / TASK).is_symlink(), "alignment was touched after audit")
    for path, expected in audit["evidence_sha256"].items():
        require(digest(path) == expected, "prior/frozen evidence changed after audit: " + path)
    for item in audit["evidence_inventories"]:
        require(sorted(p.name for p in Path(item["directory"]).glob(item["pattern"])) == item["files"],
                "prior evidence inventory changed after audit")
    if "claim" in audit:
        claim = audit["claim"]
        require(digest(claim["path"]) == claim["sha256"], "continuation claim changed")


def claim_continuation(audit, output):
    verify_evidence(audit)
    output = Path(output).resolve()
    require(re.fullmatch(r"qwen3_[0-9]+", output.name) and output.name != PRIOR_RUN_ID, "invalid new run ID")
    path = Path(audit["prior_output"]) / "continuation_claim.json"
    claim = {"schema_version": 1, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "prior_run_id": PRIOR_RUN_ID, "continuation_run_id": output.name,
             "continuation_output": str(output), "allowed_episode_ids": [EPISODE],
             "evidence_sha256": audit["evidence_sha256"], "no_automatic_resume": True}
    encoded = (json.dumps(claim, sort_keys=True, indent=2) + "\n").encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        raise RuntimeError("continuation already claimed; no automatic resume") from None
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    audit["claim"] = {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}
    return audit["claim"]
