"""Own one local vLLM server, its interface smoke, and three authorized episodes.

Run only inside the four-L40S Slurm allocation. Isaac's Vulkan device zero uses
the first visible device; vLLM uses the second and third devices.
This process never reads credentials or downloads model weights.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib import request


MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_ROOT = Path("/srv/ROBODATA_RUNTIME/isaac6/aspire_validate")
ENV_REPORT_KEYS = (
    "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
    "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "ASPIRE_SIM_GPU", "ASPIRE_SIM_GPU_UUID",
    "ASPIRE_LOCAL_BASE_URL", "HF_HOME", "HF_TOKEN_PATH", "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN", "HF_DATASETS_OFFLINE",
    "VLLM_NO_USAGE_STATS", "VLLM_USE_FLASHINFER_SAMPLER", "NO_PROXY", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM",
    "ASPIRE_SPLIT_CPU", "ASPIRE_PRIOR_RUN_ID",
)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def allocated_devices(value):
    devices = value.split(",") if value else []
    if (len(devices) != 4 or len(set(devices)) != 4
            or any(not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", item)
                   for item in devices)):
        raise RuntimeError("expected four distinct allocated CUDA_VISIBLE_DEVICES tokens")
    return devices


def cpu_partition(enabled):
    """Opt-in initialization diagnostic; set launch affinity, not a CPU cgroup."""
    if not enabled:
        return {"enabled": False, "method": "inherited scheduler affinity"}
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("ASPIRE_SPLIT_CPU=1 requires sched_getaffinity support")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 16:
        raise RuntimeError("ASPIRE_SPLIT_CPU=1 requires at least 16 allocated affinity CPUs")
    taskset = shutil.which("taskset")
    if not taskset:
        raise RuntimeError("ASPIRE_SPLIT_CPU=1 requires the taskset executable")
    middle = len(cpus) // 2
    return {
        "enabled": True, "method": "taskset launch affinity, inherited by children",
        "diagnostic_only": True, "taskset": taskset, "allocated_cpus": cpus,
        "server_cpus": cpus[:middle], "smoke_cpus": cpus[middle:],
        "suite_cpus": cpus[middle:],
    }


def with_cpu_partition(command, partition, role):
    if not partition["enabled"]:
        return list(command)
    return [partition["taskset"], "--cpu-list",
            ",".join(str(cpu) for cpu in partition[role + "_cpus"]), *command]


def simulator_proxy_names(source):
    """Return only inherited proxy variable names, including container forwarding."""
    names = []
    for name in source:
        normalized = name.upper()
        for prefix in ("SINGULARITYENV_", "APPTAINERENV_"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        if normalized in {"HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY", "NO_PROXY"}:
            names.append(name)
    return sorted(names)


def restore_simulator_proxies(sanitized, source):
    """Preserve the cluster's asset-access route only for the simulator suite.

    Proxy values may contain credentials: never put this environment in a report.
    This allowlist cannot restore API keys or other removed authentication fields.
    """
    result = dict(sanitized)
    for name in simulator_proxy_names(source):
        result[name] = source[name]
    return result


def sanitized_environment(source, root):
    result = dict(source)
    private_names = {
        "HF_TOKEN", "HF_TOKEN_PATH", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
        "HF_API_TOKEN", "ARK_API_KEY", "ARK_API_KEY_BACKUP", "SAFEHARNESS_KEY_FILE",
        "SAFEHARNESS_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY",
    }
    proxy_names = set(simulator_proxy_names(source))
    for name in list(result):
        normalized = name.upper()
        for prefix in ("SINGULARITYENV_", "APPTAINERENV_"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        if (name in proxy_names or normalized in private_names or normalized.startswith("OPENAI_")
                or normalized.startswith("AZURE_OPENAI_")
                or normalized.endswith("_API_KEY") or normalized.endswith("_ACCESS_TOKEN")):
            result.pop(name)
    result.update({
        "HF_HOME": str(root / "qwen" / "runtime-hf"),
        "HF_TOKEN_PATH": "/dev/null",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
        "VLLM_NO_USAGE_STATS": "1", "PYTHONUNBUFFERED": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8",
        "NO_PROXY": "127.0.0.1,localhost,::1", "no_proxy": "127.0.0.1,localhost,::1",
    })
    return result


def unused_loopback_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def resolve_gpu_uuids(csv_text, devices):
    rows = [{key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(csv_text.splitlines())]
    resolved = []
    for device in devices:
        matches = [row for row in rows if row.get("index") == device or row.get("uuid") == device]
        if len(matches) != 1 or "L40S" not in matches[0].get("name", ""):
            raise RuntimeError("allocated CUDA device cannot be uniquely mapped to an L40S UUID")
        resolved.append(matches[0])
    if len({row["uuid"] for row in resolved}) != 4:
        raise RuntimeError("allocated devices resolve to duplicate GPU UUIDs")
    return resolved


def server_command(python, snapshot, port):
    return [str(python), "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(snapshot), "--served-model-name", MODEL,
            "--dtype", "bfloat16", "--tensor-parallel-size", "2",
            "--max-num-seqs", "1", "--max-model-len", "139264",
            "--max-num-batched-tokens", "8192", "--enable-chunked-prefill",
            "--limit-mm-per-prompt", '{"image":{"count":24,"width":640,"height":480},"video":0}',
            "--reasoning-parser", "qwen3", "--host", "127.0.0.1", "--port", str(port)]


def suite_launcher(integration, prior_run_id=None):
    if not prior_run_id:
        return integration / "run_qwen_suite.py", []
    from qwen_remaining_audit import PRIOR_RUN_ID
    if prior_run_id != PRIOR_RUN_ID:
        raise RuntimeError("continuation is restricted to the reviewed original run")
    return integration / "run_qwen_remaining.py", ["--prior-run-id", prior_run_id]


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("local readiness redirect rejected")


def local_json(opener, url):
    with opener.open(url, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError("local readiness returned non-200 status")
        data = response.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise RuntimeError("oversized local readiness response")
    return json.loads(data)


def wait_ready(process, base_url, deadline, record):
    opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
    next_notice = time.monotonic() + 30
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before readiness (status {process.returncode})")
        try:
            models = local_json(opener, base_url + "/models")
            if not any(item.get("id") == MODEL for item in models.get("data", [])):
                raise RuntimeError("local server did not advertise the expected model")
            if process.poll() is not None:
                raise RuntimeError("vLLM exited while readiness was checked")
            return models
        except Exception as exc:
            last_error = type(exc).__name__
        if time.monotonic() >= next_notice:
            record("server_starting", last_readiness_error_type=last_error)
            next_notice = time.monotonic() + 30
        time.sleep(1)
    raise TimeoutError(f"vLLM readiness deadline exceeded ({last_error})")


def stop_process_group(process, grace=20):
    """Kill descendants even when the direct server process already exited."""
    details = {"pid": process.pid, "term_sent": False, "kill_sent": False}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        details["term_sent"] = True
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(.1)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            details["kill_sent"] = True
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        details["direct_process_wait_timed_out"] = True
    details["returncode"] = process.poll()
    return details


def wait_child(process, server, deadline):
    while True:
        status = process.poll()
        if status is not None:
            return status
        if server.poll() is not None:
            raise RuntimeError("vLLM stopped while an interface smoke or episode was running")
        if time.monotonic() >= deadline:
            raise TimeoutError("phase or overall job deadline exceeded; no automatic retry")
        time.sleep(.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"qwen3_[0-9]+", args.run_id):
        raise ValueError("run ID must be qwen3_<Slurm job ID>")
    if args.run_id != "qwen3_" + os.environ.get("SLURM_JOB_ID", ""):
        raise ValueError("run ID must match this Slurm allocation")
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = args.root.resolve(strict=True)
    integration = root / "qwen_integration"
    python = root / "qwen" / ".venv-runtime" / "bin" / "python"
    prior_run_id = os.environ.get("ASPIRE_PRIOR_RUN_ID")
    launcher, continuation_args = suite_launcher(integration, prior_run_id)
    smoke = integration / "qwen_model_smoke.py"
    for path in (python, launcher, smoke):
        if not path.is_file():
            raise RuntimeError(f"required prepared runtime file is missing: {path}")
    devices = allocated_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    cpu_assignment = cpu_partition(os.environ.get("ASPIRE_SPLIT_CPU") == "1")
    manifest_path = root / "qwen" / "model_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("repo") != MODEL or not manifest.get("revision"):
        raise RuntimeError("model manifest is not a pinned Qwen3.8-27B snapshot")
    snapshot = Path(manifest["snapshot_path"]).resolve(strict=True)
    if not (snapshot / "config.json").is_file() or not list(snapshot.glob("*.safetensors")):
        raise RuntimeError("prepared local model snapshot is incomplete")
    output = root / "qwen" / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    env = sanitized_environment(os.environ, root)
    env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
    env["ASPIRE_SIM_GPU"] = devices[0]
    port = unused_loopback_port()
    base_url = f"http://127.0.0.1:{port}/v1"
    env["ASPIRE_LOCAL_BASE_URL"] = base_url
    server_env = dict(env, CUDA_VISIBLE_DEVICES=",".join(devices[1:3]))
    command = with_cpu_partition(server_command(python, snapshot, port), cpu_assignment, "server")
    started = time.monotonic()
    overall_deadline = started + 12300
    report = {
        "schema_version": 1, "run_id": args.run_id, "started_utc": utc_now(),
        "host": socket.gethostname(), "model": MODEL,
        "model_revision": manifest["revision"], "snapshot_path": str(snapshot),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "allocated_devices": devices, "server_devices": devices[1:3],
        "simulator_device": devices[0], "unused_allocated_device": devices[3],
        "cpu_partition": cpu_assignment,
        "environment": {key: env[key] for key in ENV_REPORT_KEYS if key in env},
        "server_environment": {key: server_env[key] for key in ENV_REPORT_KEYS if key in server_env},
        "server_command": command, "complete": False, "timings_seconds": {},
        "interface_smoke_consumes_episode": False, "automatic_retries": False,
        "image_profile": {"count": 24, "width": 640, "height": 480,
                          "scope": "dummy profiling dimensions; no image processor resize override"},
        "deadline_seconds": {"overall": 12300, "startup": 1800, "smoke": 900, "suite": 10860},
    }
    if prior_run_id:
        report["continuation"] = {"prior_run_id": prior_run_id, "allowed_tasks": ["align_blocks"],
            "claim_path": str(root / "results" / prior_run_id / "continuation_claim.json")}
    write_json(output / "server_suite.json", report)
    processes = []
    server = None
    active_phase = "preflight"
    phase_started = started

    def record(event, **fields):
        row = {"event": event, "utc": utc_now(), "elapsed_seconds": time.monotonic() - started, **fields}
        with (output / "server_events.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(row, sort_keys=True), flush=True)

    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}; stopping owned processes")

    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    exit_code = 1
    try:
        with (output / "gpus.csv").open("w") as handle:
            subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                            "--format=csv"], env=env, stdout=handle, stderr=subprocess.STDOUT,
                           check=True, timeout=30)
        gpu_mapping = resolve_gpu_uuids((output / "gpus.csv").read_text(), devices)
        env["ASPIRE_SIM_GPU_UUID"] = gpu_mapping[0]["uuid"]
        server_env["ASPIRE_SIM_GPU_UUID"] = gpu_mapping[0]["uuid"]
        report["gpu_mapping"] = gpu_mapping
        report["simulator_gpu_uuid"] = gpu_mapping[0]["uuid"]
        report["server_gpu_uuids"] = [row["uuid"] for row in gpu_mapping[1:3]]
        report["environment"]["ASPIRE_SIM_GPU_UUID"] = gpu_mapping[0]["uuid"]
        report["server_environment"]["ASPIRE_SIM_GPU_UUID"] = gpu_mapping[0]["uuid"]
        versions_code = ("import json,importlib.metadata as m; "
                         "print(json.dumps({p:m.version(p) for p in "
                         "['vllm','torch','transformers','flashinfer-python','xgrammar']}))")
        versions_result = subprocess.run([str(python), "-c", versions_code], env=env,
                                        capture_output=True, text=True, check=True, timeout=30)
        report["package_versions"] = json.loads(versions_result.stdout)
        if any(report["package_versions"][key].split("+")[0] != value for key, value in
               {"vllm": "0.32.0", "torch": "2.13.0", "transformers": "5.16.1"}.items()):
            raise RuntimeError("prepared inference package versions differ from the reviewed pins")
        source_paths = [Path(__file__), launcher, smoke, integration / "local_model_driver.py"]
        if prior_run_id:
            source_paths += [integration / "qwen_remaining_audit.py", integration / "qwen_remaining_driver.py"]
        report["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
        write_json(output / "server_suite.json", report)
        began = time.monotonic()
        active_phase, phase_started = "server_startup", began
        with (output / "server.log").open("w") as handle:
            server = subprocess.Popen(command, env=server_env, cwd=root, stdout=handle,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(("server", server))
        record("server_process_started", pid=server.pid)
        report["advertised_models"] = wait_ready(server, base_url, min(began + 1800, overall_deadline), record)
        report["timings_seconds"]["server_startup"] = time.monotonic() - began
        record("server_ready", startup_seconds=report["timings_seconds"]["server_startup"])
        for name, phase_command, timeout in (
            ("smoke", [str(python), str(smoke), "--base-url", base_url,
                       "--output", str(output / "smoke" / "model_smoke.json")], 900),
            ("suite", ["/usr/bin/python3.12", str(launcher), "--run-id", args.run_id, *continuation_args], 10860),
        ):
            phase_command = with_cpu_partition(phase_command, cpu_assignment, name)
            # Isaac's first scene reset can need the cluster's proxy for assets.
            # The model server and isolated model smoke retain the sanitized env;
            # the suite's local HTTP client also explicitly disables proxy use.
            phase_env = restore_simulator_proxies(env, os.environ) if name == "suite" else env
            began = time.monotonic()
            active_phase, phase_started = name, began
            report[name + "_command"] = phase_command
            if name == "suite":
                report["simulator_proxy_environment"] = {
                    "restored_variable_names": simulator_proxy_names(os.environ),
                    "values_recorded": False, "scope": "suite launcher and simulator children",
                }
            write_json(output / "server_suite.json", report)
            with (output / ("smoke.log" if name == "smoke" else "eval_launcher.log")).open("w") as handle:
                process = subprocess.Popen(phase_command, cwd=root.parent, env=phase_env,
                                           stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((name, process))
            record(name + "_started", pid=process.pid)
            status = wait_child(process, server, min(began + timeout, overall_deadline))
            report["timings_seconds"][name] = time.monotonic() - began
            report[name + "_exit_code"] = status
            record(name + "_finished", exit_code=status, wall_seconds=report["timings_seconds"][name])
            if name == "suite" and prior_run_id:
                claim_path = Path(report["continuation"]["claim_path"])
                if claim_path.is_file():
                    claim_bytes = claim_path.read_bytes()
                    if json.loads(claim_bytes).get("continuation_run_id") == args.run_id:
                        report["continuation"]["claim_sha256"] = hashlib.sha256(claim_bytes).hexdigest()
            if status != 0:
                raise RuntimeError(name + " failed; no episode retry or alternate model fallback")
            evidence_path = (output / "smoke" / "model_smoke.json" if name == "smoke"
                             else root / "results" / args.run_id / "launcher_summary.json")
            evidence = json.loads(evidence_path.read_text())
            if (name == "smoke" and evidence.get("ok") is not True) or (
                    name == "suite" and evidence.get("complete") is not True):
                raise RuntimeError(name + " exited zero without a successful completion report")
            report[name + "_report"] = str(evidence_path)
            report[name + "_report_sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        report["complete"] = True
        exit_code = 0
    except BaseException as exc:
        report["timings_seconds"].setdefault(active_phase, time.monotonic() - phase_started)
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        record("server_suite_failed", **report["error"])
    finally:
        # A second TERM must not interrupt cleanup and leave vLLM workers alive.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        began = time.monotonic()
        report["cleanup"] = []
        for name, process in reversed(processes):
            try:
                # The suite owns simulator/controller sessions and needs time
                # for their own TERM/KILL cleanup before its parent is killed.
                report["cleanup"].append({"name": name,
                    **stop_process_group(process, grace=90 if name == "suite" else 20)})
            except Exception as exc:
                report["cleanup"].append({"name": name, "error_type": type(exc).__name__})
                report["complete"] = False
                exit_code = 1
        report["timings_seconds"]["cleanup"] = time.monotonic() - began
        report["timings_seconds"]["total"] = time.monotonic() - started
        report["finished_utc"] = utc_now()
        write_json(output / "server_suite.json", report)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
