"""Isolated BF16/MTP performance replay; never runs robot programs or scores.

prepare reconstructs a historical public request and verifies its original hash.
command prints an argv list; it never starts a server. run targets a separately
launched, explicitly acknowledged benchmark server. Only the performance replay
has a finite output cap; production evaluation source and budgets are untouched.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib import parse, request

from full_model_driver import (LoopbackStreamingHTTP, MODEL, NATIVE_CONTEXT_LENGTH,
                               build_chat_requests, canonical, sha)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(canonical(value) + "\n"); f.flush(); os.fsync(f.fileno())


def restore_images(value, images):
    if isinstance(value, dict):
        if value.get("inline_image_redacted") is True:
            url = images[value["sha256_data_url"]]
            if len(url) != value["data_url_characters"]:
                raise ValueError("recorded image length mismatch")
            return url
        return {key: restore_images(item, images) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_images(item, images) for item in value]
    return value


def validate_policy_request(payload):
    """Require exact current harness schema/sampling; don't quietly alter them."""
    expected, _ = build_chat_requests(payload["messages"], seed=payload["seed"],
                                     remaining_context=payload["max_tokens"])
    if payload != expected:
        raise ValueError("recorded request differs from full driver sampling/schema/template contract")


def prepare_case(usage_jsonl, episode_root, episode_id, attempt, iteration, output):
    wanted = {"episode_id": episode_id, "attempt": attempt, "iteration": iteration}
    selected = []
    counted = []
    with Path(usage_jsonl).open() as f:
        for line in f:
            row = json.loads(line)
            if all(row.get(key) == value for key, value in wanted.items()):
                if row["event"] == "program_input":
                    selected.append(row)
                elif row["event"] == "input_token_count":
                    counted.append(row)
    if len(selected) != 1 or len(counted) != 1:
        raise ValueError("selector must identify exactly one audited request and token preflight")
    images = {}
    public_sources = []
    for path in sorted(Path(episode_root).glob("public_input_*.json")):
        document = json.loads(path.read_text())
        for image in document["public_observation"].get("images", []):
            url = image["data_url"]
            images[hashlib.sha256(url.encode()).hexdigest()] = url
        public_sources.append(str(path.resolve()))
    payload = restore_images(selected[0]["request"], images)
    if sha(payload) != selected[0]["request_sha256"]:
        raise ValueError("restored request does not match original audited SHA256")
    validate_policy_request(payload)
    count = counted[0]["count"]
    if counted[0]["max_model_len"] != NATIVE_CONTEXT_LENGTH or payload["max_tokens"] != NATIVE_CONTEXT_LENGTH - count:
        raise ValueError("source request was not a full native-context request")
    image_count = sum(part.get("type") == "image_url" for message in payload["messages"]
        if isinstance(message.get("content"), list) for part in message["content"])
    case = {"purpose": "isolated_performance_replay_not_episode", "source": {
        **wanted, "usage_jsonl": str(Path(usage_jsonl).resolve()),
        "episode_root": str(Path(episode_root).resolve()), "public_inputs": public_sources,
        "request_sha256": sha(payload)}, "prompt_tokens": count,
        "image_count": image_count, "request": payload}
    write_new(output, case)
    return {key: case[key] for key in ("purpose", "source", "prompt_tokens", "image_count")}


def server_command(python, snapshot, port, variant):
    # Importing this function starts no processes and modifies no production code.
    from full_supervisor import server_command as production_command
    command = production_command(Path(python), Path(snapshot), port)
    if variant != "bf16":
        command += ["--speculative-config", json.dumps({"method": "mtp",
                     "num_speculative_tokens": int(variant[-1])})]
    return command


def validate_server_log(path, base_url, variant):
    path = Path(path).resolve()
    if "full_results" in path.parts and "allocations" in path.parts:
        raise ValueError("production allocation server log is forbidden for benchmark replay")
    args = engine = None
    with path.open() as f:
        for line in f:
            if "non-default args: " in line:
                args = ast.literal_eval(line.split("non-default args: ", 1)[1].strip())
            if "Initializing a V1 LLM engine" in line:
                engine = line.strip()
    if args is None or engine is None:
        raise ValueError("server startup evidence is missing")
    expected = {"dtype": "bfloat16", "max_model_len": NATIVE_CONTEXT_LENGTH,
                "tensor_parallel_size": 2, "max_num_seqs": 2}
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError("server dtype/context/parallelism differs from comparison protocol")
    if args.get("port") != parse.urlsplit(base_url).port or args.get("host") != "127.0.0.1":
        raise ValueError("server log port/host does not match explicitly selected loopback server")
    if "quantization=None" not in engine:
        raise ValueError("comparison requires unquantized BF16 target")
    spec = re.search(r"speculative_config=(.*?), tokenizer=", engine)
    if spec is None:
        raise ValueError("missing speculative engine configuration")
    if variant == "bf16":
        valid = spec.group(1) == "None"
    else:
        valid = "mtp" in spec.group(1) and re.search(
            r"num_spec(?:ulative)?_tokens\s*=\s*" + variant[-1] + r"\b", spec.group(1))
    if not valid:
        raise ValueError("requested variant differs from actual engine speculative configuration")
    return {"startup_args": args, "engine_config": engine,
            "startup_log": str(path), "startup_log_sha256_at_start": hashlib.sha256(path.read_bytes()).hexdigest()}


def parse_metrics(text):
    selected = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r'(vllm:[A-Za-z0-9_:]+)(\{.*\})?\s+([-+0-9.eE]+)(?:\s+\d+)?', line)
        if match and any(word in match[1] for word in ("spec_decode", "prefix_cache", "mm_cache", "num_requests", "generation_tokens", "request_success")):
            selected[match[1] + (match[2] or "")] = float(match[3])
    return selected


def metric_delta(before, after):
    result = {key: value - before.get(key, 0) for key, value in after.items()
              if "spec_decode" in key and not key.split("{", 1)[0].endswith("_created")}
    def total(prefix):
        values = [value for key, value in result.items()
                  if key.split("{", 1)[0] in {prefix, prefix + "_total"}]
        return sum(values) if values else None
    drafted = total("vllm:spec_decode_num_draft_tokens")
    accepted = total("vllm:spec_decode_num_accepted_tokens")
    valid = drafted is not None and accepted is not None and 0 <= accepted <= drafted
    return {"raw_speculative_counter_deltas": result, "draft_tokens": drafted,
            "accepted_tokens": accepted,
            "acceptance_rate": accepted / drafted if valid and drafted else None,
            "acceptance_counters_valid": valid,
            "scope": "isolated whole batch, not attributable to individual concurrent requests"}


def generation_counter(metrics):
    values = [value for key, value in metrics.items()
              if key.split("{", 1)[0] in {"vllm:generation_tokens", "vllm:generation_tokens_total"}]
    return sum(values) if values else None


class ReplayHTTP:
    def __init__(self, base_url):
        self.local = LoopbackStreamingHTTP(base_url, socket_idle_timeout_s=None)

    def metrics(self):
        with self.local.opener.open(self.local.origin + "/metrics", timeout=30) as f:
            return parse_metrics(f.read().decode())

    def tokenize(self, payload):
        _, preflight = build_chat_requests(payload["messages"], seed=payload["seed"])
        return self.local("/tokenize", preflight)

    def generate(self, payload):
        started = time.monotonic()
        first_text = last_text = None
        content, reasoning = [], []
        usage = finish = response_model = response_id = None
        done = False
        req = request.Request(self.local.origin + "/v1/chat/completions",
            data=canonical(payload).encode(), method="POST", headers={"Content-Type": "application/json"})
        with self.local.opener.open(req, timeout=None) as f:
            data_lines = []
            for raw in f:
                line = raw.decode().rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif not line and data_lines:
                    data = "\n".join(data_lines); data_lines.clear()
                    if data == "[DONE]":
                        done = True
                        break
                    packet = json.loads(data)
                    if packet.get("error"):
                        raise RuntimeError("server emitted error: " + str(packet["error"]))
                    response_model = packet.get("model", response_model)
                    response_id = packet.get("id", response_id)
                    usage = packet.get("usage") or usage
                    for choice in packet.get("choices", []):
                        delta = choice.get("delta", {})
                        texts = [delta.get("content"), delta.get("reasoning"), delta.get("reasoning_content")]
                        if any(isinstance(text, str) and text for text in texts):
                            now = time.monotonic()
                            first_text = now if first_text is None else first_text
                            last_text = now
                        content.append(delta.get("content") or "")
                        reasoning.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
                        finish = choice.get("finish_reason") or finish
        wall = time.monotonic() - started
        if not done or usage is None or finish is None or response_model != MODEL:
            raise RuntimeError("incomplete or mismatched SSE response")
        output = usage.get("completion_tokens")
        if type(output) is not int or output < 0:
            raise RuntimeError("missing output token usage")
        final = "".join(content)
        try:
            parsed = json.loads(final)
            schema_ok = (isinstance(parsed, dict) and set(parsed) == {"code", "summary", "give_up"}
                         and isinstance(parsed["code"], str) and isinstance(parsed["summary"], str)
                         and type(parsed["give_up"]) is bool
                         and bool(parsed["code"].strip()) != parsed["give_up"])
        except ValueError:
            schema_ok = False
        return {"response_id": response_id, "usage": usage, "output_tokens": output,
                "wall_s": wall, "observed_ttft_s": None if first_text is None else first_text - started,
                "ttft_definition": "first nonempty streamed reasoning or final-text chunk; MTP may emit multiple tokens/chunk",
                "stream_text_span_s": None if first_text is None else last_text - first_text,
                "output_tokens_per_wall_second": output / wall, "finish_reason": finish,
                "benchmark_cap_reached": finish == "length", "final_json_schema_valid": schema_ok,
                "final_content": final, "reasoning_text": "".join(reasoning),
                "generated_code_executed": False}


def gpu_snapshot(gpu_ids):
    result = subprocess.run(["nvidia-smi", "--id=" + ",".join(gpu_ids),
        "--query-gpu=uuid,index,name,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    return {"unix_time": time.time(), "returncode": result.returncode,
            "csv": result.stdout.strip(), "error": result.stderr.strip()}


def runtime_versions():
    versions = {}
    for name in ("vllm", "torch", "transformers"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_batch(cases, concurrency, max_output, repeat, warmup, http, gpu_ids):
    before = http.metrics()
    memory = []
    stop_sampling = threading.Event()
    def sample():
        while not stop_sampling.is_set():
            try:
                memory.append(gpu_snapshot(gpu_ids))
            except Exception as exc:
                memory.append({"error": str(exc), "unix_time": time.time()})
            stop_sampling.wait(1)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    barrier = threading.Barrier(concurrency)
    def run_one(index):
        case = cases[index % len(cases)]
        payload = dict(case["request"])
        remaining = NATIVE_CONTEXT_LENGTH - case["verified_prompt_tokens"]
        payload["max_tokens"] = min(max_output, remaining)
        row = {"case_source": case["source"], "worker": index, "repeat": repeat,
               "warmup": warmup, "source_request_sha256": sha(case["request"]),
               "benchmark_request_sha256": sha(payload), "requested_output_cap": payload["max_tokens"],
               "prompt_tokens": case["verified_prompt_tokens"], "image_count": case["image_count"]}
        barrier.wait()
        started = time.monotonic()
        try:
            row.update(http.generate(payload))
            if row["usage"]["prompt_tokens"] != case["verified_prompt_tokens"] or row["output_tokens"] > payload["max_tokens"]:
                raise RuntimeError("generation token accounting differs from preflight/cap")
            if row["finish_reason"] == "stop" and not row["final_json_schema_valid"]:
                raise RuntimeError("completed replay violated the production JSON schema")
            row["ok"] = True
        except Exception as exc:
            row.update(ok=False, error_type=type(exc).__name__, error=str(exc), wall_s=time.monotonic()-started)
        return row
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(run_one, range(concurrency)))
        batch_wall = time.monotonic() - started
    finally:
        stop_sampling.set(); monitor.join(12)
    tokens = sum(row.get("output_tokens", 0) for row in rows)
    after = http.metrics()
    # Metrics reporting can lag the final SSE packet. Waiting here is outside
    # measured latency and never changes a model's allowed generation time.
    settle_started = time.monotonic()
    baseline_count = generation_counter(before)
    while baseline_count is not None and generation_counter(after) is not None:
        if generation_counter(after) - baseline_count >= tokens or time.monotonic() - settle_started >= 15:
            break
        time.sleep(.2)
        after = http.metrics()
    metrics_complete = (baseline_count is not None and generation_counter(after) is not None
                        and generation_counter(after) - baseline_count >= tokens)
    return {"concurrency": concurrency, "repeat": repeat, "warmup": warmup,
            "batch_wall_s": batch_wall, "aggregate_output_tokens_per_second": tokens / batch_wall,
            "requests": rows, "metrics_before": before, "metrics_after": after,
            "metrics_generation_accounting_caught_up": metrics_complete,
            "speculation": metric_delta(before, after), "gpu_memory_samples": memory}


def benchmark(case_paths, base_url, server_log, variant, output_dir, concurrency,
              max_output_tokens, repeats, warmups, gpu_ids, *, _http=None):
    if concurrency not in (1, 2) or repeats < 1 or warmups < 0 or max_output_tokens < 1:
        raise ValueError("invalid benchmark plan")
    if concurrency == 2 and len(case_paths) != 2:
        raise ValueError("concurrency2 requires exactly two recorded cases")
    LoopbackStreamingHTTP(base_url)  # Reject credentials/proxies/non-loopback URLs.
    evidence = validate_server_log(server_log, base_url, variant)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    http = _http or ReplayHTTP(base_url)
    cases = []
    for path in case_paths:
        case = json.loads(Path(path).read_text())
        validate_policy_request(case["request"])
        if sha(case["request"]) != case["source"]["request_sha256"]:
            raise ValueError("prepared request has changed")
        counted = http.tokenize(case["request"])
        if counted.get("max_model_len") != NATIVE_CONTEXT_LENGTH or counted.get("count") != case["prompt_tokens"]:
            raise ValueError("benchmark server tokenization differs from recorded request")
        case["verified_prompt_tokens"] = counted["count"]
        cases.append(case)
    if not cases or len(cases) < concurrency:
        raise ValueError("provide at least one distinct recorded case per concurrent worker")
    manifest = {"purpose": "isolated_performance_replay_not_episode", "variant": variant,
                "started_utc": datetime.now(timezone.utc).isoformat(), "base_url": base_url,
                "concurrency": concurrency, "test_only_max_output_tokens": max_output_tokens,
                "repeats": repeats, "warmups": warmups, "gpu_ids": gpu_ids,
                "server": evidence, "case_files": [str(Path(p).resolve()) for p in case_paths],
                "benchmark_client_runtime_versions": runtime_versions(),
                "benchmark_client_relevant_environment": {key: value for key, value in os.environ.items()
                    if key in {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "VLLM_USE_V2_MODEL_RUNNER",
                               "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ATTENTION_BACKEND",
                               "VLLM_DISABLE_COMPILE_CACHE", "VLLM_USE_AOT_COMPILE"}},
                "cases": [{"source": c["source"], "prompt_tokens": c["prompt_tokens"],
                           "image_count": c["image_count"]} for c in cases],
                "production_modified": False, "code_execution": False,
                "policy": "same recorded prompt/images/schema/sampling/seed; only max_tokens replaced for timing test",
                "cache_policy": "identical warmup/repeat order per freshly launched variant; caches are not reset mid-run",
                "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_new(out / "manifest.json", manifest)
    batches = []
    # At concurrency1, measure every provided case separately. At concurrency2,
    # exactly two cases run together; use multiple invocations for other pairs.
    groups = [[case] for case in cases] if concurrency == 1 else [cases]
    for group_index, group in enumerate(groups):
        for index in range(warmups + repeats):
            warmup = index < warmups
            batch = run_batch(group, concurrency, max_output_tokens, index, warmup, http, gpu_ids)
            batch["case_group"] = group_index
            write_new(out / f"batch_{group_index:02d}_{index:03d}.json", batch)
            batches.append(batch)
            if any(not row["ok"] for row in batch["requests"]):
                write_new(out / "result.json", {"ok": False, "error": "request failed; no automatic retry",
                                               "completed_batches": len(batches)})
                raise RuntimeError("isolated benchmark request failed; inspect batch artifact")
    measured = [batch for batch in batches if not batch["warmup"]]
    total_tokens = sum(row["output_tokens"] for batch in measured for row in batch["requests"])
    total_wall = sum(batch["batch_wall_s"] for batch in measured)
    result = {"ok": True, "variant": variant, "measured_batches": len(measured),
              "output_tokens": total_tokens, "batch_wall_s": total_wall,
              "aggregate_output_tokens_per_second": total_tokens / total_wall,
              "completed_json_responses": sum(row["final_json_schema_valid"] for b in measured for row in b["requests"]),
              "cap_truncated_responses": sum(row["benchmark_cap_reached"] for b in measured for row in b["requests"]),
              "note": "Timing-only bounded replays; not task successes, robot episodes, or bit-identical-output guarantees"}
    write_new(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("usage-jsonl", "episode-root", "episode-id", "output"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--attempt", type=int, required=True)
    prep.add_argument("--iteration", type=int, required=True)
    command = sub.add_parser("command")
    command.add_argument("--python", required=True); command.add_argument("--snapshot", required=True)
    command.add_argument("--port", type=int, required=True)
    command.add_argument("--variant", choices=("bf16", "mtp1", "mtp3"), required=True)
    run = sub.add_parser("run")
    run.add_argument("--case", action="append", required=True)
    run.add_argument("--base-url", required=True); run.add_argument("--isolated-server-log", required=True)
    run.add_argument("--variant", choices=("bf16", "mtp1", "mtp3"), required=True)
    run.add_argument("--output-dir", required=True); run.add_argument("--concurrency", type=int, choices=(1, 2), required=True)
    run.add_argument("--max-output-tokens", type=int, default=1024)
    run.add_argument("--repeats", type=int, default=2); run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--gpu-id", action="append", required=True)
    run.add_argument("--ack-isolated-benchmark", action="store_true", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_case(args.usage_jsonl, args.episode_root, args.episode_id, args.attempt, args.iteration, args.output)
    elif args.command == "command":
        result = server_command(args.python, args.snapshot, args.port, args.variant)
    else:
        result = benchmark(args.case, args.base_url, args.isolated_server_log, args.variant, args.output_dir,
            args.concurrency, args.max_output_tokens, args.repeats, args.warmups, args.gpu_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
