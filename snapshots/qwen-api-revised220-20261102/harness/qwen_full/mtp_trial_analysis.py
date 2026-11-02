#!/usr/bin/env python3
"""Read isolated BF16/MTP trial JSONs; report evidence, never select a rollout.

No inference, GPU command, network access, or production write is performed.
Only --output writes, and only to a new file. Request text/images are omitted.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import statistics

VARIANTS = ("bf16", "mtp1", "mtp3")
CONDITIONS = {
    "c1_short": ("concurrency1", 0, 1, ("short",)),
    "c1_medium": ("concurrency1", 1, 1, ("medium",)),
    "c1_long": ("concurrency1", 2, 1, ("long",)),
    "c2_short_medium": ("concurrency2", 0, 2, ("short", "medium")),
    "c2_medium_long": ("concurrency2_long", 0, 2, ("medium", "long")),
}


def read_json(path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("report JSON exceeds the 16 MiB read bound")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def ratio(a, b):
    return a / b if number(a) and number(b, positive=True) else None


def stats(values):
    values = [value for value in values if number(value)]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def problem(errors, kind, message, path=None):
    errors.append({"kind": kind, "message": message, **({"path": str(path)} if path else {})})


def gpu_memory(batches):
    peaks, totals, errors = {}, [], 0
    for batch in batches:
        samples = batch.get("gpu_memory_samples", [])
        if not isinstance(samples, list):
            errors += 1
            continue
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("returncode") != 0:
                errors += 1
                continue
            total, seen = 0.0, set()
            try:
                for row in csv.reader(io.StringIO(sample.get("csv", ""))):
                    if len(row) < 5 or row[0].strip() in seen:
                        raise ValueError("invalid GPU CSV row")
                    uuid, used = row[0].strip(), float(row[3].strip())
                    if not uuid or not number(used):
                        raise ValueError("invalid GPU memory")
                    seen.add(uuid)
                    peaks[uuid] = max(peaks.get(uuid, 0), used)
                    total += used
                if seen:
                    totals.append(total)
                else:
                    errors += 1
            except (ValueError, TypeError, IndexError):
                errors += 1
    return {"max_by_gpu_mib": peaks, "max_any_gpu_mib": max(peaks.values(), default=None),
            "max_simultaneous_sum_mib": max(totals, default=None), "sample_errors": errors,
            "scope": "recorded measured and warmup generation samples; initialization peak is unknown"}


def read_condition(variant_dir, variant, name):
    directory_name, case_group, concurrency, labels = CONDITIONS[name]
    directory = variant_dir / directory_name
    errors, signature, batches = [], [], []
    manifest_path = directory / "manifest.json"
    result = {"concurrency": concurrency, "case_labels": list(labels), "errors": errors,
              "complete": False, "full_context_verified": False, "measured_repeats": [],
              "measured_repeat_count": 0, "comparison_signature": signature}
    if not manifest_path.exists():
        problem(errors, "missing_evidence", "condition manifest missing", manifest_path)
        return result
    try:
        manifest = read_json(manifest_path)
        all_cases, files = manifest["cases"], manifest["case_files"]
        cases = [all_cases[case_group]] if concurrency == 1 else all_cases
        names = [Path(files[case_group]).stem] if concurrency == 1 else [Path(path).stem for path in files]
        args = manifest["server"]["startup_args"]
        engine = manifest["server"]["engine_config"]
        spec = re.search(r"speculative_config=(.*?), tokenizer=", engine)
        actual_spec = spec.group(1) if spec else ""
        spec_ok = (actual_spec == "None" if variant == "bf16" else
                   bool(re.search(r"\bmethod\s*=\s*['\"]mtp['\"]", actual_spec)) and bool(re.search(
                       r"num_spec(?:ulative)?_tokens\s*=\s*" + variant[-1] + r"\b", actual_spec)))
        full_context = (args.get("max_model_len") == 262144 and args.get("dtype") == "bfloat16"
                        and args.get("tensor_parallel_size") == 2 and args.get("max_num_seqs") == 2
                        and "quantization=None" in engine and spec_ok)
        result["full_context_verified"] = full_context
        if (manifest.get("purpose") != "isolated_performance_replay_not_episode"
                or manifest.get("variant") != variant or manifest.get("concurrency") != concurrency
                or manifest.get("test_only_max_output_tokens") != 1024
                or manifest.get("production_modified") is not False or manifest.get("code_execution") is not False
                or manifest.get("repeats") != 2 or manifest.get("warmups") != 1
                or names != list(labels) or len(cases) != concurrency or not full_context):
            problem(errors, "protocol", "manifest differs from the BF16/full-context/1024-token/2-repeat comparison protocol")
        for case in cases:
            source_hash = case["source"]["request_sha256"]
            if (not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash)
                    or not number(case["prompt_tokens"], positive=True)
                    or case["prompt_tokens"] >= 262144 or not number(case["image_count"], positive=True)):
                raise ValueError("invalid original case identity/token/image metadata")
            signature.append({"source_request_sha256": source_hash, "prompt_tokens": case["prompt_tokens"],
                              "image_count": case["image_count"], "output_cap": 1024})
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        problem(errors, "manifest", str(exc), manifest_path)
        return result

    repeats_seen = set()
    for path in sorted(directory.glob(f"batch_{case_group:02d}_*.json")):
        try:
            batch = read_json(path)
            repeat = batch["repeat"]
            if (type(repeat) is not int or repeat not in (0, 1, 2) or repeat in repeats_seen
                    or path.name != f"batch_{case_group:02d}_{repeat:03d}.json"
                    or batch.get("case_group") != case_group or batch.get("concurrency") != concurrency
                    or batch.get("warmup") is not (repeat == 0)):
                raise ValueError("duplicate or inconsistent batch identity")
            repeats_seen.add(repeat)
            requests = batch["requests"]
            if not isinstance(requests, list) or len(requests) != concurrency:
                raise ValueError("wrong request count")
            if (any(type(row.get("worker")) is not int for row in requests)
                    or {row.get("worker") for row in requests} != set(range(concurrency))):
                raise ValueError("duplicate or missing concurrent worker")
            if not number(batch.get("batch_wall_s"), positive=True):
                raise ValueError("invalid batch wall time")
            request_errors, schema_errors = 0, 0
            tokens, request_wall, ttfts, request_rates, caps = 0, 0.0, [], [], 0
            for row in sorted(requests, key=lambda item: item["worker"]):
                worker = row["worker"]
                expected = signature[worker]
                if (row.get("source_request_sha256") != expected["source_request_sha256"]
                        or row.get("case_source", {}).get("request_sha256") != expected["source_request_sha256"]
                        or row.get("prompt_tokens") != expected["prompt_tokens"]
                        or row.get("image_count") != expected["image_count"]
                        or row.get("requested_output_cap") != expected["output_cap"]
                        or row.get("repeat") != repeat or row.get("warmup") is not (repeat == 0)):
                    problem(errors, "request_identity", f"repeat {repeat}, worker {worker}: case/cap mismatch", path)
                if row.get("ok") is not True:
                    request_errors += 1
                    problem(errors, "transport_or_request", f"repeat {repeat}, worker {worker}: {row.get('error_type', 'request_failed')}: {row.get('error', '')}", path)
                completed_invalid = row.get("finish_reason") == "stop" and row.get("final_json_schema_valid") is not True
                if completed_invalid:
                    schema_errors += 1
                    problem(errors, "schema", f"repeat {repeat}, worker {worker}: completed invalid JSON", path)
                if row.get("finish_reason") not in ("stop", "length"):
                    problem(errors, "request_finish", f"repeat {repeat}, worker {worker}: missing or unexpected finish reason", path)
                if (row.get("generated_code_executed") is not False
                        or not number(row.get("output_tokens")) or row["output_tokens"] > 1024
                        or (row.get("finish_reason") == "length" and row.get("output_tokens") != row.get("requested_output_cap"))
                        or not number(row.get("wall_s"), positive=True)
                        or row.get("usage", {}).get("prompt_tokens") != expected["prompt_tokens"]
                        or row.get("usage", {}).get("completion_tokens") != row.get("output_tokens")):
                    problem(errors, "request_accounting", f"repeat {repeat}, worker {worker}: execution/token/time evidence invalid", path)
                if number(row.get("output_tokens")) and number(row.get("wall_s"), positive=True):
                    tokens += row["output_tokens"]
                    request_wall += row["wall_s"]
                    request_rates.append(row["output_tokens"] / row["wall_s"])
                if number(row.get("observed_ttft_s")):
                    ttfts.append(row["observed_ttft_s"])
                else:
                    ttfts.append(None)
                    problem(errors, "ttft", f"repeat {repeat}, worker {worker}: missing TTFT", path)
                caps += row.get("finish_reason") == "length"
            counter_errors = []
            if batch.get("metrics_generation_accounting_caught_up") is not True:
                counter_errors.append("generation counter did not catch up")
            speculation = batch.get("speculation", {})
            draft, accepted = speculation.get("draft_tokens"), speculation.get("accepted_tokens")
            valid_counts = number(draft) and number(accepted) and accepted <= draft
            if variant != "bf16":
                if not valid_counts or speculation.get("acceptance_counters_valid") is not True:
                    counter_errors.append("missing or invalid speculative counters")
            elif (number(draft) and draft > 0) or (number(accepted) and accepted > 0):
                counter_errors.append("BF16 baseline unexpectedly recorded speculative activity")
            for message in counter_errors:
                problem(errors, "counter", f"repeat {repeat}: {message}", path)
            rate = tokens / batch["batch_wall_s"]
            reported_rate = batch.get("aggregate_output_tokens_per_second")
            if not number(reported_rate) or not math.isclose(rate, reported_rate, rel_tol=1e-8, abs_tol=1e-8):
                problem(errors, "batch_accounting", f"repeat {repeat}: reported batch rate differs from token/time evidence", path)
            batch["_summary"] = {"repeat": repeat, "output_tokens": tokens, "batch_wall_s": batch["batch_wall_s"],
                                 "batch_tokens_per_second": rate, "request_sum_wall_s": request_wall,
                                 "request_pooled_tokens_per_second": ratio(tokens, request_wall),
                                 "request_rate_stats": stats(request_rates), "request_ttft_s": ttfts,
                                 "cap_truncated_responses": caps, "transport_or_request_errors": request_errors,
                                 "completed_invalid_json_responses": schema_errors, "counter_errors": counter_errors,
                                 "draft_tokens": draft if valid_counts else None,
                                 "accepted_tokens": accepted if valid_counts else None}
            batches.append(batch)
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            problem(errors, "batch", str(exc), path)
    if repeats_seen != {0, 1, 2}:
        problem(errors, "missing_evidence", "requires warmup 0 and measured repeats 1 and 2")
    measured = [batch["_summary"] for batch in batches if not batch["warmup"]]
    warmups = [batch["_summary"] for batch in batches if batch["warmup"]]
    drafts = [row["draft_tokens"] for row in measured]
    accepts = [row["accepted_tokens"] for row in measured]
    all_counters_known = len(measured) == 2 and all(number(v) for v in drafts + accepts)
    drafted = sum(drafts) if all_counters_known else None
    accepted = sum(accepts) if all_counters_known else None
    mtp_active = number(drafted, positive=True) and number(accepted, positive=True)
    if variant != "bf16" and not mtp_active:
        problem(errors, "mtp_activity", "measured repeats do not prove positive drafted and accepted tokens")
    tokens = sum(row["output_tokens"] for row in measured)
    wall = sum(row["batch_wall_s"] for row in measured)
    request_wall = sum(row["request_sum_wall_s"] for row in measured)
    memory = gpu_memory(batches)
    if not memory["max_by_gpu_mib"]:
        problem(errors, "gpu_memory", "no usable GPU memory samples")
    result.update(complete=not errors and len(measured) == 2,
                  measured_repeats=measured, measured_repeat_count=len(measured),
                  batch_rate_stats=stats([row["batch_tokens_per_second"] for row in measured]),
                  pooled_batch_tokens_per_second=ratio(tokens, wall),
                  request_pooled_tokens_per_second=ratio(tokens, request_wall),
                  measured_output_tokens=tokens, measured_batch_wall_s=wall,
                  measured_request_wall_s=request_wall,
                  measured_ttft_s=stats([value for row in measured for value in row["request_ttft_s"]]),
                  measured_ttft_by_case={label: stats([row["request_ttft_s"][i] for row in measured])
                                        for i, label in enumerate(labels)},
                  warmup_ttft_s=[{"repeat": row["repeat"], "request_ttft_s": row["request_ttft_s"]} for row in warmups],
                  warmup_cache_cold_verified=False,
                  measured_speculation={"draft_tokens": drafted, "accepted_tokens": accepted,
                                        "acceptance_ratio": ratio(accepted, drafted), "positive_mtp_activity": mtp_active},
                  gpu_memory=memory)
    return result


def analyze_trial(trial_dir):
    trial_dir = Path(trial_dir).resolve(strict=True)
    trial = read_json(trial_dir / "trial.json")
    trial_errors = []
    if (trial.get("purpose") != "isolated_performance_test_not_robot_evaluation"
            or trial.get("native_context") != 262144 or trial.get("target_dtype") != "bfloat16"
            or trial.get("test_only_output_cap") != 1024
            or trial.get("changes_to_evaluation_budgets") is not False
            or trial.get("simulation_running") is not False):
        trial_errors.append("trial protocol is not the isolated BF16/full-context/1024-token comparison")
    recorded_variants = trial.get("variants", {})
    if not isinstance(recorded_variants, dict):
        trial_errors.append("trial variant records are malformed")
        recorded_variants = {}
    variants = {}
    for variant in VARIANTS:
        directory = trial_dir / variant
        recorded = recorded_variants.get(variant, {})
        if not isinstance(recorded, dict):
            recorded = {}
        smoke_path = directory / "vision_schema_smoke.json"
        try:
            smoke = read_json(smoke_path)
            smoke_ok = (smoke.get("ok") is True and smoke.get("reported_colors") == ["red", "green", "blue"]
                        and smoke.get("expected_context_length") == 262144
                        and smoke.get("measured_episode") is False)
            smoke_error = None if smoke_ok else "vision/schema smoke did not pass the full-context contract"
        except (OSError, ValueError, TypeError) as exc:
            smoke_ok, smoke_error = False, str(exc)
        conditions = {name: read_condition(directory, variant, name) for name in CONDITIONS}
        reasons = list(trial_errors)
        if not smoke_ok:
            reasons.append(smoke_error)
        if recorded.get("ok") is not True or not recorded.get("finished_utc"):
            reasons.append("variant did not finish successfully")
        if not all(row["complete"] for row in conditions.values()):
            reasons.append("one or more conditions lack complete valid evidence")
        variants[variant] = {"recorded_ok": recorded.get("ok"), "finished_utc": recorded.get("finished_utc"),
                             "recorded_error": recorded.get("error"), "vision_schema_smoke_ok": smoke_ok,
                             "conditions": conditions, "eligible_for_review": not reasons,
                             "review_blockers": reasons}
    baseline = variants["bf16"]
    for variant in ("mtp1", "mtp3"):
        comparisons_ok = True
        for name, condition in variants[variant]["conditions"].items():
            base = baseline["conditions"][name]
            matched = bool(condition["comparison_signature"]) and condition["comparison_signature"] == base["comparison_signature"]
            comparable = condition["complete"] and base["complete"] and matched and baseline["eligible_for_review"]
            comparison = {"matched_original_cases": matched, "eligible_for_comparison": comparable}
            if comparable:
                comparison.update(batch_throughput_speedup=ratio(condition["pooled_batch_tokens_per_second"], base["pooled_batch_tokens_per_second"]),
                                  median_batch_rate_speedup=ratio(condition["batch_rate_stats"]["median"], base["batch_rate_stats"]["median"]),
                                  request_pooled_rate_speedup=ratio(condition["request_pooled_tokens_per_second"], base["request_pooled_tokens_per_second"]),
                                  mean_ttft_ratio=ratio(condition["measured_ttft_s"]["mean"], base["measured_ttft_s"]["mean"]))
            else:
                comparisons_ok = False
            condition["vs_bf16"] = comparison
        if not comparisons_ok:
            variants[variant]["eligible_for_review"] = False
            variants[variant]["review_blockers"].append("BF16 baseline unavailable/invalid or original comparison cases differ")
    return {"schema_version": 1, "trial_dir": str(trial_dir), "job_id": trial.get("job_id"),
            "analyzed_utc": datetime.now(timezone.utc).isoformat(), "trial_finished_utc": trial.get("finished_utc"),
            "server_runtime_versions": trial.get("server_runtime_versions"), "trial_protocol_errors": trial_errors,
            "variants": variants, "eligible_mtp_variants_for_review": [v for v in ("mtp1", "mtp3") if variants[v]["eligible_for_review"]],
            "automatic_selection": False, "selected_variant": None, "api_cost_usd": 0,
            "caveats": ["1024-token timing-only replays are not robot episodes, task success measurements, or a full-episode speedup guarantee.",
                        "Eligibility only means complete comparable evidence for human review; it does not require a speedup or authorize rollout.",
                        "Warmups are excluded from rates and acceptance totals. Warmup/first-pass TTFT is shown separately; caches were not reset, so cache-cold behavior is unverified.",
                        "Batch throughput uses summed output tokens / summed batch wall time; request-pooled rate uses summed output tokens / summed request wall times.",
                        "TTFT observes the first nonempty reasoning/text chunk; speculative decoding may stream multiple tokens per chunk.",
                        "MTP acceptance belongs to an entire concurrent batch. GPU samples can miss transient peaks. No production artifacts are modified."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        result = analyze_trial(args.trial_dir)
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        if args.output:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
        print(encoded, end="")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"trial analysis error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
