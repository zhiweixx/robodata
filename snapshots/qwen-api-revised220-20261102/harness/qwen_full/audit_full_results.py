#!/usr/bin/env python3
"""Read-only full-campaign audit; no simulator, model, scheduler or network calls.

Print JSON to stdout. Native results own scoring; logs only measure consumed
resources and interrupted attempts. An unfinished allocation's age is not proof
that its job is running. No ETA is inferred from partial results.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import filecmp
import hashlib
import json
import math
from pathlib import Path
import re

from full_supervisor import MODEL, path_host, validate_group

CAMERAS = ("head", "left_wrist", "right_wrist")
VIDEO = re.compile(r"episode_([0-9]{7})_cam_(head|left_wrist|right_wrist)_(success|fail)\.mp4")
OFFICIAL = {"groups": 162, "task_configurations": 54, "families": 42,
            "seeds": [0, 1, 2], "target_scored": 6300, "candidate_layouts": 8070}
UNCAPPED_CONFIG = {"max_iterations": None, "max_input_tokens": None,
                   "independent_max_output_tokens": None,
                   "absolute_generation_timeout_s": None, "expected_context_length": 262144}


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite_number(value, name, *, integer=False):
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"invalid nonnegative {name}")
    return value


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def issue(errors, path, message):
    errors.append({"path": str(path), "message": str(message)})


def native_details(native, group):
    """Validate partial results with the same identity/count rules as completion."""
    details = native["details"]
    if not isinstance(details, dict) or set(details) != {str(i) for i in range(len(details))}:
        raise ValueError("noncanonical native episode indices")
    if (type(native.get("eval_time")) is not int or native["eval_time"] != len(details)
            or len(details) > group["target_episodes"]):
        raise ValueError("native scored count does not match detail mapping/target")
    layout_ids = []
    for row in details.values():
        layout = row["layout_id"]
        if type(layout) is not int or layout not in group["candidate_layout_ids"]:
            raise ValueError("native layout outside authorized candidates")
        if type(row["success"]) is not bool:
            raise ValueError("invalid native success value")
        score = finite_number(row["score"], "native score")
        if score > 1:
            raise ValueError("native score exceeds one")
        layout_ids.append(layout)
    if len(set(layout_ids)) != len(layout_ids):
        raise ValueError("duplicate native scored layout")
    return details


def video_evidence(native_dir, details):
    good, bad = set(), []
    for path in sorted(native_dir.glob("*.mp4")):
        match = VIDEO.fullmatch(path.name)
        if match is None or path.stat().st_size == 0:
            bad.append(path.name)
            continue
        index, camera, outcome = str(int(match[1])), match[2], match[3]
        key = (index, camera)
        if (index not in details or key in good
                or outcome != ("success" if details[index]["success"] else "fail")):
            bad.append(path.name)
            continue
        good.add(key)
    verified = sum(all((index, camera) in good for camera in CAMERAS) for index in details)
    return {"verified_scored_episodes": verified, "matching_nonempty_videos": len(good),
            "invalid_or_unexpected_videos": bad}


def expected_result_path(simroot, campaign, group):
    return (simroot / "RoboDojo/eval_result/RoboDojo" / group["task"]
            / "ASPIRE_Pilot/arx_x5" / f"{group['eval_seed']}_aspire_qwen_full"
            / f"{campaign}_{group['group_id']}" / "_result.json")


def audit_group(campaign_dir, simroot, group, errors, *, progress_only=False):
    campaign, gid = campaign_dir.name, group["group_id"]
    directory = campaign_dir / "groups" / gid
    report_path = directory / "group_report.json"
    if progress_only:
        snapshots = [p for p in (report_path, directory / "group_progress.json") if p.is_file()]
        if snapshots:
            report_path = max(snapshots, key=lambda path: path.stat().st_mtime_ns)
    result_path = expected_result_path(simroot, campaign, group)
    summary = {"group_id": gid, "task": group["task"], "category": group["dimension"],
               "family": group["family"], "seed": group["eval_seed"],
               "target": group["target_episodes"], "state": "not_started",
               "native_complete_verified": False, "scored": 0, "success": 0,
               "score_sum": 0.0, "episode_reports_verified": 0,
               "video_evidence": {"verified_scored_episodes": 0, "matching_nonempty_videos": 0,
                                  "invalid_or_unexpected_videos": []},
               "attempts": Counter(), "native_result_path": str(result_path)}
    report, details = None, {}
    try:
        if report_path.exists():
            report = read_json(report_path)
            if report.get("mode") != "evaluation":
                summary["state"] = "excluded_non_evaluation"
                return summary
            # Failed initialization may publish a report before native identity exists.
            if "task" not in report and report.get("scored") == 0:
                summary["state"] = "unscored"
                report = None
            else:
                if (report.get("task") != group["task"]
                        or type(report.get("eval_seed")) is not int
                        or report["eval_seed"] != group["eval_seed"]
                        or report.get("native_run_id") != f"{campaign}_{gid}"
                        or report.get("target") != group["target_episodes"]):
                    raise ValueError("group report task, seed, target or campaign identity mismatch")
                result_path = path_host(report["native_result_path"], simroot)
                if result_path != expected_result_path(simroot, campaign, group):
                    raise ValueError("native result path differs from pinned campaign/group location")
                summary["reported_scored"] = report.get("scored")
                summary["reported_complete"] = report.get("complete") is True
                summary["reported_error"] = report.get("error")
                summary["reported_native_unstable"] = report.get("unstable")
                if report.get("complete") is True and not progress_only:
                    try:
                        validated = validate_group(report_path, group, simroot,
                                                   expected_campaign=campaign)
                        summary["native_complete_verified"] = True
                        summary["native_result_sha256"] = validated["native_result_sha256"]
                        summary["group_report_sha256"] = validated["group_report_sha256"]
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                        issue(errors, report_path, f"complete-group validation failed: {exc}")
        if result_path.exists():
            details = native_details(read_json(result_path), group)
            summary.update(scored=len(details), success=sum(d["success"] for d in details.values()),
                           score_sum=sum(d["score"] for d in details.values()))
            if not progress_only:
                summary["video_evidence"] = video_evidence(result_path.parent, details)
            summary["state"] = ("complete" if summary["native_complete_verified"] else
                                "scores_complete_unverified" if len(details) == group["target_episodes"] else
                                "partial" if details else "unscored")
            if report is not None and report.get("scored") != len(details):
                issue(errors, report_path, "report/native counts differ; native snapshot used")
        elif report is not None and report.get("scored", 0):
            raise ValueError("report claims scores but native result file is missing")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        issue(errors, report_path, exc)
        summary["state"] = "invalid_evidence"
        summary["native_complete_verified"] = False
        summary.update(scored=0, success=0, score_sum=0.0)
        details = {}

    if progress_only:
        # Never glob episode/video trees or read usage for a monitoring snapshot.
        # Absence of a JSON snapshot does not prove the group has not started.
        if summary["state"] == "not_started":
            summary["state"] = "no_native_score_snapshot"
        for key in ("attempts", "episode_reports_verified", "video_evidence"):
            summary.pop(key, None)
        summary["remaining_to_target"] = group["target_episodes"] - summary["scored"]
        return summary

    verified_ids = set()
    scored_ids = {row["layout_id"] for row in details.values()}
    for attempt_dir in sorted(directory.glob("episodes/*/attempt_*")):
        if not attempt_dir.is_dir():
            continue
        summary["attempts"]["total"] += 1
        attempt_report = attempt_dir / "episode_report.json"
        if not attempt_report.exists():
            summary["attempts"]["without_final_report"] += 1
            continue
        try:
            episode = read_json(attempt_report)
            layout = episode["layout_id"]
            if (type(layout) is not int or layout not in group["candidate_layout_ids"]
                    or episode.get("episode_id") != f"{gid}_l{layout}"
                    or attempt_dir.parent.name != episode["episode_id"]
                    or episode.get("task") != group["task"]
                    or type(episode.get("eval_seed")) is not int
                    or episode["eval_seed"] != group["eval_seed"]):
                raise ValueError("episode report identity mismatch")
            state = episode.get("status", "unknown")
            summary["attempts"]["reported_" + state] += 1
            if episode.get("native_score_committed") is True:
                if (layout not in scored_ids or state != "completed"
                        or episode.get("step_limit") != group["step_limit"]):
                    raise ValueError("committed episode report lacks matching native score/budget")
                limits = episode.get("limits", {})
                if ("program_count" not in limits or limits["program_count"] is not None
                        or "program_wall_seconds" not in limits or limits["program_wall_seconds"] is not None
                        or limits.get("output_tokens") != "remaining native model context"
                        or limits.get("action_budget") != "native task config"):
                    raise ValueError("committed episode report does not verify uncapped native-budget protocol")
                if layout in verified_ids:
                    raise ValueError("more than one committed attempt for a native scored layout")
                verified_ids.add(layout)
            else:
                summary["attempts"]["without_committed_score_receipt"] += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issue(errors, attempt_report, exc)
            summary["attempts"]["invalid_reports"] += 1
    summary["episode_reports_verified"] = len(verified_ids)
    if summary["state"] == "not_started" and summary["attempts"]["total"]:
        summary["state"] = "unscored"
    summary["remaining_to_target"] = group["target_episodes"] - summary["scored"]
    return summary


def jsonl_rows(path, errors):
    # Stream the ledger: image hashes/program bodies need not be retained.
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected JSON object")
                yield row
            except (ValueError, TypeError) as exc:
                issue(errors, path, f"line {line_number}: {exc}; incomplete line excluded")


def empty_usage():
    return {"ledgers": 0, "model_calls_started": 0, "model_calls_with_usage": 0,
            "prompt_tokens": 0, "completion_tokens_including_reasoning": 0,
            "generation_wall_seconds": 0.0, "tokenization_wall_seconds": 0.0,
            "calls_without_final_token_usage": 0, "transport_failure_calls": 0,
            "explicit_infrastructure_interrupted_attempts": 0,
            "native_unstable_abandoned_attempts": 0, "model_episode_failure_attempts": 0,
            "api_cost_usd": 0}


def audit_usage(path, allowed_ids, seeds, errors):
    totals = empty_usage()
    starts, usages, runtime, tokenizations = set(), {}, {}, {}
    infrastructure, unstable, model_failures, transport_failures = set(), set(), set(), set()
    model, initialized = None, False
    try:
        for row in jsonl_rows(path, errors):
            event = row.get("event")
            if event == "suite_initialized":
                config = row["config"]
                if (initialized or config.get("purpose") != "aspire_robodojo_full_local_qwen"
                        or config.get("model") != MODEL
                        or set(config.get("episode_ids", [])) != set(allowed_ids)
                        or config.get("episode_seeds") != seeds
                        or config.get("remote_api_calls") is not False
                        or any(key not in config or config[key] != value
                               for key, value in UNCAPPED_CONFIG.items())):
                    raise ValueError("usage ledger protocol/model/episode identity mismatch")
                initialized, model = True, config["model"]
                continue
            if not initialized:
                raise ValueError("usage events precede suite_initialized")
            if "episode_id" not in row:
                continue
            episode_id = row["episode_id"]
            if episode_id not in allowed_ids:
                raise ValueError("usage event outside authorized episodes")
            attempt = finite_number(row.get("attempt", 1), "attempt", integer=True)
            if attempt == 0:
                raise ValueError("attempt numbers start at one")
            attempt_key = (episode_id, attempt)
            key = (*attempt_key, row.get("iteration"))
            if event == "generation_started":
                if key in starts:
                    raise ValueError("duplicate generation_started request")
                starts.add(key)
            elif event == "generation_usage":
                if key in usages:
                    raise ValueError("duplicate generation_usage request")
                usage = row["usage"]
                prompt = finite_number(usage["prompt_tokens"], "prompt tokens", integer=True)
                completion = finite_number(usage["completion_tokens"], "completion tokens", integer=True)
                if usage.get("api_cost_usd") != 0:
                    raise ValueError("unexpected nonzero/missing API cost")
                usages[key] = (prompt, completion)
                runtime.setdefault(key, finite_number(usage["generation_wall_s"], "generation time"))
            elif event in {"raw_response", "generation_transport_failure"}:
                runtime[key] = finite_number(row["generation_wall_s"], "generation time")
                if event == "generation_transport_failure":
                    transport_failures.add(key)
                    infrastructure.add(attempt_key)
            elif event == "input_token_count":
                if key in tokenizations:
                    raise ValueError("duplicate tokenization request")
                tokenizations[key] = finite_number(row["wall_s"], "tokenization time")
            elif event == "infrastructure_stopped":
                infrastructure.add(attempt_key)
            elif event == "unscored_attempt_abandoned":
                if row.get("reason") == "infrastructure_interrupted":
                    infrastructure.add(attempt_key)
                elif row.get("reason") == "native_unstable":
                    unstable.add(attempt_key)
            elif event == "model_episode_failed":
                model_failures.add(attempt_key)
        if not initialized:
            raise ValueError("ledger has no initialization record")
        if usages.keys() - starts:
            raise ValueError("generation usage has no matching request start")
        totals.update(ledgers=1, model_calls_started=len(starts), model_calls_with_usage=len(usages),
                      prompt_tokens=sum(value[0] for value in usages.values()),
                      completion_tokens_including_reasoning=sum(value[1] for value in usages.values()),
                      generation_wall_seconds=sum(runtime.values()),
                      tokenization_wall_seconds=sum(tokenizations.values()),
                      calls_without_final_token_usage=len(starts - usages.keys()),
                      transport_failure_calls=len(transport_failures),
                      explicit_infrastructure_interrupted_attempts=len(infrastructure),
                      native_unstable_abandoned_attempts=len(unstable),
                      model_episode_failure_attempts=len(model_failures))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issue(errors, path, exc)
        # No partial sum from an untrusted ledger: identity may be wrong.
        return None
    return {"model": model, **totals}


def add_usage(destination, source):
    for key in empty_usage():
        destination[key] += source[key]


def allocation_usage_ledgers(directory, errors):
    """Prefer scored/ paths; legacy paths remain valid for older allocations.

    A worker/group has one controller ledger per allocation. Migration copies,
    symlinks and hardlinks must not multiply usage. Conflicting copies remain an
    explicit audit error, rather than silently counting both as separate work.
    """
    paths, aliases, by_group, by_file = [], [], {}, {}
    for base in (directory / "scored", directory):
        for path in sorted(base.glob("worker*/*/usage.jsonl")):
            try:
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                key = (path.parent.parent.name, path.parent.name)
                prior = by_file.get(identity) or by_group.get(key)
                if prior is not None:
                    same_file = identity in by_file
                    same_content = same_file or filecmp.cmp(prior, path, shallow=False)
                    aliases.append({"included": str(prior), "excluded": str(path),
                                    "reason": "same_file" if same_file else
                                              "identical_legacy_copy" if same_content else
                                              "conflicting_legacy_copy"})
                    if not same_content:
                        issue(errors, path, "scored and legacy ledgers conflict for the same allocation/worker/group; legacy excluded to prevent double counting")
                    continue
                by_group[key], by_file[identity] = path, path
                paths.append(path)
            except OSError as exc:
                issue(errors, path, exc)
    return paths, aliases


def audit_allocations(campaign_dir, groups, now, errors):
    allocations, usage = [], {scope: defaultdict(empty_usage) for scope in ("evaluation", "model_smoke")}
    interruption_events = Counter()
    for directory in sorted((campaign_dir / "allocations").glob("*")):
        if not directory.is_dir():
            continue
        allocation_path = directory / "allocation.json"
        summary = {"allocation": directory.name, "state": "record_missing"}
        try:
            allocation = read_json(allocation_path)
            if allocation.get("campaign") != campaign_dir.name or str(allocation.get("allocation")) != directory.name:
                raise ValueError("allocation identity mismatch")
            start = timestamp(allocation["started_utc"])
            finish = timestamp(allocation["finished_utc"]) if allocation.get("finished_utc") else None
            end = finish or now
            if end < start:
                raise ValueError("allocation end precedes start")
            summary.update(mode=allocation.get("mode"), started_utc=start.isoformat(),
                           finished_utc=finish.isoformat() if finish else None,
                           elapsed_seconds=(end - start).total_seconds(),
                           state="closed" if finish else "no_finish_record",
                           supervisor_complete=allocation.get("complete"), error=allocation.get("error"))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issue(errors, allocation_path, exc)
        events_path = directory / "events.jsonl"
        if events_path.exists():
            for row in jsonl_rows(events_path, errors):
                if row.get("event") in {"worker_interrupted", "allocation_signal", "allocation_incomplete"}:
                    interruption_events[row["event"]] += 1
        allocations.append(summary)
        ledgers, aliases = allocation_usage_ledgers(directory, errors)
        if aliases:
            summary["usage_ledger_aliases_excluded"] = aliases
        for path in ledgers:
            group = groups.get(path.parent.name)
            if group is None:
                issue(errors, path, "usage ledger does not belong to manifest group; excluded")
                continue
            result = audit_usage(path, set(group["episode_ids"]),
                                 {eid: group["eval_seed"] for eid in group["episode_ids"]}, errors)
            if result:
                add_usage(usage["evaluation"][result["model"]], result)
        for path in sorted((directory / "model_smoke").glob("*.usage.jsonl")):
            result = audit_usage(path, {"interface_smoke"}, {"interface_smoke": 0}, errors)
            if result:
                add_usage(usage["model_smoke"][result["model"]], result)
    output = {}
    for scope, models in usage.items():
        total = empty_usage()
        for measured in models.values():
            add_usage(total, measured)
        output[scope] = {"by_model": dict(models), "total": total}
    output["all_recorded_model_work"] = empty_usage()
    for scope in usage:
        add_usage(output["all_recorded_model_work"], output[scope]["total"])
    return allocations, output, dict(interruption_events)


def rollup(groups, fields):
    totals = {}
    for group in groups:
        key = tuple(group[field] for field in fields)
        row = totals.setdefault(key, {**dict(zip(fields, key)), "target": 0, "scored": 0,
                                     "success": 0, "score_sum": 0.0, "native_complete_groups": 0})
        for field in ("target", "scored", "success", "score_sum"):
            row[field] += group[field]
        row["native_complete_groups"] += group["native_complete_verified"]
    for row in totals.values():
        row["success_rate"] = row["success"] / row["scored"] if row["scored"] else None
        row["mean_score"] = row["score_sum"] / row["scored"] if row["scored"] else None
        row["score_percent"] = 100 * row["mean_score"] if row["scored"] else None
    return list(totals.values())


def audit_campaign(campaign_dir, *, simroot=None, now=None, include_details=False, progress_only=False):
    campaign_dir = Path(campaign_dir).resolve(strict=True)
    simroot = Path(simroot).resolve(strict=True) if simroot else campaign_dir.parents[2]
    now = now or datetime.now(timezone.utc)
    manifest_path = campaign_dir / "manifest.json"
    manifest = read_json(manifest_path)
    groups = manifest["groups"]
    group_map = {group["group_id"]: group for group in groups}
    if len(group_map) != len(groups) or len({(g["task"], g["eval_seed"]) for g in groups}) != len(groups):
        raise ValueError("duplicate manifest group or task/seed")
    episode_ids = [eid for group in groups for eid in group["episode_ids"]]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("duplicate manifest candidate episode identity")
    budgets = {}
    for group in groups:
        gid, task, seed = group["group_id"], group["task"], group["eval_seed"]
        candidates = group["candidate_layout_ids"]
        if (type(seed) is not int or seed not in (0, 1, 2) or gid != f"{task}_s{seed}"
                or group["family"] != task.removesuffix("_random")
                or any(type(layout) is not int or layout < 0 for layout in candidates)
                or len(candidates) != len(set(candidates))
                or group["episode_ids"] != [f"{gid}_l{layout}" for layout in candidates]):
            raise ValueError("invalid manifest group/candidate identity")
        target = finite_number(group["target_episodes"], "target episodes", integer=True)
        budget = finite_number(group["step_limit"], "native step limit", integer=True)
        if not target or target > len(candidates) or not budget or budgets.setdefault(task, budget) != budget:
            raise ValueError("invalid/inconsistent native target or action budget")
    observed = {"groups": len(groups), "task_configurations": len({g["task"] for g in groups}),
                "families": len({g["family"] for g in groups}),
                "seeds": sorted({g["eval_seed"] for g in groups}),
                "target_scored": sum(g["target_episodes"] for g in groups),
                "candidate_layouts": len(episode_ids)}
    if (manifest["group_count"] != observed["groups"]
            or manifest["total_target_episodes"] != observed["target_scored"]
            or manifest["total_candidate_episodes"] != observed["candidate_layouts"]):
        raise ValueError("manifest aggregate counts disagree with groups")
    errors = []
    summaries = [audit_group(campaign_dir, simroot, group, errors, progress_only=progress_only)
                 for group in groups]
    aggregate = rollup(summaries, [])[0] if summaries else {}
    states = Counter(group["state"] for group in summaries)
    if progress_only:
        return {"schema_version": 1, "audit_mode": "progress_only",
                "campaign": campaign_dir.name, "campaign_dir": str(campaign_dir),
                "audited_utc": now.isoformat(), "read_only": True, "campaign_complete": False,
                "final_evidence_audit_required": True,
                "official_scope": {"expected": OFFICIAL, "observed": observed, "matches": observed == OFFICIAL},
                "scoring": {**aggregate,
                            "remaining_to_target": observed["target_scored"] - aggregate.get("scored", 0),
                            "unscored_candidates_are_not_failures": True,
                            "denominator": "unique validated native scored layouts only; final evidence not audited"},
                "groups": {"target": len(groups), "native_complete_verified": 0,
                           "reported_complete_claims": sum(g.get("reported_complete", False) for g in summaries),
                           "native_scores_at_target": sum(g["scored"] == g["target"] for g in summaries),
                           "states": dict(states)},
                "by_category": rollup(summaries, ["category"]),
                "by_seed": rollup(summaries, ["seed"]),
                "evidence_audited": False, "usage_audited": False,
                "scheduler_liveness_verified": False, "eta": None,
                "measurement_note": "Lightweight JSON snapshot only. Camera files, episode reports, usage ledgers and allocations are not read. Run the full audit before declaring completion.",
                "errors": errors, **({"group_details": summaries} if include_details else {})}
    allocations, usage, interruptions = audit_allocations(campaign_dir, group_map, now, errors)
    native_completed = sum(group["native_complete_verified"] for group in summaries)
    videos_verified = sum(g["video_evidence"]["verified_scored_episodes"] for g in summaries)
    episodes_verified = sum(g["episode_reports_verified"] for g in summaries)
    attempts = Counter()
    for group in summaries:
        attempts.update(group["attempts"])
    completed_scored = sum(g["scored"] for g in summaries if g["native_complete_verified"])
    complete = (observed == OFFICIAL and native_completed == OFFICIAL["groups"]
                and aggregate["scored"] == OFFICIAL["target_scored"]
                and videos_verified == episodes_verified == OFFICIAL["target_scored"]
                and not errors)
    return {"schema_version": 1, "campaign": campaign_dir.name, "campaign_dir": str(campaign_dir),
            "audited_utc": now.isoformat(), "read_only": True, "campaign_complete": complete,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "official_scope": {"expected": OFFICIAL, "observed": observed, "matches": observed == OFFICIAL},
            "scoring": {**aggregate, "scored_in_verified_complete_groups": completed_scored,
                        "scored_in_other_groups": aggregate.get("scored", 0) - completed_scored,
                        "remaining_to_target": observed["target_scored"] - aggregate.get("scored", 0),
                        "unscored_candidates_are_not_failures": True,
                        "denominator": "unique validated native scored layouts only"},
            "groups": {"target": len(groups), "native_complete_verified": native_completed,
                       "states": dict(states)},
            "official_summarizer_input": {
                "isolated_campaign_view_required": True,
                "warning": "The official summarizer selects directories by name order; a raw result tree can select old smoke runs. Use only the verified exact-campaign directories listed here.",
                "verified_native_group_directories": {
                    g["group_id"]: str(Path(g["native_result_path"]).parent)
                    for g in summaries if g["native_complete_verified"]}},
            "evidence": {"scored_episodes_with_three_matching_nonempty_videos": videos_verified,
                         "scored_episodes_with_committed_report_and_native_budget": episodes_verified},
            "by_category": rollup(summaries, ["category"]),
            "by_family": rollup(summaries, ["category", "family"]),
            "by_seed": rollup(summaries, ["seed"]),
            "by_category_family_seed": rollup(summaries, ["category", "family", "seed"]),
            "native_action_budgets": {"unit": manifest.get("action_budget_unit"),
                                      "by_task": budgets},
            "model_protocol": {**UNCAPPED_CONFIG, "api_cost_usd": 0,
                               "usage_ledgers_checked_against_this_protocol": True},
            "episode_attempts": dict(attempts), "usage": usage,
            "allocation_time": {"closed_recorded_seconds": sum(a.get("elapsed_seconds", 0) for a in allocations if a["state"] == "closed"),
                                "unfinished_record_age_seconds": sum(a.get("elapsed_seconds", 0) for a in allocations if a["state"] == "no_finish_record"),
                                "scheduler_elapsed_verified": False,
                                "note": "Supervisor start/finish timestamps; unfinished age does not establish job liveness."},
            "allocations": allocations, "infrastructure_event_counts": interruptions,
            "interruption_note": "Ledger infrastructure attempts, episode interrupted reports and allocation events overlap; do not add them or count them as policy failures.",
            "api_cost_usd": 0, "eta": None,
            "measurement_note": "Usage includes every recorded evaluation attempt, including unscored interruptions. Smoke usage is separate. Runtime is summed request wall time, not allocation elapsed. Calls lacking final usage have unknown tokens.",
            "errors": errors, **({"group_details": summaries} if include_details else {})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--simroot", type=Path, help="Host directory corresponding to container /work")
    parser.add_argument("--now-utc", help="Optional ISO timestamp for reproducible open-allocation age")
    parser.add_argument("--details", action="store_true", help="Include each group and its evidence counts")
    parser.add_argument("--compact", action="store_true", help="Print one JSON line")
    parser.add_argument("--summary-only", action="store_true", help="Omit detailed family/seed breakdowns and per-task budgets for monitoring")
    parser.add_argument("--progress-only", action="store_true", help="Read only manifest and group/native JSON snapshots; defer all evidence and usage auditing")
    args = parser.parse_args(argv)
    try:
        result = audit_campaign(args.campaign_dir, simroot=args.simroot,
                                now=timestamp(args.now_utc) if args.now_utc else None,
                                include_details=args.details, progress_only=args.progress_only)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"audit error: {exc}\n")
    if args.summary_only:
        for key in ("by_family", "by_category_family_seed", "native_action_budgets", "group_details"):
            result.pop(key, None)
        result.get("official_summarizer_input", {}).pop("verified_native_group_directories", None)
    print(json.dumps(result, indent=None if args.compact else 2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
