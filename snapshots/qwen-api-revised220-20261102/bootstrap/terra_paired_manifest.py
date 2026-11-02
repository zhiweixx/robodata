#!/usr/bin/env python3
"""Freeze exact Terra pairing states from the terminal, full Qwen evidence audit.

CPU-only and read-only except for an explicitly requested, new output file. This
is a state-selection artifact, not an executable model configuration. The audit
is an attestation from audit_full807_results.py; raw videos are not re-read here.
Never use its --compact stdout: --output retains the required full evidence.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

SOURCE_CAMPAIGN = "professor_api_full807_20261029"
STAGE = "non_cloth_non_fluid_non_charger_727"
PAIRED_EPISODES = 721
USER_HELD_GROUP = "stack_bowls_random_s0"
USER_HELD_SCORED = 4
USER_HELD_SHORTFALLS = {USER_HELD_GROUP: 6}
TERMINAL_ERROR = {"type": "RuntimeError", "message":
                  "full807 allocation interrupted; explicit inspected native resume required"}
DEFERRED = {"fold_clothes_s0": 10, "fold_clothes_random_s0": 10,
            "pour_by_language_s0": 20, "pour_liquid_into_cup_s0": 10,
            "pour_liquid_into_cup_random_s0": 10, "plug_in_charger_s0": 20}
HASH = re.compile(r"[0-9a-f]{64}\Z")
CAMERAS = ("head", "left_wrist", "right_wrist")
USAGE_KEYS = ("requests_started", "requests_with_usage", "requests_without_usage", "prompt_tokens",
              "completion_tokens", "summed_request_wall_seconds", "transport_failures",
              "native_context_rejections", "raw_archive_bytes")
UNKNOWN_USAGE_WARNING = "Request ended without provider usage counters; retained as unknown, not zero."


class PairingError(ValueError):
    """Evidence is incomplete, inconsistent, or outside the authorized stage."""


def require(condition, message):
    if not condition:
        raise PairingError(message)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    raw = Path(path).read_bytes()
    def invalid_number(value):
        raise PairingError("Non-finite JSON number: " + value)
    value = json.loads(raw, object_pairs_hook=_object_pairs, parse_constant=invalid_number)
    require(isinstance(value, dict), "JSON root must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _digest(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def _index(rows, label):
    require(isinstance(rows, list), label + " must be a list")
    result = {}
    for row in rows:
        require(isinstance(row, dict), label + " contains a non-object")
        gid = row.get("group_id")
        require(isinstance(gid, str) and gid not in result, label + " has duplicate/missing group")
        result[gid] = row
    return result


def _time(value):
    require(isinstance(value, str), "Missing allocation/audit timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PairingError("Invalid allocation/audit timestamp") from exc
    require(parsed.tzinfo is not None, "Timestamp must contain a timezone")
    return parsed


def _safe_inventory(files, expected_count):
    require(isinstance(files, dict) and len(files) == expected_count, "Wrong source inventory count")
    for relative, digest in files.items():
        p = PurePosixPath(relative)
        require(relative and not p.is_absolute() and ".." not in p.parts and _digest(digest),
                "Invalid source inventory path/hash")


def _validate_source_groups(manifest):
    groups = _index(manifest.get("groups"), "manifest.groups")
    require(len(groups) == 54 and sum(g.get("target_episodes", 0) for g in groups.values()) == 807,
            "Source must retain all 54 original groups and 807 targets")
    require(len({g.get("family") for g in groups.values()}) == 42, "Source must retain 42 families")
    seen_by_family = {}
    for gid, group in groups.items():
        task = group.get("task")
        require(isinstance(task, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", task)
                and gid == task + "_s0" and type(group.get("eval_seed")) is int
                and group["eval_seed"] == 0 and group.get("family") == task.removesuffix("_random")
                and group.get("variant") == ("random" if task.endswith("_random") else "standard"),
                "Task/family/seed identity mismatch: " + gid)
        budget, target = group.get("step_limit"), group.get("target_episodes")
        require(type(budget) is int and budget > 0 and type(target) is int and target > 0,
                "Invalid native action budget/target: " + gid)
        if gid in DEFERRED:
            require(target == DEFERRED[gid], "Deferred group target changed")
        pool = group.get("native_candidate_pool", {})
        pool_ids = pool.get("candidate_layout_ids")
        ids = group.get("candidate_layout_ids")
        require(isinstance(pool_ids, list) and pool_ids == list(range(pool.get("candidate_count", 0)))
                and isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids)
                and ids == sorted(set(ids)) and len(ids) >= target
                and group.get("candidate_count") == len(ids), "Invalid candidate IDs: " + gid)
        pool_rows = pool.get("layouts", [])
        require([r.get("layout_id") for r in pool_rows] == pool_ids, "Native candidate rows mismatch")
        for key in ("layout_source_sha256_by_id", "layout_canonical_json_sha256_by_id", "layout_content_sha256_by_id"):
            mapping = group.get(key)
            require(isinstance(mapping, dict) and set(mapping) == {str(i) for i in pool_ids}
                    and all(_digest(v) for v in mapping.values()), "Missing/invalid state hashes: " + gid)
        unique_ids, seen = [], set()
        for i in pool_ids:
            digest = group["layout_content_sha256_by_id"][str(i)]
            if digest not in seen:
                unique_ids.append(i); seen.add(digest)
        require(ids == unique_ids, "Source native semantic deduplication changed: " + gid)
        family_seen = seen_by_family.setdefault(group["family"], set())
        require(not family_seen.intersection(seen), "Duplicate physical states across source family variants")
        family_seen.update(seen)
        require(group.get("primary_layout_ids") == ids[:target] and group.get("reserve_layout_ids") == ids[target:]
                and group.get("runtime_overrides") == {"eval_nums": target, "eval_layout_ids": ids}
                and group.get("episode_ids") == [f"{gid}_l{i}" for i in ids], "Source selection metadata mismatch")
        by_id = {r["layout_id"]: r for r in pool_rows}
        require(group.get("layouts") == [by_id[i] for i in ids], "Source path/layout rows mismatch")
        for row in pool_rows:
            p = PurePosixPath(row.get("source_path", ""))
            require(p.is_absolute() and ".." not in p.parts and p.name == row.get("filename"),
                    "Invalid original layout source path")
        # Reconstruct the official group whose digest was frozen before outcome selection.
        original = copy.deepcopy(group)
        source_hash = original.pop("source_group_sha256", None)
        original["target_episodes"] = original.pop("native_target_episodes", None)
        original.pop("native_candidate_pool", None)
        for key in ("runtime_overrides", "layout_source_sha256_by_id", "layout_canonical_json_sha256_by_id",
                    "layout_content_sha256_by_id", "candidate_selection", "primary_layout_ids", "reserve_layout_ids"):
            original.pop(key, None)
        for key in ("candidate_count", "candidate_layout_ids", "episode_ids", "layouts"):
            original[key] = copy.deepcopy(pool[key])
        require(_digest(source_hash) and canonical_sha256(original) == source_hash,
                "Native task/budget provenance hash mismatch: " + gid)
    return groups


def _validate_terminal_receipt(receipt, receipt_sha256, manifest_sha256, audit, latest):
    """Require explicit inspected stop provenance, separate from original scope.

    The full audit proves native scores, source contracts and closed children.
    This receipt attests why the still-incomplete allocation was deliberately
    closed and binds the bounded pre-stop/process evidence inspected by its
    operator. It does not reinterpret any missing episode as a model failure.
    """
    require(isinstance(receipt, dict) and _digest(receipt_sha256),
            "Explicit inspected terminal 721-subset receipt and exact file hash required")
    for key, expected in {
            "schema_version": 1, "campaign": SOURCE_CAMPAIGN,
            "allocation": latest["allocation"],
            "reason": "all_other_groups_complete_user_held_stack_bowls_random",
            "native_scored_episodes": PAIRED_EPISODES,
            "native_successes": audit.get("successes"), "complete_group_count": 47,
            "user_held_group_id": USER_HELD_GROUP, "user_held_scored": USER_HELD_SCORED,
            "user_held_unscored": 6, "children_stopped": True,
            "active_own_gpu_jobs": [], "frozen_sources_unchanged": True,
            "source_manifest_sha256": manifest_sha256}.items():
        require(key in receipt and type(receipt[key]) is type(expected) and receipt[key] == expected,
                "Terminal receipt mismatch: " + key)
    require(_time(latest["started_utc"]) <= _time(receipt.get("pre_stop_verified_utc"))
            <= _time(receipt.get("stop_requested_utc")) <= _time(latest["finished_utc"])
            <= _time(receipt.get("post_cleanup_verified_utc")),
            "Terminal receipt lifecycle ordering differs from allocation evidence")
    for key in ("pre_stop_evidence", "post_cleanup_evidence"):
        evidence = receipt.get(key, {})
        path = evidence.get("path")
        require(isinstance(path, str) and path and _digest(evidence.get("sha256")),
                "Missing terminal receipt evidence reference: " + key)


def _validate_usage(audit, evidence):
    require(audit.get("unexpected_attempt_usage") == [], "Unexpected attempt usage is not accounted for")
    ledgers, unknown_streams = [], {}
    for gid, ev in evidence.items():
        usage = ev.get("usage")
        require(isinstance(usage, list), "Missing attempt usage: " + gid)
        for entry in usage:
            require(isinstance(entry, dict), "Invalid attempt usage entry")
            for key in USAGE_KEYS:
                value = entry.get(key)
                require(type(value) in ((int, float) if key == "summed_request_wall_seconds" else (int,))
                        and math.isfinite(value) and value >= 0, "Invalid usage counter: " + key)
            require(entry["requests_started"] == entry["requests_with_usage"] + entry["requests_without_usage"]
                    and type(entry.get("token_accounting_complete")) is bool
                    and entry["token_accounting_complete"] is (entry["requests_without_usage"] == 0),
                    "Usage accounting fails to distinguish known and unknown requests")
            if entry["requests_without_usage"]:
                attempt = entry.get("attempt_directory")
                streams = entry.get("raw_streams")
                require(isinstance(attempt, str) and PurePosixPath(attempt).name == gid
                        and isinstance(streams, list), "Unknown usage lacks matching attempt/raw-stream evidence")
                missing = [s for s in streams if isinstance(s, dict) and s.get("provider_usage") is None]
                require(len(missing) == entry["requests_without_usage"],
                        "Unknown usage count differs from raw-stream evidence")
                for stream in missing:
                    path = stream.get("path")
                    ordinal = stream.get("request_ordinal")
                    require(type(ordinal) is int and 1 <= ordinal <= entry["requests_started"]
                            and path == str(PurePosixPath(attempt) / "raw_inference" / f"request_{ordinal:04d}.sse")
                            and stream.get("usage_logged") is False and _digest(stream.get("sha256"))
                            and path not in unknown_streams,
                            "Missing/duplicate/unbound unknown-usage raw stream")
                    unknown_streams[path] = stream
        if ev.get("verified_report_count", 0):
            require(sum(u["requests_started"] for u in usage) > 0, "Scored group has no inference evidence: " + gid)
        ledgers.extend(usage)
    totals = audit.get("total_usage", {})
    require(all(type(totals.get(key)) in (int, float) and math.isfinite(totals[key])
                and totals[key] == sum(u[key] for u in ledgers) for key in USAGE_KEYS),
            "Audit totals differ from all attempt usage, including interrupted attempts")
    require(type(audit.get("token_accounting_complete")) is bool
            and audit["token_accounting_complete"] is all(u["token_accounting_complete"] for u in ledgers),
            "Aggregate unknown usage accounting differs")
    warnings = audit.get("issues")
    require(isinstance(warnings, list), "Missing full audit issue list")
    warning_paths = set()
    for warning in warnings:
        require(isinstance(warning, dict) and set(warning) == {"severity", "message", "path"}
                and warning["severity"] == "warning" and warning["message"] == UNKNOWN_USAGE_WARNING
                and warning["path"] in unknown_streams and warning["path"] not in warning_paths,
                "Qwen audit has an error or warning other than evidenced unknown provider usage")
        warning_paths.add(warning["path"])
    require(warning_paths == set(unknown_streams), "Unknown provider usage warnings are missing or unaccounted for")


def build_pairing(manifest, audit, *, manifest_sha256, audit_sha256, manifest_path=None, audit_path=None,
                  terminal_receipt=None, terminal_receipt_sha256=None, terminal_receipt_path=None):
    """Build only from the actual full-audit schema; never select by outcome."""
    require(_digest(manifest_sha256) and _digest(audit_sha256), "Exact input file hashes required")
    require(manifest.get("campaign") == SOURCE_CAMPAIGN and audit.get("campaign") == SOURCE_CAMPAIGN
            and manifest.get("protocol") == "remote_api_full807_v1"
            and manifest.get("model", {}).get("served_alias") == "qwen38-27b", "Wrong Qwen source campaign/protocol")
    require(audit.get("evidence_audited") is True and audit.get("stage_complete_verified") is False
            and audit.get("complete_verified") is False,
            "Qwen terminal full audit must preserve the incomplete original 727/807 scope without integrity issues")
    stage = manifest.get("execution_stage", {})
    require(stage.get("name") == STAGE and stage.get("target_episodes") == 727
            and stage.get("deferred_episode_target") == 80
            and len(stage.get("deferred_group_ids", [])) == 6
            and set(stage["deferred_group_ids"]) == set(DEFERRED) and audit.get("execution_stage") == stage,
            "Wrong authorized execution stage")
    for key, expected in {"planned_episodes": 807, "scored_episodes": PAIRED_EPISODES,
                          "verified_episode_reports": PAIRED_EPISODES, "episodes_with_three_videos": PAIRED_EPISODES,
                          "current_stage_target_episodes": 727, "current_stage_scored_episodes": PAIRED_EPISODES,
                          "current_stage_complete_groups": 47, "environment_blocked_episodes": 80}.items():
        require(type(audit.get(key)) is int and audit[key] == expected, "Audit count mismatch: " + key)
    require(audit.get("official_full_leaderboard_score") is False
            and audit.get("environment_blocked_are_model_failures") is False
            and sorted(audit.get("environment_blocked_groups", [])) == sorted(DEFERRED), "Audit scope differs")
    require(audit.get("maintenance_lineage", {}).get("current_manifest_sha256") == manifest_sha256,
            "Full audit does not bind these exact manifest bytes")
    profile = manifest.get("source_profile", {})
    _safe_inventory(profile.get("files_sha256"), 536)
    _safe_inventory(manifest.get("bootstrap_sha256"), 8)
    require(profile.get("files_manifest_sha256") == canonical_sha256(profile["files_sha256"]),
            "Harness inventory digest mismatch")
    require(audit.get("source_inventory") == {"files_expected": 544, "files_verified": 544},
            "Full audit did not verify all frozen sources")
    allocations = audit.get("allocations")
    require(isinstance(allocations, list) and allocations, "Missing allocation lifecycle evidence")
    allocation_ids = set()
    for row in allocations:
        identifier = row.get("allocation")
        require(isinstance(identifier, str) and identifier.isdigit() and identifier not in allocation_ids,
                "Invalid/duplicate allocation identity")
        allocation_ids.add(identifier)
        require(row.get("children_stopped") is True and not row.get("cleanup_errors")
                and _time(row.get("started_utc")) <= _time(row.get("finished_utc")) <= _time(audit.get("audited_utc")),
                "Allocation is active or lacks clean child cleanup")
    latest = max(allocations, key=lambda row: _time(row["started_utc"]))
    require(latest.get("stage_complete") is False and latest.get("complete") is False
            and type(latest.get("exit_code")) is int and latest["exit_code"] == 1
            and latest.get("execution_stage") == stage and latest.get("error") == TERMINAL_ERROR
            and not latest.get("memory_guard_triggered") and not latest.get("exhausted_groups"),
            "Latest allocation is not the inspected deliberate terminal 721-subset stop")
    _validate_terminal_receipt(terminal_receipt, terminal_receipt_sha256, manifest_sha256, audit, latest)
    receipts = latest.get("environment_blocked_groups", {})
    require(set(receipts) == set(DEFERRED), "Missing deferred environment receipts")
    for gid, target in DEFERRED.items():
        row = receipts[gid]
        require(row.get("state") == "environment_blocked" and row.get("worker_id") is None
                and row.get("target_episodes") == target
                and row.get("prior_evidence", {}).get("state") == "pending"
                and row["prior_evidence"].get("scored") == 0, "Invalid deferred receipt: " + gid)
    groups = _validate_source_groups(manifest)
    progress = _index(audit.get("groups"), "audit.groups")
    evidence = _index(audit.get("group_evidence"), "audit.group_evidence")
    require(set(groups) == set(progress) == set(evidence), "Missing/extra group evidence")
    _validate_usage(audit, evidence)
    output, all_episodes, video_total, success_total = [], [], 0, 0
    for gid, group in groups.items():
        row, ev = progress[gid], evidence[gid]
        native = ev.get("native", {})
        details = native.get("details")
        require(isinstance(details, dict), "Missing native detail rows: " + gid)
        target = 0 if gid in DEFERRED else USER_HELD_SCORED if gid == USER_HELD_GROUP else group["target_episodes"]
        require(set(details) == {str(i) for i in range(target)}, "Missing/extra native indices: " + gid)
        for key in ("family", "dimension", "variant"):
            require(row.get(key) == group.get(key), "Progress group identity differs: " + gid)
        require(row.get("target") == group["target_episodes"] and row.get("scored") == target
                and row.get("complete") is (gid not in DEFERRED and gid != USER_HELD_GROUP)
                and row.get("environment_blocked") is (gid in DEFERRED)
                and native.get("scored") == target and native.get("three_camera_episodes") == target
                and native.get("matching_video_count") == target * 3
                and ev.get("verified_report_count") == target, "Group evidence count mismatch: " + gid)
        ids, successes = [], 0
        videos = native.get("videos", [])
        require(isinstance(videos, list) and len(videos) == target * 3, "Missing/extra camera evidence")
        video_map = {}
        for video in videos:
            name = video.get("name")
            require(isinstance(name, str) and name not in video_map and type(video.get("bytes")) is int
                    and video["bytes"] > 0, "Duplicate/empty/invalid video evidence")
            video_map[name] = video
        pairs = []
        layout_rows = {r["layout_id"]: r for r in group["layouts"]}
        for i in range(target):
            detail = details[str(i)]; layout = detail.get("layout_id")
            require(type(layout) is int and layout in layout_rows and layout not in ids,
                    "Duplicate/unauthorized native scored layout: " + gid)
            require(type(detail.get("success")) is bool and type(detail.get("score")) in (int, float)
                    and math.isfinite(detail["score"]) and 0 <= detail["score"] <= 1, "Invalid native score metadata")
            ids.append(layout); successes += detail["success"]
            outcome = "success" if detail["success"] else "fail"
            for camera in CAMERAS:
                require(f"episode_{i:07d}_cam_{camera}_{outcome}.mp4" in video_map,
                        "Camera/outcome mismatch: " + gid)
            pairs.append({"episode_id": f"{gid}_l{layout}", "qwen_native_index": i,
                          "layout_id": layout, "eval_seed": group["eval_seed"],
                          "source_layout": copy.deepcopy(layout_rows[layout]),
                          "source_sha256": group["layout_source_sha256_by_id"][str(layout)],
                          "canonical_json_sha256": group["layout_canonical_json_sha256_by_id"][str(layout)],
                          "semantic_state_sha256": group["layout_content_sha256_by_id"][str(layout)]})
        require(isinstance(row.get("layout_ids"), list) and len(row["layout_ids"]) == len(ids)
                and set(row["layout_ids"]) == set(ids) and row.get("successes") == successes
                and native.get("successes") == successes, "Native/progress result identities differ")
        if not target:
            require(not any(u.get("requests_started", 0) for u in ev.get("usage", [])),
                    "Deferred group has inference evidence")
            continue
        require(_digest(native.get("sha256")) and isinstance(native.get("path"), str)
                and native["path"] == row.get("native_path")
                and native["path"].endswith(f"/{SOURCE_CAMPAIGN}_{gid}/_result.json"),
                "Native result provenance differs")
        ids = sorted(ids); pairs.sort(key=lambda item: item["layout_id"])
        episode_ids = [f"{gid}_l{i}" for i in ids]
        output.append({"group_id": gid, "task": group["task"], "family": group["family"],
                       "dimension": group["dimension"], "variant": group["variant"],
                       "eval_seed": group["eval_seed"], "step_limit": group["step_limit"],
                       "native_options": copy.deepcopy(group["native_options"]),
                       "source_group_sha256": group["source_group_sha256"],
                       "target_episodes": target, "candidate_count": target,
                       "candidate_layout_ids": ids, "primary_layout_ids": ids, "reserve_layout_ids": [],
                       "episode_ids": episode_ids, "runtime_overrides": {"eval_nums": target, "eval_layout_ids": ids},
                       "qwen_native_result": {"path": native["path"], "sha256": native["sha256"]},
                       "states": pairs})
        all_episodes.extend(episode_ids); video_total += len(videos); success_total += successes
    require(len(output) == 48 and len({g["family"] for g in output}) == 38
            and len(all_episodes) == len(set(all_episodes)) == PAIRED_EPISODES and video_total == 2163,
            "Final paired scope differs from 721 states / 48 configurations / 38 families")
    require(audit.get("successes") == success_total and audit.get("current_stage_successes") == success_total,
            "Aggregate native success metadata differs")
    subset_gate = {"schema_version": 1, "artifact_type": "qwen_authorized_721_subset_gate",
                   "subset_complete_verified": True, "campaign": SOURCE_CAMPAIGN,
                   "paired_scored_episodes": PAIRED_EPISODES, "verified_episode_reports": PAIRED_EPISODES,
                   "verified_camera_videos": video_total, "configuration_count": 48, "family_count": 38,
                   "original_stage_target": 727, "original_stage_complete_verified": False,
                   "original_full_target": 807, "original807_complete_verified": False,
                   "environment_blocked_episodes": 80, "environment_blocked_group_targets": dict(DEFERRED),
                   "user_held_unscored_episodes": 6, "user_held_group_shortfalls": dict(USER_HELD_SHORTFALLS),
                   "unscored_are_model_failures": False, "all_allocation_children_stopped": True,
                   "source_manifest_sha256": manifest_sha256, "source_full_audit_sha256": audit_sha256,
                   "terminal_receipt_sha256": terminal_receipt_sha256,
                   "source_requests_started": audit["total_usage"]["requests_started"],
                   "source_requests_without_usage": audit["total_usage"]["requests_without_usage"],
                   "source_token_accounting_complete": audit["token_accounting_complete"],
                   "preserved_source_audit_warnings": copy.deepcopy(audit["issues"]),
                   "terminal_allocation": latest["allocation"], "pairing_sha256": canonical_sha256(output)}
    return {"schema_version": 2, "artifact_type": "terra_exact_state_pairing",
            "status": "states_frozen_runtime_not_authorized_by_this_artifact",
            "model": "gpt-6.1-terra", "planned_scored_episodes": PAIRED_EPISODES, "group_count": 48, "family_count": 38,
            "source": {"campaign": SOURCE_CAMPAIGN, "execution_stage": copy.deepcopy(stage),
                       "manifest_path": str(manifest_path) if manifest_path else None,
                       "manifest_sha256": manifest_sha256, "audit_path": str(audit_path) if audit_path else None,
                       "audit_sha256": audit_sha256, "audit_timestamp": audit["audited_utc"],
                       "terminal_receipt_path": str(terminal_receipt_path) if terminal_receipt_path else None,
                       "terminal_receipt_sha256": terminal_receipt_sha256,
                       "terminal_allocation": latest["allocation"], "allocation_ids": sorted(allocation_ids)},
            "source_subset_gate": subset_gate, "user_held_group_shortfalls": dict(USER_HELD_SHORTFALLS),
            "source_profile": copy.deepcopy(profile), "source_bootstrap_sha256": copy.deepcopy(manifest["bootstrap_sha256"]),
            "perception_artifacts": copy.deepcopy(manifest.get("perception_artifacts")),
            "grasp_calibration": copy.deepcopy(manifest.get("grasp_calibration")),
            "groups": output, "episode_ids": all_episodes,
            "pairing_sha256": canonical_sha256(output), "deferred_group_targets": dict(DEFERRED),
            "policy": {"new_scores_required": True, "historical_score_reuse": False,
                       "selection_uses_outcomes": False, "native_invalid_initialization": "unscored_missing_pair_no_replacement",
                       "reserve_substitution": False, "scored_failure_retry": False,
                       "action_budget": "identical native task control ticks", "official_full_leaderboard_score": False},
            "limitations": ["State selection is attested by the full Qwen audit; this module does not reread remote raw evidence.",
                            "Matching source states and seeds does not guarantee identical stochastic trajectories.",
                            "The 80 environment-blocked episodes remain outstanding and unscored.",
                            "Six additional stack_bowls_random states remain user-held and unscored; the original 727 stage is incomplete."]}


def build_from_files(manifest_path, audit_path, *, terminal_receipt_path=None):
    manifest, manifest_digest = read_json(manifest_path)
    audit, audit_digest = read_json(audit_path)
    receipt, receipt_digest = read_json(terminal_receipt_path) if terminal_receipt_path else (None, None)
    return build_pairing(manifest, audit, manifest_sha256=manifest_digest, audit_sha256=audit_digest,
                         manifest_path=Path(manifest_path).resolve(), audit_path=Path(audit_path).resolve(),
                         terminal_receipt=receipt, terminal_receipt_sha256=receipt_digest,
                         terminal_receipt_path=Path(terminal_receipt_path).resolve() if terminal_receipt_path else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-manifest", required=True, type=Path)
    parser.add_argument("--qwen-full-audit", required=True, type=Path)
    parser.add_argument("--qwen-terminal-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_from_files(args.qwen_manifest, args.qwen_full_audit,
                              terminal_receipt_path=args.qwen_terminal_receipt)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with args.output.open("x") as handle:
        handle.write(encoded)
    print(json.dumps({"output": str(args.output.resolve()), "episodes": PAIRED_EPISODES,
                      "pairing_sha256": result["pairing_sha256"], "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
