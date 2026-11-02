#!/usr/bin/env python3
"""Read-only audit for the user-requested 20-state-per-family custom subset.

The new campaign contains 767 scored episodes across 40 families. Two families
have only six/one distinct official initial states; the others target20. The two
excluded families may contribute their first 20 seed-0 native scored outcomes
through an explicit frozen provenance file; they are never replayed here.
This is not the official 6300-episode evaluation or an official leaderboard Avg.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import audit_full_results as full
from build_full_manifest import literal_assignment

EXCLUDED = frozenset({"general_pickup", "stack_bowls"})
PER_FAMILY = 20
NEW_FAMILIES = 40
FAMILY_EXCEPTIONS = {"cover_blocks": 6, "play_tic_tac_toe": 1}
NEW_TARGET = 767
COMBINED_TARGET = 807
REUSE_SELECTION = "first_20_native_scored_seed0"
SEMANTIC_HASH_METHOD = "canonical parsed layout JSON with unused Room.default_rot removed and missing Room.default_ori materialized as identity"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_subset_manifest(manifest):
    """Pin the family inventory and balanced variant allocation, not just totals."""
    if (manifest.get("protocol") != "robodojo_20_states_per_family_subset"
            or manifest.get("is_official_full_evaluation") is not False
            or set(manifest.get("excluded_families", [])) != EXCLUDED
            or manifest.get("exception_family_targets") != FAMILY_EXCEPTIONS
            or manifest.get("selection_policy", {}).get("semantic_hash_method") != SEMANTIC_HASH_METHOD):
        raise ValueError("manifest is not the authorized custom subset protocol")
    source = manifest["source"]
    inventory_rel = "scripts/internal/task_inventory.py"
    inventory = Path(source["robodojo_root"]) / inventory_rel
    if sha256(inventory) != source["sha256"][inventory_rel]:
        raise ValueError("official task inventory differs from manifest provenance")
    dimensions = literal_assignment(inventory, "DIMENSION_TASKS")
    family_dimensions = {family: dimension for dimension, names in dimensions.items()
                         for family in names}
    if len(family_dimensions) != 42 or not EXCLUDED <= set(family_dimensions):
        raise ValueError("official family inventory differs from the selected 42-family scope")
    wanted = set(family_dimensions) - EXCLUDED
    groups = manifest["groups"]
    observed_families = {group["family"] for group in groups}
    if observed_families != wanted or len(wanted) != NEW_FAMILIES:
        raise ValueError("subset must contain exactly the 40 non-excluded official families")
    by_family = defaultdict(list)
    for group in groups:
        family = group["family"]
        if group["dimension"] != family_dimensions[family]:
            raise ValueError("task dimension differs from official inventory")
        if type(group["eval_seed"]) is not int or group["eval_seed"] != 0:
            raise ValueError("subset requires seed0 with distinct native layout identities")
        pool = group["native_candidate_pool"]
        native_ids = pool["candidate_layout_ids"]
        hashes = group["layout_content_sha256_by_id"]
        if len(native_ids) != len(set(native_ids)):
            raise ValueError("duplicate original native candidate identity")
        for hash_key in ("layout_content_sha256_by_id", "layout_canonical_json_sha256_by_id",
                         "layout_source_sha256_by_id"):
            mapping = group[hash_key]
            if set(mapping) != {str(i) for i in native_ids}:
                raise ValueError("semantic, raw JSON and file hashes must cover every original native candidate")
            for digest in mapping.values():
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest)):
                    raise ValueError("invalid layout evidence SHA256")
        seen, selected = set(), []
        for layout_id in native_ids:
            digest = hashes[str(layout_id)]
            if digest not in seen:
                selected.append(layout_id)
                seen.add(digest)
        if group["candidate_layout_ids"] != selected:
            raise ValueError("authorized candidates must be first-occurrence distinct official initial states")
        by_family[family].append(group)
    for family, items in by_family.items():
        generalization = family_dimensions[family] == "generalization"
        tasks = {family, family + "_random"} if generalization else {family}
        if len(items) != len(tasks) or {g["task"] for g in items} != tasks:
            raise ValueError(f"{family}: missing, duplicate or unexpected native variant")
        family_target = FAMILY_EXCEPTIONS.get(family, PER_FAMILY)
        each_target = family_target // len(tasks)
        if any(type(g["target_episodes"]) is not int or g["target_episodes"] != each_target
               for g in items):
            raise ValueError(f"{family}: expected {family_target} total states balanced across variants")
        digests = [g["layout_content_sha256_by_id"][str(i)]
                   for g in items for i in g["candidate_layout_ids"]]
        if len(digests) != len(set(digests)):
            raise ValueError(f"{family}: duplicate initial state across task variants")
    if manifest.get("family_targets") != {family: FAMILY_EXCEPTIONS.get(family, PER_FAMILY)
                                          for family in wanted}:
        raise ValueError("family target metadata differs from selected scope")
    if manifest["total_target_episodes"] != NEW_TARGET:
        raise ValueError("subset target must be 767 new scored episodes")
    return {"kind": "custom_20_states_per_family", "official_full_evaluation": False,
            "families": NEW_FAMILIES, "task_configurations": len(groups), "groups": len(groups),
            "seeds": [0], "target_scored": NEW_TARGET, "per_family_target": PER_FAMILY,
            "family_target_exceptions": FAMILY_EXCEPTIONS,
            "excluded_families": sorted(EXCLUDED),
            "variant_policy": "10 standard + 10 random for generalization families; 20 otherwise except cover_blocks6/play_tic_tac_toe1",
            "state_identity": "first native layout ID per normalized initial-state content hash; no duplicate-state replay across configurations of a family",
            "semantic_hash_method": SEMANTIC_HASH_METHOD}


def audit_reused_results(path, *, simroot, progress_only):
    """Validate immutable first20 selection; defer source videos/attempts to full audit."""
    if not path.exists():
        return {"present": False, "scored": 0, "success": 0, "score_sum": 0.0,
                "target": 40, "evidence_audited": False, "errors": [], "groups": []}
    reuse = full.read_json(path)
    if reuse.get("schema_version") != 1 or reuse.get("selection") != REUSE_SELECTION:
        raise ValueError("unsupported reused-result selection protocol")
    source_dir = Path(reuse["source_campaign_dir"]).resolve(strict=True)
    source_manifest = full.read_json(source_dir / "manifest.json")
    source_groups = {g["group_id"]: g for g in source_manifest["groups"]}
    selected = reuse["groups"]
    expected_ids = {task + "_s0" for task in EXCLUDED}
    if len(selected) != 2 or {row["group_id"] for row in selected} != expected_ids:
        raise ValueError("reused results must select only the two excluded seed0 groups")
    errors, rows = [], []
    for item in selected:
        gid = item["group_id"]
        group = source_groups[gid]
        if group["family"] not in EXCLUDED or group["task"] != group["family"] or group["eval_seed"] != 0:
            raise ValueError("reused source group identity mismatch")
        if item["selected_native_indices"] != list(range(PER_FAMILY)):
            raise ValueError("reused subset must select native indices0..19 without outcome-based filtering")
        native_path = full.expected_result_path(simroot, source_dir.name, group)
        native_sha = sha256(native_path)
        if native_sha != item["native_result_sha256"]:
            raise ValueError("reused native result changed since the selection was frozen")
        details = full.native_details(full.read_json(native_path), group)
        if len(details) < PER_FAMILY:
            raise ValueError("reused source has fewer than20 valid scored layouts")
        picked = [{"native_index": i, **details[str(i)]} for i in range(PER_FAMILY)]
        evidence_verified = False
        if not progress_only:
            inspected = full.audit_group(source_dir, simroot, group, errors, progress_only=False)
            evidence_verified = (inspected["native_complete_verified"]
                and inspected["episode_reports_verified"] == inspected["scored"]
                and inspected["video_evidence"]["verified_scored_episodes"] == inspected["scored"])
            if not evidence_verified:
                full.issue(errors, native_path, "reused source group lacks fully verified native/report/video evidence")
        rows.append({"group_id": gid, "family": group["family"], "category": group["dimension"],
                     "source_campaign": source_dir.name, "native_result_path": str(native_path),
                     "native_result_sha256": native_sha, "selected_native_indices": list(range(PER_FAMILY)),
                     "scored": PER_FAMILY, "success": sum(d["success"] for d in picked),
                     "score_sum": sum(d["score"] for d in picked), "selected_details": picked,
                     "evidence_verified": evidence_verified})
    return {"present": True, "selection": REUSE_SELECTION, "path": str(path),
            "provenance_sha256": sha256(path), "source_campaign_dir": str(source_dir),
            "target": 40, "scored": sum(row["scored"] for row in rows),
            "success": sum(row["success"] for row in rows),
            "score_sum": sum(row["score_sum"] for row in rows), "groups": rows, "errors": errors,
            "evidence_audited": not progress_only and all(row["evidence_verified"] for row in rows) and not errors,
            "inference_provenance_note": "Reused outcomes retain their original allocation/inference phase; they are not relabeled as new BF16+MTP3 episodes."}


def audit_campaign(campaign_dir, *, simroot=None, now=None, include_details=False,
                   progress_only=False, reuse_file=None):
    campaign_dir = Path(campaign_dir).resolve(strict=True)
    simroot = Path(simroot).resolve(strict=True) if simroot else campaign_dir.parents[2]
    manifest = full.read_json(campaign_dir / "manifest.json")
    scope = validate_subset_manifest(manifest)
    # Reuse all established per-group identity, budget, native-score, video,
    # committed-attempt and usage checks. Only the expected sweep size changes.
    result = full.audit_campaign(campaign_dir, simroot=simroot, now=now,
                                include_details=include_details, progress_only=progress_only)
    result.pop("official_scope", None)
    result.pop("official_summarizer_input", None)
    result["evaluation_scope"] = scope
    result["audit_mode"] = "progress_only" if progress_only else "full_evidence"
    result["official_leaderboard_eligible"] = False
    result["summary_note"] = "Custom user-selected subset; report family/category breakdowns separately from the official full-benchmark Avg."
    wanted_groups = {g["group_id"] for g in manifest["groups"]}
    unexpected = sorted(p.name for p in (campaign_dir / "groups").glob("*")
                        if p.is_dir() and p.name not in wanted_groups)
    for name in unexpected:
        full.issue(result["errors"], campaign_dir / "groups" / name,
                   "unexpected group (including smoke/excluded tasks) in subset campaign")
    result["unexpected_group_directories"] = unexpected
    if result["groups"]["states"].get("excluded_non_evaluation", 0):
        full.issue(result["errors"], campaign_dir / "groups",
                   "non-evaluation/smoke report found under an authorized scored group")
    evidence = result.get("evidence", {})
    result["campaign_complete"] = (not progress_only and not result["errors"]
        and result["groups"]["native_complete_verified"] == scope["groups"]
        and result["scoring"]["scored"] == NEW_TARGET
        and evidence.get("scored_episodes_with_three_matching_nonempty_videos") == NEW_TARGET
        and evidence.get("scored_episodes_with_committed_report_and_native_budget") == NEW_TARGET)
    result["final_evidence_audit_required"] = not result["campaign_complete"]
    result["evidence_audited"] = not progress_only
    result["usage_audited"] = not progress_only
    reuse_path = Path(reuse_file) if reuse_file else campaign_dir / "reused_results.json"
    try:
        reused = audit_reused_results(reuse_path, simroot=simroot, progress_only=progress_only)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        reused = {"present": reuse_path.exists(), "scored": 0, "success": 0, "score_sum": 0.0,
                  "target": 40, "evidence_audited": False, "groups": [],
                  "errors": [{"path": str(reuse_path), "message": str(exc)}]}
    result["reused_results"] = reused
    n = result["scoring"]["scored"] + reused["scored"]
    success = result["scoring"]["success"] + reused["success"]
    result["combined_807_summary"] = {
        "target": COMBINED_TARGET, "scored": n, "success": success,
        "success_rate": success / n if n else None,
        "score_sum": result["scoring"]["score_sum"] + reused["score_sum"],
        "complete": result["campaign_complete"] and reused["evidence_audited"],
        "new_scored": result["scoring"]["scored"], "reused_scored": reused["scored"],
        "scope": "767 new outcomes plus first20 seed0 outcomes for each of two excluded families",
        "comparability_note": "Historical stack_bowls reuses20 standard states, whereas new generalization families use10 standard +10 random. Reused inference phases may differ. Keep new767 primary and combined807 descriptive.",
        "official_avg": False}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--simroot", type=Path)
    parser.add_argument("--reuse-file", type=Path)
    parser.add_argument("--now-utc")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--progress-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = audit_campaign(args.campaign_dir, simroot=args.simroot,
            now=full.timestamp(args.now_utc) if args.now_utc else None,
            include_details=args.details, progress_only=args.progress_only, reuse_file=args.reuse_file)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"subset audit error: {exc}\n")
    if args.summary_only:
        for key in ("by_family", "by_category_family_seed", "native_action_budgets", "group_details"):
            result.pop(key, None)
        for row in result["reused_results"].get("groups", []):
            row.pop("selected_details", None)
    print(json.dumps(result, indent=None if args.compact else 2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
