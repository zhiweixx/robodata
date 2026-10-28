#!/usr/bin/env python3
"""A user-requested native-layout subset, not the official full evaluation.

20 distinct initial layouts/family, excluding two previously evaluated families.
The user's approved finite-diversity exceptions are cover_blocks=6 and
play_tic_tac_toe=1. Generalization families retain 10 standard + 10 random.
Deduplication is outcome-independent and preserves original native layout IDs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

class ManifestError(ValueError):
    pass

PROTOCOL = "robodojo_20_states_per_family_subset"
EXCLUDED_FAMILIES = ("general_pickup", "stack_bowls")
EXCEPTION_TARGETS = {"cover_blocks": 6, "play_tic_tac_toe": 1}
POOL_KEYS = ("candidate_count", "candidate_layout_ids", "episode_ids", "layouts")
HASH_METHOD = "canonical parsed layout JSON with unused Room.default_rot removed and missing Room.default_ori materialized as identity"


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def semantic_layout(layout):
    """Normalize only an observed, loader-proven nonphysical metadata change.

    env/scene_manager/objects/room.py:46 reads default_ori with identity fallback;
    default_rot is never read there. This does NOT alter an object's pose.
    """
    result = copy.deepcopy(layout)
    room = result.get("Room") if isinstance(result, dict) else None
    if isinstance(room, dict):
        room.pop("default_rot", None)
        room.setdefault("default_ori", [1.0, 0.0, 0.0, 0.0])
    return result


def _validate_group_identity(group, *, contiguous=True):
    task, seed = group["task"], group["eval_seed"]
    gid = f"{task}_s{seed}"
    if group["group_id"] != gid or group["family"] != task.removesuffix("_random"):
        raise ManifestError(f"{task}: invalid native group/family identity")
    if group["variant"] != ("random" if task.endswith("_random") else "standard"):
        raise ManifestError(f"{task}: invalid variant")
    count, ids = group["candidate_count"], group["candidate_layout_ids"]
    if type(count) is not int or count < group["target_episodes"]:
        raise ManifestError(f"{task}: insufficient candidate pool")
    if len(ids) != count or ids != sorted(set(ids)) or any(type(i) is not int or i < 0 for i in ids):
        raise ManifestError(f"{task}: invalid candidate IDs")
    if contiguous and ids != list(range(count)):
        raise ManifestError(f"{task}: source native enumeration changed")
    if group["episode_ids"] != [f"{gid}_l{i}" for i in ids]:
        raise ManifestError(f"{task}: episode IDs differ from native layout IDs")
    layouts = group["layouts"]
    if len(layouts) != count or [x["layout_id"] for x in layouts] != ids:
        raise ManifestError(f"{task}: layout records differ from candidate IDs")
    if len({x["source_path"] for x in layouts}) != count:
        raise ManifestError(f"{task}: repeated layout file identity")
    for item in layouts:
        if (item["filename"] != f"{task}_{item['filename_suffix']}.json"
                or Path(item["source_path"]).name != item["filename"]):
            raise ManifestError(f"{task}: filename differs from task/suffix")
    suffixes = [x["filename_suffix"] for x in layouts]
    if suffixes != sorted(set(suffixes)):
        raise ManifestError(f"{task}: duplicate or out-of-order filename suffix")
    if type(group["step_limit"]) is not int or group["step_limit"] <= 0:
        raise ManifestError(f"{task}: invalid action budget")


def collect_layout_evidence(full_manifest):
    """Read existing layout assets only; never launch a simulator or score model."""
    groups = {}
    for group in full_manifest["groups"]:
        # Seed0 evaluates new groups; all seeds establish finite diversity of
        # the exceptions; seed0 of excluded tasks supports legacy reuse audit.
        if group["eval_seed"] != 0 and group["family"] not in EXCEPTION_TARGETS:
            continue
        rows = []
        for item in group["layouts"]:
            raw = Path(item["source_path"]).read_bytes()
            obj = json.loads(raw)
            rows.append({"layout_id": item["layout_id"], "source_path": item["source_path"],
                         "bytes": len(raw), "source_sha256": hashlib.sha256(raw).hexdigest(),
                         "canonical_json_sha256": canonical_sha256(obj),
                         "semantic_state_sha256": canonical_sha256(semantic_layout(obj))})
        groups[group["group_id"]] = {"task": group["task"], "family": group["family"],
                                      "eval_seed": group["eval_seed"], "layouts": rows}
    return {"schema_version": 1, "source_full_manifest_canonical_sha256": canonical_sha256(full_manifest),
            "semantic_hash_method": HASH_METHOD,
            "normalization_source": "env/scene_manager/objects/room.py:46 reads default_ori and defaults to [1,0,0,0]; unused default_rot does not change rendered Room orientation",
            "groups": groups}


def _first_unique_ids(pool, hashes):
    seen, result = set(), []
    for idx in pool["candidate_layout_ids"]:
        digest = hashes[str(idx)]
        if digest not in seen:
            seen.add(digest)
            result.append(idx)
    return result


def validate_subset_manifest(manifest):
    expected = {"protocol": PROTOCOL, "is_official_full_evaluation": False,
                "benchmark": "RoboDojo", "embodiment": "arx_x5", "family_count": 40,
                "task_count": 51, "group_count": 51, "total_target_episodes": 767,
                "per_family_target": 20, "eval_seeds": [0], "overall_family_count": 42,
                "overall_reference_target_episodes": 807,
                "excluded_families": list(EXCLUDED_FAMILIES), "exception_family_targets": EXCEPTION_TARGETS}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ManifestError("custom-subset scope or approved finite-diversity exception changed")
    if len(manifest.get("source_full_manifest_sha256", "")) != 64:
        raise ManifestError("source full-manifest hash missing")
    groups = manifest["groups"]
    if len(groups) != 51 or len({g["group_id"] for g in groups}) != 51:
        raise ManifestError("expected51uniqueconfiguration/seed groups")
    by_family = defaultdict(list)
    for group in groups:
        _validate_group_identity(group, contiguous=False)
        if group["eval_seed"] != 0 or group["family"] in EXCLUDED_FAMILIES:
            raise ManifestError("excluded family or repeated seed in subset")
        native_target = 25 if group["dimension"] == "generalization" else 50
        target = EXCEPTION_TARGETS.get(group["family"], 10 if group["dimension"] == "generalization" else 20)
        if (group.get("native_target_episodes") != native_target
                or group["native_options"].get("eval_nums") != native_target
                or group["target_episodes"] != target
                or group["runtime_overrides"] != {"eval_nums": target, "eval_layout_ids": group["candidate_layout_ids"]}):
            raise ManifestError(f"{group['task']}: count override or native provenance mismatch")
        if group['family'] in EXCEPTION_TARGETS and group['candidate_count'] != target:
            raise ManifestError(f"{group['task']}: finite-diversity evidence changed")
        original = copy.deepcopy(group)
        source_group_hash = original.pop("source_group_sha256")
        original["target_episodes"] = original.pop("native_target_episodes")
        original.pop("runtime_overrides")
        pool = original.pop("native_candidate_pool")
        for key in POOL_KEYS:
            original[key] = pool[key]
        hashes = original.pop("layout_content_sha256_by_id")
        for key in ("layout_canonical_json_sha256_by_id", "layout_source_sha256_by_id", "candidate_selection"):
            original.pop(key)
        _validate_group_identity(original)
        if canonical_sha256(original) != source_group_hash:
            raise ManifestError(f"{group['task']}: native group changed beyond declared subset overrides")
        expected_keys = {str(i) for i in pool["candidate_layout_ids"]}
        for key in ("layout_content_sha256_by_id", "layout_canonical_json_sha256_by_id", "layout_source_sha256_by_id"):
            if set(group[key]) != expected_keys or any(len(v) != 64 for v in group[key].values()):
                raise ManifestError(f"{group['task']}: incomplete layout content evidence")
        selected = _first_unique_ids(pool, hashes)
        if group["candidate_layout_ids"] != selected:
            raise ManifestError(f"{group['task']}: pool is not first-occurrence semantic-state deduplication")
        originals_by_id = {x["layout_id"]: x for x in pool["layouts"]}
        if group["layouts"] != [originals_by_id[i] for i in selected]:
            raise ManifestError(f"{group['task']}: selected layout files changed")
        by_family[group["family"]].append(group)
    if len(by_family) != 40:
        raise ManifestError("expected40remainingfamilies")
    family_targets = {}
    for family, rows in by_family.items():
        dims = {r["dimension"] for r in rows}
        paired = dims == {"generalization"}
        if (len(dims) != 1 or len(rows) != (2 if paired else 1)
                or {r["variant"] for r in rows} != ({"standard", "random"} if paired else {"standard"})):
            raise ManifestError(f"{family}: native variant coverage changed")
        target = sum(r["target_episodes"] for r in rows)
        if target != EXCEPTION_TARGETS.get(family, 20):
            raise ManifestError(f"{family}: family target changed")
        family_targets[family] = target
        digests = [r["layout_content_sha256_by_id"][str(i)] for r in rows for i in r["candidate_layout_ids"]]
        if len(digests) != len(set(digests)):
            raise ManifestError(f"{family}: repeated state across native variants")
    if manifest.get("family_targets") != family_targets:
        raise ManifestError("family target summary differs from groups")
    ids = [eid for group in groups for eid in group["episode_ids"]]
    seeds = {eid: group["eval_seed"] for group in groups for eid in group["episode_ids"]}
    if (len(ids) != len(set(ids)) or manifest["episode_ids"] != ids
            or manifest["episode_seeds"] != seeds or manifest["total_candidate_episodes"] != len(ids)):
        raise ManifestError("candidate authorization differs from subset groups")
    return manifest


def verify_group_layout_sources(group):
    """Optional startup check that pinned source bytes and semantic states match."""
    for item in group["native_candidate_pool"]["layouts"]:
        idx = str(item["layout_id"])
        raw = Path(item["source_path"]).read_bytes()
        obj = json.loads(raw)
        if (hashlib.sha256(raw).hexdigest() != group["layout_source_sha256_by_id"][idx]
                or canonical_sha256(obj) != group["layout_canonical_json_sha256_by_id"][idx]
                or canonical_sha256(semantic_layout(obj)) != group["layout_content_sha256_by_id"][idx]):
            raise ManifestError(f"{group['group_id']} layout{idx}: source asset hash changed")
    return {"group_id": group["group_id"], "verified_native_candidates": group["native_candidate_pool"]["candidate_count"]}


def build_subset_manifest(full_manifest, *, source_manifest_path=None, source_manifest_sha256=None, content_evidence=None):
    full = full_manifest
    expected = {"family_count": 42, "task_count": 54, "group_count": 162,
                "total_target_episodes": 6300, "eval_seeds": [0, 1, 2]}
    if any(full.get(k) != v for k, v in expected.items()):
        raise ManifestError("source is not frozen full native scope")
    if len(full["groups"]) != 162 or len({g["group_id"] for g in full["groups"]}) != 162:
        raise ManifestError("source group identities invalid")
    seed_zero = [g for g in full["groups"] if g["eval_seed"] == 0]
    if len(seed_zero) != 54 or len({g["family"] for g in seed_zero}) != 42:
        raise ManifestError("source seed0 does not cover54configurations/42families")
    evidence = content_evidence if content_evidence is not None else collect_layout_evidence(full)
    if evidence["source_full_manifest_canonical_sha256"] != canonical_sha256(full):
        raise ManifestError("layout evidence belongs to different full manifest")
    if evidence["semantic_hash_method"] != HASH_METHOD:
        raise ManifestError("unknown physical-state normalization")
    groups = []
    for original in sorted(seed_zero, key=lambda g: (g["family"], g["variant"] == "random")):
        _validate_group_identity(original)
        if original["family"] in EXCLUDED_FAMILIES:
            continue
        target = 25 if original["dimension"] == "generalization" else 50
        if original["target_episodes"] != target or original["native_options"].get("eval_nums") != target:
            raise ManifestError("source native count changed")
        rows = evidence["groups"][original["group_id"]]["layouts"]
        if [(r["layout_id"], r["source_path"]) for r in rows] != [(r["layout_id"], r["source_path"]) for r in original["layouts"]]:
            raise ManifestError("layout evidence identities differ from source manifest")
        group = copy.deepcopy(original)
        group["native_target_episodes"] = original["target_episodes"]
        group["target_episodes"] = EXCEPTION_TARGETS.get(group["family"], 10 if group["dimension"] == "generalization" else 20)
        group["source_group_sha256"] = canonical_sha256(original)
        group["native_candidate_pool"] = {k: copy.deepcopy(original[k]) for k in POOL_KEYS}
        for key, row_key in [("layout_content_sha256_by_id", "semantic_state_sha256"),
                             ("layout_canonical_json_sha256_by_id", "canonical_json_sha256"),
                             ("layout_source_sha256_by_id", "source_sha256")]:
            group[key] = {str(r["layout_id"]): r[row_key] for r in rows}
        selected = _first_unique_ids(original, group["layout_content_sha256_by_id"])
        group["candidate_layout_ids"] = selected
        group["candidate_count"] = len(selected)
        group["episode_ids"] = [f"{group['group_id']}_l{i}" for i in selected]
        original_layouts = {r["layout_id"]: r for r in original["layouts"]}
        group["layouts"] = [original_layouts[i] for i in selected]
        group["candidate_selection"] = "First native layout ID for each distinct semantic state; retain all unique candidates for native invalid-layout replacement"
        group["runtime_overrides"] = {"eval_nums": group["target_episodes"], "eval_layout_ids": selected}
        groups.append(group)
    ids = [eid for group in groups for eid in group["episode_ids"]]
    family_targets = dict(sorted((family, sum(g["target_episodes"] for g in groups if g["family"] == family))
                                for family in {g["family"] for g in groups}))
    result = {"schema_version": 2, "protocol": PROTOCOL, "is_official_full_evaluation": False,
              "benchmark": full["benchmark"], "embodiment": full["embodiment"],
              "family_count": len(family_targets), "task_count": len(groups), "group_count": len(groups),
              "eval_seeds": [0], "per_family_target": 20, "exception_family_targets": EXCEPTION_TARGETS,
              "family_targets": family_targets, "total_target_episodes": sum(family_targets.values()),
              "total_candidate_episodes": len(ids), "overall_family_count": 42,
              "overall_reference_target_episodes": sum(family_targets.values())+40,
              "excluded_families": list(EXCLUDED_FAMILIES),
              "excluded_task_configurations": sorted(g["task"] for g in seed_zero if g["family"] in EXCLUDED_FAMILIES),
              "prior_results": {"included_in_new_target": False, "expected_reference_episodes": 40,
                                "policy": "Excluded families are not rerun; legacy20selection must retain original phase/provenance and verify distinct layouts."},
              "selection_policy": {"eval_seed": 0, "generalization_allocation": "10 standard + 10 random per included generalization family",
                                   "state_order": "First unique semantic-state occurrence in native layout enumeration",
                                   "invalid_layouts": "Native skip/replacement within all selected unique candidates; never count invalid initialization as failure",
                                   "finite_diversity_exhaustion": "If invalidity exhausts the unique pool, report a shortfall; never repeat a duplicate state or synthesize a replacement",
                                   "failed_scored_layouts": "Count once; never retry scored policy failures",
                                   "group_order": "Alphabetical family, standard before random",
                                   "semantic_hash_method": HASH_METHOD,
                                   "exceptions": "User approved using only distinct official initial layouts: cover_blocks6, play_tic_tac_toe1. No synthetic states."},
              "action_budget_unit": full["action_budget_unit"], "episode_id_format": full["episode_id_format"],
              "layout_id_rule": full["layout_id_rule"], "source": copy.deepcopy(full["source"]),
              "source_full_manifest_path": None if source_manifest_path is None else str(source_manifest_path),
              "source_full_manifest_sha256": source_manifest_sha256 or canonical_sha256(full),
              "source_full_manifest_hash_encoding": "exact source-file bytes" if source_manifest_sha256 else "canonical JSON: sorted keys, compact separators, ASCII",
              "source_full_manifest_canonical_sha256": canonical_sha256(full),
              "layout_evidence_canonical_sha256": canonical_sha256(evidence),
              "episode_ids": ids, "episode_seeds": {eid: 0 for eid in ids}, "groups": groups}
    return validate_subset_manifest(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--layout-evidence", help="Optional previously collected, source-bound layout hashes")
    parser.add_argument("--write-layout-evidence", help="Save all raw/canonical/semantic layout hashes used")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        source = Path(args.source_manifest).expanduser().resolve(strict=True)
        raw = source.read_bytes()
        full = json.loads(raw)
        evidence = json.loads(Path(args.layout_evidence).read_text()) if args.layout_evidence else collect_layout_evidence(full)
        result = build_subset_manifest(full, source_manifest_path=source,
                                       source_manifest_sha256=hashlib.sha256(raw).hexdigest(), content_evidence=evidence)
        if args.write_layout_evidence:
            Path(args.write_layout_evidence).write_text(json.dumps(evidence, indent=2, sort_keys=True)+"\n")
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=output.name+".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        try:
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    except (ManifestError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"subset manifest error: {exc}\n")
    print(json.dumps({"manifest": str(output), "families": result["family_count"],
                      "groups": result["group_count"], "target_episodes": result["total_target_episodes"],
                      "candidate_episodes": result["total_candidate_episodes"]}))


if __name__ == "__main__":
    main()
