"""Deterministic state-count subsets of a pinned official seed-0 catalog.

This selects evaluation layouts without executing task code, a simulator or a
model. It never changes native action budgets, synthesizes states, or reads prior
scores. The separately pinned legacy 767-episode manifest remains unchanged.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path

from .subset_validation import (
    HASH_METHOD, ManifestError, POOL_KEYS, _first_unique_ids,
    _validate_group_identity, canonical_sha256,
)

STATE_SUBSET_SCHEMA = "robodata.robodojo.state_subset.v1"
CATALOG_SCHEMA = "robodata.robodojo.seed0_catalog.v1"
CATALOG_PATH = Path(__file__).with_name("catalogs") / "seed0.json"
CATALOG_SHA256 = "7043c73157f1043e5da4f386ebb196106da962e585eb9f92b9209efa6e85e324"
HASH_KEYS = (
    "layout_content_sha256_by_id",
    "layout_canonical_json_sha256_by_id",
    "layout_source_sha256_by_id",
)


def load_catalog():
    """Verify packaged official identities and evidence before using them."""
    raw = CATALOG_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CATALOG_SHA256:
        raise ManifestError("pinned seed-0 state catalog changed")
    catalog = json.loads(raw)
    if (catalog.get("schema") != CATALOG_SCHEMA or catalog.get("family_count") != 42
            or catalog.get("group_count") != 54 or catalog.get("eval_seeds") != [0]
            or catalog.get("semantic_hash_method") != HASH_METHOD):
        raise ManifestError("unsupported official state catalog")
    return catalog


def _selection(states_per_task, exclude_families, catalog):
    # Native generalization groups each score at most25 in this protocol. Even
    # counts preserve equal standard/random allocation without a hidden tie rule.
    if (type(states_per_task) is not int or not 2 <= states_per_task <= 50
            or states_per_task % 2):
        raise ManifestError("states_per_task must be an even integer from 2 through 50")
    if not isinstance(exclude_families, (list, tuple)):
        raise ManifestError("exclude_families must be a list or tuple of task-family names")
    if any(not isinstance(name, str) for name in exclude_families):
        raise ManifestError("excluded family names must be strings")
    families = {group["family"] for group in catalog["groups"]}
    exclusions = sorted(exclude_families)
    if len(exclusions) != len(set(exclusions)):
        raise ManifestError("excluded family names must not be repeated")
    unknown = set(exclusions) - families
    if unknown:
        raise ManifestError("unknown task families: " + ", ".join(sorted(unknown)))
    if set(exclusions) == families:
        raise ManifestError("a state subset must retain at least one task family")
    return exclusions


def build_state_subset(states_per_task=30, exclude_families=()):
    """Build a source-bound, balanced subset with all unique replacement states.

    ``states_per_task`` means a family total:30 selects15 standard and15 random
    states for paired generalization families. Limited official diversity reduces
    the target, with the shortfall recorded explicitly. The native evaluator may
    exhaust invalid candidates before reaching that target; it must then report
    a partial run, never duplicate a state to fill the requested count.
    """
    catalog = load_catalog()
    exclusions = _selection(states_per_task, exclude_families, catalog)
    groups = []
    by_family = defaultdict(list)
    for record in catalog["groups"]:
        if record["family"] in exclusions:
            continue
        original = copy.deepcopy(record)
        hashes = {key: original.pop(key) for key in HASH_KEYS}
        _validate_group_identity(original)
        selected = _first_unique_ids(original, hashes["layout_content_sha256_by_id"])
        native_count = 25 if original["dimension"] == "generalization" else 50
        requested = states_per_task // 2 if native_count == 25 else states_per_task
        if (original["target_episodes"] != native_count
                or original["native_options"].get("eval_nums") != native_count):
            raise ManifestError("catalog native scoring count changed")
        target = min(requested, len(selected))
        if target <= 0:
            raise ManifestError("catalog task has no unique official candidate states")
        group = copy.deepcopy(original)
        group.update(hashes)
        group["source_group_sha256"] = canonical_sha256(original)
        group["native_target_episodes"] = native_count
        group["target_episodes"] = target
        group["native_candidate_pool"] = {key: copy.deepcopy(original[key]) for key in POOL_KEYS}
        group["candidate_layout_ids"] = selected
        group["candidate_count"] = len(selected)
        group["episode_ids"] = [f"{group['group_id']}_l{index}" for index in selected]
        source_layouts = {row["layout_id"]: row for row in original["layouts"]}
        group["layouts"] = [source_layouts[index] for index in selected]
        group["candidate_selection"] = (
            "First native layout ID for each distinct semantic state; retain all "
            "unique candidates for native invalid-layout replacement"
        )
        group["runtime_overrides"] = {"eval_nums": target, "eval_layout_ids": selected}
        _validate_group_identity(group, contiguous=False)
        groups.append(group)
        by_family[group["family"]].append(group)

    family_targets, available = {}, {}
    for family, rows in sorted(by_family.items()):
        paired = rows[0]["dimension"] == "generalization"
        if ({row["variant"] for row in rows}
                != ({"standard", "random"} if paired else {"standard"})
                or len(rows) != (2 if paired else 1)):
            raise ManifestError("catalog variant coverage is incomplete")
        states = [row["layout_content_sha256_by_id"][str(index)]
                  for row in rows for index in row["candidate_layout_ids"]]
        if len(states) != len(set(states)):
            raise ManifestError("catalog repeats a physical state across family variants")
        family_targets[family] = sum(row["target_episodes"] for row in rows)
        available[family] = len(states)
    exceptions = {family: target for family, target in family_targets.items()
                  if target < states_per_task}
    ids = [eid for group in groups for eid in group["episode_ids"]]
    if len(ids) != len(set(ids)):
        raise ManifestError("catalog repeats native episode identities")
    total = sum(family_targets.values())
    return {
        "schema_version": 1,
        "portable_schema": STATE_SUBSET_SCHEMA,
        "protocol": "robodojo_distinct_states_per_family_subset",
        "is_official_full_evaluation": False,
        "benchmark": catalog["benchmark"], "embodiment": catalog["embodiment"],
        "catalog_schema": CATALOG_SCHEMA, "catalog_sha256": CATALOG_SHA256,
        "family_count": len(family_targets), "task_count": len(groups),
        "group_count": len(groups), "eval_seeds": [0],
        "per_family_target": states_per_task,
        "family_targets": family_targets, "available_states_per_family": available,
        "requested_target_episodes": states_per_task * len(family_targets),
        "total_target_episodes": total, "total_candidate_episodes": len(ids),
        "exception_family_targets": exceptions, "excluded_families": exclusions,
        "excluded_task_configurations": sorted(
            g["task"] for g in catalog["groups"] if g["family"] in exclusions),
        "prior_results": {"included_in_new_target": False, "expected_reference_episodes": 0,
                          "policy": "Fresh campaign only; no historical outcomes are merged."},
        "selection_policy": {
            "eval_seed": 0,
            "generalization_allocation": f"{states_per_task // 2} standard + {states_per_task // 2} random per paired family",
            "state_order": "First unique semantic-state occurrence in native layout enumeration",
            "invalid_layouts": "Native skip/replacement within the frozen unique candidate queue",
            "finite_diversity_exhaustion": "Report a shortfall; never repeat or synthesize states",
            "failed_scored_layouts": "Count once; never retry scored policy failures",
            "group_order": "Alphabetical family, standard before random",
            "semantic_hash_method": HASH_METHOD,
        },
        "source": copy.deepcopy(catalog["source"]),
        "catalog_provenance": copy.deepcopy(catalog["provenance"]),
        "action_budget_unit": catalog["action_budget_unit"],
        "episode_id_format": catalog["episode_id_format"],
        "layout_id_rule": catalog["layout_id_rule"],
        "episode_ids": ids, "episode_seeds": {eid: 0 for eid in ids}, "groups": groups,
    }


def validate_state_subset(manifest):
    """Reject any change beyond the explicit, supported selection parameters."""
    if not isinstance(manifest, dict) or manifest.get("portable_schema") != STATE_SUBSET_SCHEMA:
        raise ManifestError("unsupported configurable state-subset schema")
    if "per_family_target" not in manifest or "excluded_families" not in manifest:
        raise ManifestError("state subset is missing explicit selection parameters")
    expected = build_state_subset(manifest["per_family_target"], manifest["excluded_families"])
    if canonical_sha256(manifest) != canonical_sha256(expected):
        raise ManifestError("state subset differs from its pinned catalog and declared selection")
    return manifest
