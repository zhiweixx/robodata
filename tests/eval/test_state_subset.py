"""Independent CPU contracts for configurable, outcome-independent state subsets."""
from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path

import pytest

from robodata.eval.robodojo import manifest as manifests
from robodata.eval.robodojo.state_subset import build_state_subset

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "experiments/manifests/robodojo_subset767.json"


@pytest.fixture(scope="module")
def thirty():
    return build_state_subset(states_per_task=30)


def test_thirty_states_per_family_has_expected_native_scope(thirty):
    assert manifests.validate_manifest(thirty) == thirty
    assert thirty["per_family_target"] == 30
    assert thirty["family_count"] == 42
    assert thirty["task_count"] == thirty["group_count"] == 54
    assert thirty["total_target_episodes"] == 1207
    assert thirty["total_candidate_episodes"] == 2565
    assert thirty["eval_seeds"] == [0]
    assert thirty["excluded_families"] == []
    assert thirty["is_official_full_evaluation"] is False
    assert thirty["exception_family_targets"] == {"cover_blocks": 6, "play_tic_tac_toe": 1}
    targets = thirty["family_targets"]
    assert targets["general_pickup"] == targets["stack_bowls"] == 30
    assert sum(targets.values()) == 1207
    assert {key for key, value in targets.items() if value < 30} == {"cover_blocks", "play_tic_tac_toe"}
    # New campaigns include no historical scores or the old 807-reference total.
    assert thirty.get("overall_reference_target_episodes") in (None, 1207)
    assert thirty.get("prior_results", {}).get("included_in_new_target", False) is False
    assert thirty.get("prior_results", {}).get("expected_reference_episodes", 0) == 0


def test_balanced_variants_preserve_native_action_and_count_contracts(thirty):
    legacy = manifests.load_manifest(LEGACY)
    legacy_groups = {g["group_id"]: g for g in legacy["groups"]}
    family_rows = defaultdict(list)
    for group in thirty["groups"]:
        family_rows[group["family"]].append(group)
        native = 25 if group["dimension"] == "generalization" else 50
        assert group["native_target_episodes"] == native
        assert group["native_options"]["eval_nums"] == native
        assert group["runtime_overrides"] == {
            "eval_nums": group["target_episodes"],
            "eval_layout_ids": group["candidate_layout_ids"],
        }
        if group["dimension"] == "generalization":
            assert group["target_episodes"] == 15
        if group["group_id"] in legacy_groups:
            prior = legacy_groups[group["group_id"]]
            for key in ("step_limit", "native_target_episodes", "native_options",
                        "candidate_layout_ids", "layout_content_sha256_by_id",
                        "layout_source_sha256_by_id", "layout_canonical_json_sha256_by_id"):
                assert group[key] == prior[key], (group["group_id"], key)
    for rows in family_rows.values():
        if rows[0]["dimension"] == "generalization":
            assert {row["variant"] for row in rows} == {"standard", "random"}
            assert len(rows) == 2
            assert sum(row["target_episodes"] for row in rows) == 30


def test_original_layout_ids_and_unique_replacement_pools_preserved(thirty):
    groups = {g["group_id"]: g for g in thirty["groups"]}
    assert groups["cover_blocks_s0"]["candidate_layout_ids"] == [0, 1, 2, 7, 12, 14]
    assert groups["play_tic_tac_toe_s0"]["candidate_layout_ids"] == [0]
    per_family_hashes = defaultdict(list)
    for group in thirty["groups"]:
        ids = group["candidate_layout_ids"]
        assert ids == sorted(set(ids))
        assert [row["layout_id"] for row in group["layouts"]] == ids
        assert group["episode_ids"] == [f"{group['group_id']}_l{layout}" for layout in ids]
        assert len(ids) >= group["target_episodes"]
        assert all(not Path(row["source_path"]).is_absolute() for row in group["layouts"])
        hashes = [group["layout_content_sha256_by_id"][str(layout)] for layout in ids]
        assert len(hashes) == len(set(hashes))
        per_family_hashes[group["family"]].extend(hashes)
    for hashes in per_family_hashes.values():
        assert len(hashes) == len(set(hashes))


def test_exclusions_generate_a_separate_1147_episode_plan():
    value = build_state_subset(states_per_task=30, exclude_families=("stack_bowls", "general_pickup"))
    assert manifests.validate_manifest(value) == value
    assert value["family_count"] == 40
    assert value["group_count"] == 51
    assert value["total_target_episodes"] == 1147
    assert value["total_candidate_episodes"] == 2425
    assert not {"stack_bowls", "general_pickup"} & set(value["family_targets"])


def test_legacy_767_identity_and_default_sharding_remain_supported(thirty):
    legacy = manifests.load_manifest(LEGACY)
    assert manifests.canonical_sha256(legacy) == manifests.PORTABLE_MANIFEST_SHA256
    assert legacy["total_target_episodes"] == 767
    assert len(manifests.plan_shards(legacy, None)) == 51
    assert len(manifests.plan_shards(thirty, None)) == 54
    for count in (1, 4, 16, 54):
        rows = manifests.plan_shards(thirty, count)
        owned = [group for row in rows for group in row["group_ids"]]
        assert len(owned) == len(set(owned)) == 54
        assert set(owned) == {group["group_id"] for group in thirty["groups"]}
    with pytest.raises(ValueError):
        manifests.plan_shards(thirty, 55)


@pytest.mark.parametrize("count", [None, True, False, 0, -2, 1, 3, 31, 51, 52, "30", 30.0])
def test_invalid_or_unbalanced_counts_rejected(count):
    with pytest.raises(ValueError):
        build_state_subset(states_per_task=count)


@pytest.mark.parametrize("excluded", [("not_a_task",), ("stack_bowls", "stack_bowls")])
def test_unknown_or_duplicate_exclusions_rejected(excluded):
    with pytest.raises(ValueError):
        build_state_subset(states_per_task=30, exclude_families=excluded)


def test_excluding_every_family_rejected(thirty):
    with pytest.raises(ValueError):
        build_state_subset(states_per_task=30, exclude_families=tuple(thirty["family_targets"]))


def test_native_budget_change_cannot_be_authorized_by_rehashing(thirty):
    value = copy.deepcopy(thirty)
    group = value["groups"][0]
    group["step_limit"] += 1
    group["source_group_sha256"] = manifests.canonical_sha256(manifests._native_group(group))
    # Even all downstream plan hashes could be recomputed by a caller. Validation
    # must still compare against the pinned native catalog rather than trust them.
    manifests.canonical_sha256(value)
    with pytest.raises(ValueError):
        manifests.validate_manifest(value)


@pytest.mark.parametrize("mutation", ["duplicate_layout", "source_escape", "changed_target", "changed_hash", "old_reference"])
def test_rehashed_selection_tampering_rejected(thirty, mutation):
    value = copy.deepcopy(thirty)
    group = value["groups"][0]
    if mutation == "duplicate_layout":
        group["candidate_layout_ids"][1] = group["candidate_layout_ids"][0]
    elif mutation == "source_escape":
        group["layouts"][0]["source_path"] = "../../escape.json"
    elif mutation == "changed_target":
        group["target_episodes"] -= 1
        group["runtime_overrides"]["eval_nums"] = group["target_episodes"]
        value["total_target_episodes"] -= 1
        value["family_targets"][group["family"]] -= 1
    elif mutation == "changed_hash":
        key = str(group["candidate_layout_ids"][0])
        group["layout_content_sha256_by_id"][key] = "a" * 64
    else:
        value["overall_reference_target_episodes"] = 807
    manifests.canonical_sha256(value)
    with pytest.raises(ValueError):
        manifests.validate_manifest(value)


def test_builds_are_deterministic_and_do_not_share_mutable_group_state():
    first = build_state_subset(states_per_task=30)
    baseline = manifests.canonical_sha256(first)
    first["groups"][0]["step_limit"] += 5
    first["groups"][0]["candidate_layout_ids"].clear()
    second = build_state_subset(states_per_task=30)
    assert manifests.canonical_sha256(second) == baseline
    assert manifests.validate_manifest(second) == second


def test_catalog_survives_campaign_snapshot_and_frozen_process(tmp_path, monkeypatch):
    import json
    import os
    import subprocess
    import sys
    from robodata.cli.main import plan_campaign

    monkeypatch.setenv("ROBODATA_DATA", str(tmp_path / "data"))
    generated = tmp_path / "thirty.json"
    generated.write_text(json.dumps(build_state_subset(30)))
    campaign = tmp_path / "campaign"
    result = plan_campaign(ROOT, generated, ROOT / "configs/clusters/rtx-slurm.example.json", campaign)
    assert result["groups"] == result["shards"] == 54
    source = campaign / "source"
    record = json.loads((campaign / "source_inventory.json").read_text())
    assert "src/robodata/eval/robodojo/catalogs/seed0.json" in record["files_sha256"]
    # A fresh child resolves only the frozen source tree, as a Slurm worker does.
    command = (
        "import json; from pathlib import Path; "
        "from robodata.eval.robodojo import manifest, state_subset; "
        "assert Path(state_subset.__file__).resolve().is_relative_to(Path.cwd()); "
        "m=manifest.load_manifest(Path.cwd().parent/'manifest.json'); "
        "assert m['total_target_episodes']==1207; "
        "print(json.dumps({'groups':len(manifest.plan_shards(m))}))"
    )
    env = dict(os.environ, PYTHONPATH=f"{source}/src:{source}/third_party/aspire/src",
               PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, "-B", "-c", command], cwd=source,
                               env=env, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout) == {"groups": 54}
