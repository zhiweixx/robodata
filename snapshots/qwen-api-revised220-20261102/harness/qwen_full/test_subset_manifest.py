"""CPU-only subset scope, state identity, and frozen-native provenance tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from build_full_manifest import ManifestError
from build_subset_manifest import (build_subset_manifest, main, validate_subset_manifest,
                                   canonical_sha256, semantic_layout, HASH_METHOD)


def full_fixture():
    dimensions = {
        "generalization": ["stack_bowls"] + [f"gen_{i:02}" for i in range(11)],
        "open": ["general_pickup"] + [f"open_{i:02}" for i in range(7)],
        "precision": [f"precision_{i:02}" for i in range(8)],
        "long-horizon": ["play_tic_tac_toe"] + [f"long_{i:02}" for i in range(7)],
        "memory": ["cover_blocks"] + [f"memory_{i:02}" for i in range(5)],
    }
    groups = []
    for seed in (0, 1, 2):
        for dimension, families in dimensions.items():
            for family in families:
                for variant in (["standard", "random"] if dimension == "generalization" else ["standard"]):
                    task = family + ("_random" if variant == "random" else "")
                    target = 25 if dimension == "generalization" else 50
                    count = target + 5
                    gid = f"{task}_s{seed}"
                    layouts = [{"layout_id": i, "filename": f"{task}_{i*2}.json",
                                "filename_suffix": i*2,
                                "source_path": f"/assets/Eval_Layout/RoboDojo/arx_x5/{seed}/{task}_{i*2}.json"}
                               for i in range(count)]
                    groups.append({"group_id": gid, "task": task, "eval_seed": seed,
                                   "family": family, "dimension": dimension, "variant": variant,
                                   "step_limit": 200+len(task)*10, "target_episodes": target,
                                   "native_options": {"eval_nums": target, "robot_self_collision": True},
                                   "candidate_count": count, "candidate_layout_ids": list(range(count)),
                                   "episode_ids": [f"{gid}_l{i}" for i in range(count)],
                                   "layouts": layouts, "priority": 2})
    return {"benchmark": "RoboDojo", "embodiment": "arx_x5", "family_count": 42,
            "task_count": 54, "group_count": 162, "total_target_episodes": 6300,
            "eval_seeds": [0, 1, 2], "action_budget_unit": "native_control_actions",
            "episode_id_format": "{task}_s{eval_seed}_l{layout_id}",
            "layout_id_rule": "numeric suffix sort, enumeration ID",
            "source": {"robodojo_root": "/frozen/RoboDojo", "assets_root": "/assets",
                       "sha256": {"task/config": "a"*64}}, "groups": groups}


def evidence_fixture(full):
    groups = {}
    for g in full['groups']:
        rows = []
        for item in g['layouts']:
            i = item['layout_id']
            state = i % 6 if g['family'] == 'cover_blocks' else (0 if g['family'] == 'play_tic_tac_toe' else i)
            obj = {'task': g['task'], 'state': state, 'Room': {'default_ori': [1.0, 0.0, 0.0, 0.0]}}
            if g['family'] == 'play_tic_tac_toe' and i < 25:
                obj['Room']['default_rot'] = obj['Room'].pop('default_ori')
            rows.append({'layout_id': i, 'source_path': item['source_path'], 'bytes': 1,
                         'source_sha256': canonical_sha256(obj), 'canonical_json_sha256': canonical_sha256(obj),
                         'semantic_state_sha256': canonical_sha256(semantic_layout(obj))})
        groups[g['group_id']] = {'task': g['task'], 'family': g['family'], 'eval_seed': g['eval_seed'], 'layouts': rows}
    return {'schema_version': 1, 'source_full_manifest_canonical_sha256': canonical_sha256(full),
            'semantic_hash_method': HASH_METHOD, 'groups': groups}


class SubsetManifestTests(unittest.TestCase):
    def setUp(self):
        self.full = full_fixture()

    def build(self):
        return build_subset_manifest(self.full, content_evidence=evidence_fixture(self.full))

    def test_exact_family_scope_and_exclusions(self):
        result = self.build()
        self.assertEqual((result["family_count"], result["task_count"], result["group_count"]), (40, 51, 51))
        self.assertEqual(result["total_target_episodes"], 767)
        self.assertEqual(result["eval_seeds"], [0])
        self.assertFalse(result["is_official_full_evaluation"])
        self.assertEqual(result["overall_reference_target_episodes"], 807)
        self.assertEqual(result["excluded_task_configurations"], ["general_pickup", "stack_bowls", "stack_bowls_random"])
        family_targets = {}
        for group in result["groups"]:
            family_targets[group["family"]] = family_targets.get(group["family"], 0) + group["target_episodes"]
        self.assertEqual(set(family_targets.values()), {1, 6, 20})
        self.assertEqual(family_targets['cover_blocks'], 6)
        self.assertEqual(family_targets['play_tic_tac_toe'], 1)
        self.assertNotIn("general_pickup", family_targets)
        self.assertNotIn("stack_bowls", family_targets)

    def test_paired_families_split_10_10_not_20_per_configuration(self):
        result = self.build()
        paired = [g for g in result["groups"] if g["dimension"] == "generalization"]
        self.assertEqual(len(paired), 22)
        self.assertEqual(sum(g["target_episodes"] for g in paired), 220)
        self.assertEqual({g["target_episodes"] for g in paired}, {10})

    def test_native_budgets_options_and_all_candidates_preserved(self):
        originals = {g["group_id"]: g for g in self.full["groups"]}
        original_copy = copy.deepcopy(self.full)
        result = self.build()
        self.assertEqual(self.full, original_copy)
        for group in result["groups"]:
            original = originals[group["group_id"]]
            for key in ["step_limit", "native_options"]:
                self.assertEqual(group[key], original[key])
            for key in ['candidate_count', 'candidate_layout_ids', 'episode_ids', 'layouts']:
                self.assertEqual(group['native_candidate_pool'][key], original[key])
            self.assertGreaterEqual(group["candidate_count"], group["target_episodes"])
            self.assertEqual(group["native_target_episodes"], original["target_episodes"])
            if len(group['layouts']) > 1:
                self.assertEqual(group["layouts"][1]["layout_id"], 1)
                self.assertEqual(group["layouts"][1]["filename_suffix"], 2)
        self.assertEqual(len(result["episode_ids"]), len(set(result["episode_ids"])))

    def test_selection_does_not_depend_on_priorities_or_scores(self):
        before = self.build()
        self.full["groups"].reverse()
        self.full["prior_scores"] = {"gen_00_s0_l0": True}
        after = self.build()
        self.assertEqual(before["groups"], after["groups"])
        self.assertEqual(before["episode_ids"], after["episode_ids"])

    def test_source_scope_drift_rejected(self):
        for key, value in [("group_count", 161), ("family_count", 41), ("total_target_episodes", 6299)]:
            changed = copy.deepcopy(self.full)
            changed[key] = value
            with self.assertRaises(ManifestError):
                build_subset_manifest(changed, content_evidence=evidence_fixture(changed))

    def test_duplicate_candidate_identity_rejected(self):
        group = self.full["groups"][0]
        group["layouts"][1]["source_path"] = group["layouts"][0]["source_path"]
        with self.assertRaisesRegex(ManifestError, "repeated layout"):
            self.build()

    def test_runtime_manifest_budget_mutation_rejected(self):
        result = self.build()
        result["groups"][0]["step_limit"] += 1
        with self.assertRaisesRegex(ManifestError, "native group changed"):
            validate_subset_manifest(result)

    def test_runtime_manifest_target_or_seed_mutation_rejected(self):
        for field, value in [("target_episodes", 20), ("eval_seed", 1)]:
            result = self.build()
            paired = next(g for g in result["groups"] if g["dimension"] == "generalization")
            paired[field] = value
            with self.assertRaises(ManifestError):
                validate_subset_manifest(result)

    def test_cli_pins_exact_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "full.json"
            output = Path(directory) / "nested/subset.json"
            evidence_path = Path(directory) / 'evidence.json'
            raw = json.dumps(self.full, indent=1).encode()
            source.write_bytes(raw)
            evidence_path.write_text(json.dumps(evidence_fixture(self.full)))
            main(["--source-manifest", str(source), '--layout-evidence', str(evidence_path), "--output", str(output)])
            result = json.loads(output.read_text())
            self.assertEqual(result["source_full_manifest_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(result["source_full_manifest_path"], str(source.resolve()))
            self.assertEqual(result["source_full_manifest_hash_encoding"], "exact source-file bytes")
            validate_subset_manifest(result)

    def test_room_unused_rotation_metadata_not_a_new_state(self):
        a = {'Rigid': {'x': 1}, 'Room': {'default_rot': [1.0, 0.0, 0.0, 0.0]}}
        b = {'Rigid': {'x': 1}, 'Room': {'default_ori': [1.0, 0.0, 0.0, 0.0]}}
        self.assertNotEqual(canonical_sha256(a), canonical_sha256(b))
        self.assertEqual(canonical_sha256(semantic_layout(a)), canonical_sha256(semantic_layout(b)))
        result = self.build()
        tic = next(g for g in result['groups'] if g['family'] == 'play_tic_tac_toe')
        self.assertEqual(tic['candidate_layout_ids'], [0])
        self.assertEqual(len(set(tic['layout_canonical_json_sha256_by_id'].values())), 2)
        self.assertEqual(len(set(tic['layout_content_sha256_by_id'].values())), 1)

    def test_dedup_preserves_native_ids_with_gaps(self):
        evidence = evidence_fixture(self.full)
        rows = evidence['groups']['open_00_s0']['layouts']
        rows[1]['semantic_state_sha256'] = rows[0]['semantic_state_sha256']
        result = build_subset_manifest(self.full, content_evidence=evidence)
        group = next(g for g in result['groups'] if g['group_id'] == 'open_00_s0')
        self.assertEqual(group['candidate_layout_ids'][:3], [0, 2, 3])
        self.assertEqual(group['episode_ids'][1], 'open_00_s0_l2')
        self.assertEqual(group['layouts'][1]['filename_suffix'], 4)
        self.assertEqual(group['runtime_overrides']['eval_layout_ids'], group['candidate_layout_ids'])


if __name__ == "__main__":
    unittest.main()
