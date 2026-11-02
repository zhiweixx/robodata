"""Offline evidence fixtures; no model, simulator, scheduler or network."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import audit_subset_results as audit


class SubsetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.campaign = self.root / "aspire_validate/full_results/subset_fixture"
        self.campaign.mkdir(parents=True)
        # Maintain real dimensions/counts while avoiding any simulator imports.
        self.dimensions = {"generalization": ["stack_bowls"] + [f"gen{i}" for i in range(11)],
                           "open": ["general_pickup"] + [f"open{i}" for i in range(7)],
                           "precision": [f"precision{i}" for i in range(8)],
                           "long-horizon": ["play_tic_tac_toe"] + [f"long{i}" for i in range(7)],
                           "memory": ["cover_blocks"] + [f"memory{i}" for i in range(5)]}
        inventory = self.root / "RoboDojo/scripts/internal/task_inventory.py"
        inventory.parent.mkdir(parents=True)
        inventory.write_text("DIMENSION_TASKS = " + repr(self.dimensions))
        self.groups = []
        for dimension, names in self.dimensions.items():
            for family in names:
                if family in audit.EXCLUDED:
                    continue
                for task in ([family, family + "_random"] if dimension == "generalization" else [family]):
                    target = (10 if dimension == "generalization" else
                              audit.FAMILY_EXCEPTIONS.get(family, 20))
                    self.groups.append(self.group(task, family, dimension, target))
        self.manifest = {
            "protocol": "robodojo_20_states_per_family_subset", "is_official_full_evaluation": False,
            "excluded_families": sorted(audit.EXCLUDED), "groups": self.groups,
            "exception_family_targets": audit.FAMILY_EXCEPTIONS,
            "family_targets": {g["family"]: audit.FAMILY_EXCEPTIONS.get(g["family"], 20)
                               for g in self.groups},
            "selection_policy": {"semantic_hash_method": audit.SEMANTIC_HASH_METHOD},
            "group_count": len(self.groups), "total_target_episodes": 767,
            "total_candidate_episodes": sum(len(g["episode_ids"]) for g in self.groups),
            "action_budget_unit": "native_control_actions",
            "source": {"robodojo_root": str(self.root / "RoboDojo"),
                       "sha256": {"scripts/internal/task_inventory.py": audit.sha256(inventory)}}}
        self.write(self.campaign / "manifest.json", self.manifest)
        self.now = datetime(2026, 10, 26, 6, 0, tzinfo=timezone.utc)

    @staticmethod
    def group(task, family, dimension, target):
        gid = task + "_s0"
        candidates = list(range(target + (0 if family in audit.FAMILY_EXCEPTIONS else 3)))
        hashes = {str(i): hashlib.sha256(f"{task}/{i}".encode()).hexdigest() for i in candidates}
        return {"group_id": gid, "task": task, "family": family, "dimension": dimension,
                "eval_seed": 0, "step_limit": 200, "target_episodes": target,
                "candidate_layout_ids": candidates, "episode_ids": [f"{gid}_l{i}" for i in candidates],
                "native_candidate_pool": {"candidate_layout_ids": candidates},
                "layout_content_sha256_by_id": hashes,
                "layout_canonical_json_sha256_by_id": dict(hashes),
                "layout_source_sha256_by_id": dict(hashes)}

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def native(self, group, layouts, *, campaign=None, complete=False, evidence=False):
        campaign = campaign or self.campaign
        result = audit.full.expected_result_path(self.root, campaign.name, group)
        details = {str(i): {"layout_id": layout, "success": i % 3 == 0,
                           "score": 1.0 if i % 3 == 0 else 0.0}
                   for i, layout in enumerate(layouts)}
        self.write(result, {"eval_time": len(layouts), "details": details})
        self.write(campaign / "groups" / group["group_id"] / "group_report.json",
                   {"mode": "evaluation", "task": group["task"], "eval_seed": 0,
                    "target": group["target_episodes"], "scored": len(layouts), "complete": complete,
                    "native_run_id": campaign.name + "_" + group["group_id"],
                    "native_result_path": str(result)})
        if evidence:
            for index, detail in details.items():
                for camera in audit.full.CAMERAS:
                    (result.parent / f"episode_{int(index):07d}_cam_{camera}_{'success' if detail['success'] else 'fail'}.mp4").write_bytes(b"test video")
                eid = f"{group['group_id']}_l{detail['layout_id']}"
                self.write(campaign / "groups" / group["group_id"] / "episodes" / eid / "attempt_001/episode_report.json",
                    {"episode_id": eid, "task": group["task"], "layout_id": detail["layout_id"],
                     "eval_seed": 0, "status": "completed", "native_score_committed": True,
                     "step_limit": group["step_limit"], "limits": {
                         "program_count": None, "program_wall_seconds": None,
                         "output_tokens": "remaining native model context", "action_budget": "native task config"}})
        return result

    def run_audit(self, *, progress=True):
        return audit.audit_campaign(self.campaign, simroot=self.root, now=self.now,
                                    progress_only=progress, include_details=True)

    def test_partial_progress_has_custom_scope_and_no_false_completion(self):
        self.native(self.groups[0], [0, 2, 3])
        result = self.run_audit()
        self.assertEqual(result["scoring"]["scored"], 3)
        self.assertEqual(result["scoring"]["target"], 767)
        self.assertEqual(result["groups"]["target"], 51)
        self.assertFalse(result["campaign_complete"])
        self.assertFalse(result["official_leaderboard_eligible"])
        self.assertNotIn("official_scope", result)
        self.assertEqual(result["errors"], [])

    def test_duplicate_native_layout_is_rejected(self):
        self.native(self.groups[0], [0, 0])
        result = self.run_audit()
        self.assertEqual(result["scoring"]["scored"], 0)
        self.assertTrue(any("duplicate native scored layout" in e["message"] for e in result["errors"]))

    def test_rejects_unexpected_or_smoke_groups(self):
        (self.campaign / "groups/general_pickup_s0").mkdir(parents=True)
        result = self.run_audit()
        self.assertEqual(result["unexpected_group_directories"], ["general_pickup_s0"])
        self.assertTrue(result["errors"])

    def test_rejects_smoke_under_scored_group(self):
        group = self.groups[0]
        self.write(self.campaign / "groups" / group["group_id"] / "group_report.json", {"mode": "interface_smoke"})
        result = self.run_audit()
        self.assertTrue(any("non-evaluation/smoke" in e["message"] for e in result["errors"]))

    def test_family_balancing_cannot_be_replaced_by_same_total(self):
        changed = copy.deepcopy(self.manifest)
        changed["groups"][0]["target_episodes"] = 11
        changed["groups"][1]["target_episodes"] = 9
        with self.assertRaisesRegex(ValueError, "balanced across variants"):
            audit.validate_subset_manifest(changed)

    def test_seed_and_inventory_are_pinned(self):
        changed = copy.deepcopy(self.manifest)
        changed["groups"][0]["eval_seed"] = 1
        with self.assertRaisesRegex(ValueError, "seed0"):
            audit.validate_subset_manifest(changed)
        changed = copy.deepcopy(self.manifest)
        changed["source"]["sha256"]["scripts/internal/task_inventory.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "provenance"):
            audit.validate_subset_manifest(changed)

    def test_full_evidence_can_finish_767_without_claiming_official_full(self):
        for group in self.groups:
            self.native(group, list(range(group["target_episodes"])), complete=True, evidence=True)
        result = self.run_audit(progress=False)
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["campaign_complete"])
        self.assertEqual(result["scoring"]["scored"], 767)
        self.assertFalse(result["combined_807_summary"]["complete"])
        self.assertFalse(result["official_leaderboard_eligible"])

    def reuse_fixture(self):
        old = self.root / "aspire_validate/full_results/old_fixture"
        groups = [self.group("general_pickup", "general_pickup", "open", 50),
                  self.group("stack_bowls", "stack_bowls", "generalization", 25)]
        self.write(old / "manifest.json", {"groups": groups})
        selected = []
        for group in groups:
            result = self.native(group, list(range(group["target_episodes"])), campaign=old, complete=True, evidence=True)
            selected.append({"group_id": group["group_id"], "selected_native_indices": list(range(20)),
                             "native_result_sha256": audit.sha256(result)})
        reuse = {"schema_version": 1, "selection": audit.REUSE_SELECTION,
                 "source_campaign_dir": str(old), "groups": selected}
        self.write(self.campaign / "reused_results.json", reuse)
        return reuse

    def test_reused_first20_are_frozen_and_reported_separately(self):
        self.reuse_fixture()
        result = self.run_audit()
        self.assertEqual(result["scoring"]["scored"], 0)
        self.assertEqual(result["reused_results"]["scored"], 40)
        self.assertEqual(result["combined_807_summary"]["success"], 14)
        self.assertFalse(result["reused_results"]["evidence_audited"])
        self.assertTrue(self.run_audit(progress=False)["reused_results"]["evidence_audited"])

    def test_reuse_rejects_cherry_picked_indices_and_changed_native(self):
        reuse = self.reuse_fixture()
        reuse["groups"][0]["selected_native_indices"][-1] = 20
        self.write(self.campaign / "reused_results.json", reuse)
        result = self.run_audit()
        self.assertEqual(result["reused_results"]["scored"], 0)
        self.assertTrue(result["reused_results"]["errors"])
        reuse["groups"][0]["selected_native_indices"] = list(range(20))
        reuse["groups"][0]["native_result_sha256"] = "0" * 64
        self.write(self.campaign / "reused_results.json", reuse)
        self.assertTrue(self.run_audit()["reused_results"]["errors"])

    def test_duplicate_content_requires_excluding_later_native_id(self):
        changed = copy.deepcopy(self.manifest)
        group = changed["groups"][0]
        group["layout_content_sha256_by_id"]["1"] = group["layout_content_sha256_by_id"]["0"]
        with self.assertRaisesRegex(ValueError, "first-occurrence distinct"):
            audit.validate_subset_manifest(changed)

    def test_score_for_original_but_deduplicated_id_is_rejected(self):
        group = self.manifest["groups"][0]
        group["layout_content_sha256_by_id"]["1"] = group["layout_content_sha256_by_id"]["0"]
        group["candidate_layout_ids"] = [i for i in group["candidate_layout_ids"] if i != 1]
        group["episode_ids"] = [f"{group['group_id']}_l{i}" for i in group["candidate_layout_ids"]]
        self.manifest["total_candidate_episodes"] -= 1
        self.write(self.campaign / "manifest.json", self.manifest)
        self.native(group, [1])
        result = self.run_audit()
        self.assertEqual(result["scoring"]["scored"], 0)
        self.assertTrue(any("outside authorized candidates" in e["message"] for e in result["errors"]))

    def test_duplicate_initial_state_across_variants_is_rejected(self):
        changed = copy.deepcopy(self.manifest)
        standard, random = changed["groups"][:2]
        random["layout_content_sha256_by_id"]["0"] = standard["layout_content_sha256_by_id"]["0"]
        with self.assertRaisesRegex(ValueError, "duplicate initial state across task variants"):
            audit.validate_subset_manifest(changed)


if __name__ == "__main__":
    unittest.main()
