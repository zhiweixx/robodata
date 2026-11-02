"""Offline audit fixtures only: no GPU, simulator, model, network or Slurm."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

import audit_full_results as audit
import official_summary_plan as summary_plan


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.simroot = Path(self.tmp.name)
        self.campaign = self.simroot / "aspire_validate/full_results/offline_campaign"
        self.campaign.mkdir(parents=True)
        self.now = datetime(2026, 10, 24, 12, 10, tzinfo=timezone.utc)
        self.groups = []
        for task, seed, category, budget in [
                ("general_pickup", 0, "generalization", 200),
                ("general_pickup_random", 0, "generalization", 200),
                ("stack_bowls", 1, "precision", 800),
                ("stack_bowls", 2, "precision", 800)]:
            gid = f"{task}_s{seed}"
            self.groups.append({"group_id": gid, "task": task, "eval_seed": seed,
                                "dimension": category, "family": task.removesuffix("_random"),
                                "step_limit": budget, "target_episodes": 2,
                                "candidate_layout_ids": list(range(4)),
                                "episode_ids": [f"{gid}_l{i}" for i in range(4)]})
        self.manifest = {"groups": self.groups, "group_count": 4, "total_target_episodes": 8,
                         "total_candidate_episodes": 16, "action_budget_unit": "native_control_actions"}
        self.write(self.campaign / "manifest.json", self.manifest)
        self.native(0, [(0, False, .25), (3, True, 1.0)], complete=True)
        self.native(1, [(1, False, .5)], complete=False)
        self.episode(2, 0, status="interrupted", committed=False)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def report_path(self, index):
        return self.campaign / "groups" / self.groups[index]["group_id"] / "group_report.json"

    def native(self, index, rows, *, complete):
        group = self.groups[index]
        result = audit.expected_result_path(self.simroot, self.campaign.name, group)
        details = {str(i): {"layout_id": layout, "success": success, "score": score}
                   for i, (layout, success, score) in enumerate(rows)}
        self.write(result, {"eval_time": len(rows), "details": details})
        self.write(self.report_path(index), {
            "mode": "evaluation", "task": group["task"], "eval_seed": group["eval_seed"],
            "native_run_id": f"{self.campaign.name}_{group['group_id']}",
            "native_result_path": "/work/" + str(result.relative_to(self.simroot)),
            "target": 2, "scored": len(rows), "complete": complete})
        for i, (layout, success, _) in enumerate(rows):
            for camera in audit.CAMERAS:
                (result.parent / f"episode_{i:07d}_cam_{camera}_{'success' if success else 'fail'}.mp4").write_bytes(b"fixture")
            self.episode(index, layout)
        return result

    def episode(self, index, layout, *, status="completed", committed=True, attempt=1, budget=None):
        group = self.groups[index]
        eid = f"{group['group_id']}_l{layout}"
        path = self.campaign / "groups" / group["group_id"] / "episodes" / eid / f"attempt_{attempt:03d}" / "episode_report.json"
        self.write(path, {"episode_id": eid, "task": group["task"], "layout_id": layout,
                          "eval_seed": group["eval_seed"], "status": status,
                          "native_score_committed": committed,
                          "limits": {"program_count": None, "program_wall_seconds": None,
                                     "output_tokens": "remaining native model context",
                                     "action_budget": "native task config"},
                          "step_limit": group["step_limit"] if budget is None else budget})
        return path

    def allocation(self, name="123", *, closed=True):
        directory = self.campaign / "allocations" / name
        self.write(directory / "allocation.json", {
            "campaign": self.campaign.name, "allocation": name, "mode": "resume", "complete": False,
            "started_utc": "2026-10-24T12:00:00+00:00",
            **({"finished_utc": "2026-10-24T12:05:00+00:00"} if closed else {})})
        return directory

    def usage(self, directory, index=0, *, smoke=False, scored=False, extra=None):
        group = self.groups[index]
        ids = ["interface_smoke"] if smoke else group["episode_ids"]
        seed = 0 if smoke else group["eval_seed"]
        evaluation_root = directory / "scored" if scored else directory
        path = (directory / "model_smoke/full_model_smoke.usage.jsonl" if smoke else
                evaluation_root / "worker0" / group["group_id"] / "usage.jsonl")
        common = {"episode_id": ids[0], "attempt": 1, "iteration": 1}
        rows = [{"event": "suite_initialized", "config": {
            "purpose": "aspire_robodojo_full_local_qwen", "model": audit.MODEL,
            "episode_ids": ids, "episode_seeds": dict.fromkeys(ids, seed), "remote_api_calls": False,
            **audit.UNCAPPED_CONFIG}},
            {**common, "event": "input_token_count", "wall_s": .2},
            {**common, "event": "generation_started"},
            {**common, "event": "raw_response", "generation_wall_s": 5},
            {**common, "event": "generation_usage", "usage": {
                "prompt_tokens": 100, "completion_tokens": 40, "generation_wall_s": 5,
                "input_count_wall_s": .2, "api_cost_usd": 0}},
            # Cumulative snapshots must never be counted again.
            {**common, "event": "episode_closed", "budget": {"prompt_tokens": 100}},
        ]
        rows += extra or []
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def run_audit(self):
        return audit.audit_campaign(self.campaign, simroot=self.simroot, now=self.now, include_details=True)

    def test_counts_complete_partial_unscored_and_unstarted_separately(self):
        with patch.object(audit, "validate_group", wraps=audit.validate_group) as validate:
            result = self.run_audit()
        validate.assert_called_once()
        self.assertEqual(result["groups"]["states"], {"complete": 1, "partial": 1, "unscored": 1, "not_started": 1})
        self.assertEqual(result["scoring"]["scored"], 3)
        self.assertEqual(result["scoring"]["success"], 1)
        self.assertEqual(result["scoring"]["score_sum"], 1.75)
        self.assertEqual(result["scoring"]["scored_in_verified_complete_groups"], 2)
        self.assertEqual(result["scoring"]["scored_in_other_groups"], 1)
        self.assertEqual(result["scoring"]["remaining_to_target"], 5)
        self.assertAlmostEqual(result["scoring"]["success_rate"], 1 / 3)
        self.assertIsNone(result["by_seed"][2]["success_rate"])
        self.assertEqual(result["by_family"][0]["scored"], 3)
        self.assertEqual(result["by_family"][0]["target"], 4)
        self.assertEqual(result["by_category_family_seed"][0]["scored"], 3)
        self.assertFalse(result["campaign_complete"])
        self.assertFalse(result["official_scope"]["matches"])
        self.assertEqual(result["errors"], [])

    def test_native_authority_rejects_duplicate_layout_even_if_report_claims_complete(self):
        path = audit.expected_result_path(self.simroot, self.campaign.name, self.groups[0])
        native = audit.read_json(path)
        native["details"]["1"]["layout_id"] = 0
        self.write(path, native)
        result = self.run_audit()
        self.assertEqual(result["scoring"]["scored"], 1)
        self.assertEqual(result["groups"]["native_complete_verified"], 0)
        self.assertIn("invalid_evidence", result["groups"]["states"])
        self.assertTrue(result["errors"])

    def test_smoke_group_is_excluded_even_with_scored_native_file(self):
        path = self.report_path(0)
        report = audit.read_json(path)
        report["mode"] = "interface_smoke"
        self.write(path, report)
        result = self.run_audit()
        self.assertEqual(result["scoring"]["scored"], 1)
        self.assertEqual(result["groups"]["states"]["excluded_non_evaluation"], 1)
        self.assertFalse(result["campaign_complete"])

    def test_native_partial_can_advance_past_a_stale_group_report(self):
        path = self.report_path(1)
        report = audit.read_json(path)
        report["scored"] = 0
        self.write(path, report)
        result = self.run_audit()
        self.assertEqual(result["scoring"]["scored"], 3)
        self.assertTrue(any("counts differ" in row["message"] for row in result["errors"]))

    def test_missing_wrist_does_not_pass_on_video_count(self):
        directory = audit.expected_result_path(self.simroot, self.campaign.name, self.groups[0]).parent
        path = directory / "episode_0000000_cam_left_wrist_fail.mp4"
        path.rename(directory / "episode_0000000_cam_head_success.mp4")
        result = self.run_audit()
        self.assertEqual(result["scoring"]["scored"], 3)
        self.assertEqual(result["groups"]["native_complete_verified"], 0)
        self.assertEqual(result["evidence"]["scored_episodes_with_three_matching_nonempty_videos"], 2)
        self.assertFalse(result["campaign_complete"])

    def test_committed_episode_report_requires_native_action_budget(self):
        self.episode(0, 0, budget=8)
        result = self.run_audit()
        self.assertEqual(result["evidence"]["scored_episodes_with_committed_report_and_native_budget"], 2)
        self.assertEqual(result["native_action_budgets"]["by_task"]["general_pickup"], 200)
        self.assertTrue(any("budget" in row["message"] for row in result["errors"]))

    def test_complete_gate_requires_every_native_video_and_episode_receipt(self):
        self.native(1, [(1, False, .5), (2, True, 1)], complete=True)
        self.native(2, [(0, False, 0), (1, False, 0)], complete=True)
        self.native(3, [(0, False, 0), (1, True, 1)], complete=True)
        observed = self.run_audit()["official_scope"]["observed"]
        with patch.object(audit, "OFFICIAL", observed):
            self.assertTrue(self.run_audit()["campaign_complete"])
            self.episode(3, 1, committed=False)
            self.assertFalse(self.run_audit()["campaign_complete"])

    def test_allocation_time_and_usage_deduplicate_receipts_and_separate_smoke(self):
        closed = self.allocation()
        open_allocation = self.allocation("124", closed=False)
        eid = self.groups[0]["episode_ids"][1]
        key = {"episode_id": eid, "attempt": 1, "iteration": 1}
        self.usage(closed, extra=[
            {**key, "event": "generation_started"},
            {**key, "event": "generation_transport_failure", "generation_wall_s": 3},
            {**key, "event": "infrastructure_stopped"},
            {**key, "event": "unscored_attempt_abandoned", "reason": "infrastructure_interrupted"},
        ])
        self.usage(open_allocation)
        self.usage(closed, smoke=True)
        result = self.run_audit()
        total = result["usage"]["evaluation"]["total"]
        self.assertEqual(total["prompt_tokens"], 200)
        self.assertEqual(total["completion_tokens_including_reasoning"], 80)
        self.assertEqual(total["model_calls_started"], 3)
        self.assertEqual(total["model_calls_with_usage"], 2)
        self.assertEqual(total["calls_without_final_token_usage"], 1)
        self.assertEqual(total["explicit_infrastructure_interrupted_attempts"], 1)
        self.assertEqual(total["generation_wall_seconds"], 13)
        self.assertEqual(total["tokenization_wall_seconds"], .4)
        self.assertEqual(result["usage"]["model_smoke"]["total"]["prompt_tokens"], 100)
        self.assertEqual(result["usage"]["all_recorded_model_work"]["prompt_tokens"], 300)
        self.assertEqual(result["allocation_time"]["closed_recorded_seconds"], 300)
        self.assertEqual(result["allocation_time"]["unfinished_record_age_seconds"], 600)
        self.assertFalse(result["allocation_time"]["scheduler_elapsed_verified"])
        self.assertEqual(result["scoring"]["scored"], 3)
        self.assertEqual(result["api_cost_usd"], 0)
        self.assertIsNone(result["eta"])
        self.assertEqual(result["errors"], [])

    def test_partial_last_jsonl_line_is_flagged_without_losing_prior_usage(self):
        path = self.usage(self.allocation())
        with path.open("a") as handle:
            handle.write('{"event":')
        result = self.run_audit()
        self.assertEqual(result["usage"]["evaluation"]["total"]["prompt_tokens"], 100)
        self.assertTrue(any("incomplete line" in row["message"] for row in result["errors"]))

    def test_new_scored_and_legacy_allocation_paths_both_contribute_usage(self):
        self.usage(self.allocation("old"))
        self.usage(self.allocation("new"), scored=True)
        result = self.run_audit()
        totals = result["usage"]["evaluation"]["total"]
        self.assertEqual(totals["ledgers"], 2)
        self.assertEqual(totals["prompt_tokens"], 200)
        self.assertEqual(totals["generation_wall_seconds"], 10)
        self.assertEqual(result["errors"], [])

    def test_new_and_legacy_copies_of_one_ledger_are_counted_once(self):
        directory = self.allocation()
        preferred = self.usage(directory, scored=True)
        legacy = self.usage(directory)
        self.assertEqual(preferred.read_bytes(), legacy.read_bytes())
        result = self.run_audit()
        totals = result["usage"]["evaluation"]["total"]
        self.assertEqual(totals["ledgers"], 1)
        self.assertEqual(totals["prompt_tokens"], 100)
        aliases = result["allocations"][0]["usage_ledger_aliases_excluded"]
        self.assertEqual(aliases, [{"included": str(preferred.resolve()), "excluded": str(legacy.resolve()),
                                    "reason": "identical_legacy_copy"}])
        self.assertEqual(result["errors"], [])

    def test_legacy_symlink_to_scored_ledger_is_counted_once(self):
        directory = self.allocation()
        preferred = self.usage(directory, scored=True)
        legacy = directory / "worker0" / self.groups[0]["group_id"] / "usage.jsonl"
        legacy.parent.mkdir(parents=True)
        legacy.symlink_to(preferred)
        result = self.run_audit()
        self.assertEqual(result["usage"]["evaluation"]["total"]["ledgers"], 1)
        self.assertEqual(result["allocations"][0]["usage_ledger_aliases_excluded"][0]["reason"], "same_file")
        self.assertEqual(result["errors"], [])

    def test_conflicting_legacy_copy_is_explicit_error_and_not_double_counted(self):
        directory = self.allocation()
        preferred = self.usage(directory, scored=True)
        self.usage(directory)
        rows = [json.loads(line) for line in preferred.read_text().splitlines()]
        rows[4]["usage"]["prompt_tokens"] = 200
        preferred.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = self.run_audit()
        self.assertEqual(result["usage"]["evaluation"]["total"]["ledgers"], 1)
        self.assertEqual(result["usage"]["evaluation"]["total"]["prompt_tokens"], 200)
        self.assertTrue(any("ledgers conflict" in row["message"] for row in result["errors"]))
        self.assertFalse(result["campaign_complete"])

    def test_duplicate_usage_or_wrong_model_cannot_silently_double_count(self):
        path = self.usage(self.allocation())
        original = path.read_text()
        rows = [json.loads(line) for line in original.splitlines()]
        with path.open("a") as handle:
            handle.write(json.dumps(rows[4]) + "\n")
        result = self.run_audit()
        self.assertEqual(result["usage"]["evaluation"]["total"]["ledgers"], 0)
        self.assertTrue(any("duplicate generation_usage" in row["message"] for row in result["errors"]))
        rows[0]["config"]["model"] = "wrong_model"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assertEqual(self.run_audit()["usage"]["evaluation"]["total"]["ledgers"], 0)

    def test_manifest_count_mismatch_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        broken["total_target_episodes"] = 6300
        self.write(self.campaign / "manifest.json", broken)
        with self.assertRaisesRegex(ValueError, "aggregate counts"):
            self.run_audit()

    def test_capped_driver_ledger_is_excluded(self):
        path = self.usage(self.allocation())
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["config"]["max_iterations"] = 8
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = self.run_audit()
        self.assertEqual(result["usage"]["evaluation"]["total"]["ledgers"], 0)
        self.assertTrue(result["errors"])

    def test_cross_campaign_native_path_is_rejected(self):
        path = self.report_path(1)
        report = audit.read_json(path)
        report["native_result_path"] = report["native_result_path"].replace("offline_campaign_", "other_campaign_")
        self.write(path, report)
        result = self.run_audit()
        self.assertEqual(result["scoring"]["scored"], 2)
        self.assertEqual(result["groups"]["states"]["invalid_evidence"], 1)

    def test_audit_is_read_only_and_json_serializable(self):
        self.usage(self.allocation())
        def snapshot():
            return {str(p.relative_to(self.simroot)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in self.simroot.rglob("*") if p.is_file()}
        before = snapshot()
        result = self.run_audit()
        json.dumps(result, allow_nan=False)
        self.assertEqual(snapshot(), before)

    def test_progress_only_reads_json_snapshots_and_skips_all_heavy_evidence(self):
        self.usage(self.allocation())
        original_read = audit.read_json
        allowed = {"manifest.json", "group_report.json", "group_progress.json", "_result.json"}
        reads = []
        def snapshot_read(path):
            self.assertIn(Path(path).name, allowed)
            reads.append(Path(path))
            return original_read(path)
        with patch.object(audit, "read_json", side_effect=snapshot_read), \
                patch.object(audit, "audit_allocations", side_effect=AssertionError("allocation/ledger read")), \
                patch.object(audit, "validate_group", side_effect=AssertionError("full evidence validation")), \
                patch.object(audit, "video_evidence", side_effect=AssertionError("video read")), \
                patch.object(Path, "glob", side_effect=AssertionError("recursive/tree scan")):
            result = audit.audit_campaign(self.campaign, simroot=self.simroot, now=self.now,
                                          include_details=True, progress_only=True)
        self.assertEqual(result["scoring"]["scored"], 3)
        self.assertEqual(result["scoring"]["success"], 1)
        self.assertEqual(result["scoring"]["score_sum"], 1.75)
        self.assertEqual(result["by_category"][0]["scored"], 3)
        self.assertEqual(result["groups"]["native_scores_at_target"], 1)
        self.assertEqual(result["groups"]["native_complete_verified"], 0)
        self.assertEqual(result["groups"]["states"]["no_native_score_snapshot"], 2)
        self.assertFalse(result["campaign_complete"])
        self.assertTrue(result["final_evidence_audit_required"])
        self.assertFalse(result["usage_audited"])
        self.assertFalse(result["evidence_audited"])
        self.assertFalse(result["scheduler_liveness_verified"])
        self.assertNotIn("usage", result)
        self.assertTrue(reads)
        self.assertEqual(result["errors"], [])

    def test_progress_only_accepts_group_progress_snapshot_without_report(self):
        report = self.report_path(0)
        report.rename(report.with_name("group_progress.json"))
        result = audit.audit_campaign(self.campaign, simroot=self.simroot, now=self.now,
                                      progress_only=True)
        self.assertEqual(result["scoring"]["scored"], 3)
        self.assertEqual(result["groups"]["reported_complete_claims"], 1)
        self.assertFalse(result["campaign_complete"])
        self.assertEqual(result["errors"], [])

    def test_progress_only_never_declares_completion_even_if_every_native_score_exists(self):
        self.native(1, [(1, False, .5), (2, True, 1)], complete=True)
        self.native(2, [(0, False, 0), (1, False, 0)], complete=True)
        self.native(3, [(0, False, 0), (1, True, 1)], complete=True)
        observed = self.run_audit()["official_scope"]["observed"]
        with patch.object(audit, "OFFICIAL", observed):
            result = audit.audit_campaign(self.campaign, simroot=self.simroot, now=self.now,
                                          progress_only=True)
        self.assertTrue(result["official_scope"]["matches"])
        self.assertEqual(result["scoring"]["remaining_to_target"], 0)
        self.assertEqual(result["groups"]["native_scores_at_target"], 4)
        self.assertFalse(result["campaign_complete"])
        self.assertTrue(result["final_evidence_audit_required"])


class OfficialSummaryPlanTests(unittest.TestCase):
    """Synthetic completed-audit fixtures; never run the official summarizer."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.simroot = Path(self.tmp.name).resolve()
        self.campaign = self.simroot / "aspire_validate/full_results/fixture_campaign"
        self.campaign.mkdir(parents=True)
        self.summarizer = Path(__file__).resolve().parents[2] / "audit/RoboDojo/scripts/internal/summarize_result.py"
        dimensions = summary_plan.literal(self.summarizer.read_text(), "DIMENSIONS")
        tasks = {task for values in dimensions.values() for task in values}
        tasks |= {task + "_random" for task in dimensions["Generalization"]}
        self.native_paths = {}
        for task in tasks:
            for seed in (0, 1, 2):
                gid = f"{task}_s{seed}"
                native = audit.expected_result_path(self.simroot, self.campaign.name, {
                    "task": task, "eval_seed": seed, "group_id": gid})
                native.parent.mkdir(parents=True)
                native.write_text("{}")
                # Raw-tree smoke folders deliberately sort after this campaign.
                smoke = native.parent.parent / "zz_old_smoke"
                smoke.mkdir()
                (smoke / "_result.json").write_text("{}")
                self.native_paths[gid] = str(native.parent)
        self.report = {"campaign": self.campaign.name, "campaign_dir": str(self.campaign),
                       "campaign_complete": True, "errors": [], "usage": {},
                       "official_scope": {"matches": True, "expected": audit.OFFICIAL, "observed": audit.OFFICIAL},
                       "groups": {"target": 162, "native_complete_verified": 162, "states": {"complete": 162}},
                       "scoring": {"target": 6300, "scored": 6300,
                                   "scored_in_verified_complete_groups": 6300, "remaining_to_target": 0},
                       "evidence": {"scored_episodes_with_three_matching_nonempty_videos": 6300,
                                    "scored_episodes_with_committed_report_and_native_budget": 6300},
                       "official_summarizer_input": {"verified_native_group_directories": self.native_paths}}
        self.audit_path = self.campaign / "fixture_audit.json"
        self.audit_path.write_text(json.dumps(self.report))
        self.view = self.campaign / "official summary view"

    def plan(self):
        return summary_plan.plan_summary(self.audit_path, self.view, self.summarizer,
                                         python="/some python path/python3")

    def test_default_plan_is_read_only_and_emits_exact_quoted_command(self):
        before = {str(p) for p in self.simroot.rglob("*")}
        plan = self.plan()
        self.assertEqual(plan["mode"], "dry_run")
        self.assertEqual(plan["group_count"], 162)
        self.assertEqual({str(p) for p in self.simroot.rglob("*")}, before)
        self.assertFalse(self.view.exists())
        self.assertEqual(shlex.split(plan["official_summary_command"]), [
            "env", f"ROBODOJO_EVAL_ROOT={self.view}", "/some python path/python3", "-B", str(self.summarizer)])
        self.assertEqual(plan["output_markdown"], str(self.view / "_summary.md"))

    def test_created_view_has_only_exact_campaign_dirs_in_native_hierarchy(self):
        plan = summary_plan.create_view(self.plan())
        self.assertEqual(plan["mode"], "view_created")
        for row in plan["links"]:
            destination, source = Path(row["link"]), Path(row["source"])
            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.resolve(), source.resolve())
            self.assertEqual(list(destination.parent.iterdir()), [destination])
            self.assertTrue((source.parent / "zz_old_smoke/_result.json").is_file())
            self.assertEqual(destination.relative_to(self.view).parts[-4:-1],
                             ("ASPIRE_Pilot", "arx_x5", source.parent.name))
        self.assertFalse((self.view / "_summary.md").exists())

    def test_progress_incomplete_or_missing_camera_audit_is_rejected(self):
        for path, value in [(('campaign_complete',), False), (('audit_mode',), 'progress_only'),
                            (('scoring', 'scored'), 6299),
                            (('groups', 'native_complete_verified'), 161),
                            (('evidence', 'scored_episodes_with_three_matching_nonempty_videos'), 6299)]:
            with self.subTest(path=path):
                broken = copy.deepcopy(self.report)
                owner = broken
                for key in path[:-1]:
                    owner = owner[key]
                owner[path[-1]] = value
                self.audit_path.write_text(json.dumps(broken))
                with self.assertRaisesRegex(ValueError, 'successful full evidence audit'):
                    self.plan()
                self.assertFalse(self.view.exists())

    def test_smoke_mapping_or_existing_view_is_rejected(self):
        gid = next(iter(self.native_paths))
        smoke = Path(self.native_paths[gid]).parent / "zz_old_smoke"
        broken = copy.deepcopy(self.report)
        broken["official_summarizer_input"]["verified_native_group_directories"][gid] = str(smoke)
        self.audit_path.write_text(json.dumps(broken))
        with self.assertRaisesRegex(ValueError, 'unexpected campaign/group source'):
            self.plan()
        self.audit_path.write_text(json.dumps(self.report))
        self.view.mkdir()
        with self.assertRaisesRegex(ValueError, 'must be a new directory'):
            self.plan()


if __name__ == "__main__":
    unittest.main()
