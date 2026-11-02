"""Synthetic JSON reporting tests only; no model, GPU, network, or Slurm."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import mtp_trial_analysis as analysis


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        trial = {"purpose": "isolated_performance_test_not_robot_evaluation", "job_id": "fixture",
                 "native_context": 262144, "target_dtype": "bfloat16", "test_only_output_cap": 1024,
                 "changes_to_evaluation_budgets": False, "simulation_running": False,
                 "finished_utc": "2026-10-25T00:00:00Z", "variants": {}}
        for variant, speed in (("bf16", 1), ("mtp1", 2), ("mtp3", 3)):
            trial["variants"][variant] = {"ok": True, "finished_utc": "2026-10-25T00:00:00Z"}
            self.write(self.root / variant / "vision_schema_smoke.json", {
                "ok": True, "reported_colors": ["red", "green", "blue"],
                "expected_context_length": 262144, "measured_episode": False})
            spec = "None" if variant == "bf16" else f"SpeculativeConfig(method='mtp', num_spec_tokens={variant[-1]})"
            server = {"startup_args": {"max_model_len": 262144, "dtype": "bfloat16",
                                       "tensor_parallel_size": 2, "max_num_seqs": 2},
                      "engine_config": f"speculative_config={spec}, tokenizer='/weights', quantization=None"}
            for directory, labels, concurrency in (
                    ("concurrency1", ("short", "medium", "long"), 1),
                    ("concurrency2", ("short", "medium"), 2),
                    ("concurrency2_long", ("medium", "long"), 2)):
                cases = [self.case(name) for name in labels]
                manifest = {"purpose": "isolated_performance_replay_not_episode", "variant": variant,
                            "concurrency": concurrency, "test_only_max_output_tokens": 1024,
                            "production_modified": False, "code_execution": False,
                            "repeats": 2, "warmups": 1, "server": server, "cases": cases,
                            "case_files": [f"/fixture/{name}.json" for name in labels]}
                self.write(self.root / variant / directory / "manifest.json", manifest)
                groups = [[case] for case in cases] if concurrency == 1 else [cases]
                for group, selected in enumerate(groups):
                    for repeat in (0, 1, 2):
                        rate = 1000 if repeat == 0 else (10 * repeat * speed)
                        wall = len(selected) * 100 / rate
                        requests = []
                        for worker, case in enumerate(selected):
                            requests.append({"worker": worker, "repeat": repeat, "warmup": repeat == 0,
                                "source_request_sha256": case["source"]["request_sha256"],
                                "case_source": case["source"], "prompt_tokens": case["prompt_tokens"],
                                "image_count": case["image_count"], "requested_output_cap": 1024,
                                "ok": True, "finish_reason": "stop", "final_json_schema_valid": True,
                                "generated_code_executed": False, "output_tokens": 100, "wall_s": .75 * wall,
                                "observed_ttft_s": .01 if repeat == 0 else repeat * .1,
                                "usage": {"prompt_tokens": case["prompt_tokens"], "completion_tokens": 100},
                                "final_content": "DO_NOT_INCLUDE_GENERATED_PROGRAM", "reasoning_text": "DO_NOT_INCLUDE_REASONING"})
                        draft, accepted = ((None, None) if variant == "bf16" else
                                           (1000, 900) if repeat == 0 else (100, 50) if repeat == 1 else (300, 270))
                        batch = {"case_group": group, "repeat": repeat, "warmup": repeat == 0,
                                 "concurrency": concurrency, "batch_wall_s": wall, "requests": requests,
                                 "aggregate_output_tokens_per_second": rate,
                                 "metrics_generation_accounting_caught_up": True,
                                 "speculation": {"draft_tokens": draft, "accepted_tokens": accepted,
                                                 "acceptance_counters_valid": variant != "bf16"},
                                 "gpu_memory_samples": [{"returncode": 0,
                                     "csv": "GPU-one, 0, L40S, 40000, 46080, 100, 200\nGPU-two, 1, L40S, 41000, 46080, 100, 210"}]}
                        self.write(self.root / variant / directory / f"batch_{group:02d}_{repeat:03d}.json", batch)
        self.write(self.root / "trial.json", trial)

    @staticmethod
    def case(name):
        return {"source": {"request_sha256": hashlib.sha256(name.encode()).hexdigest()},
                "prompt_tokens": {"short": 6863, "medium": 31570, "long": 201714}[name],
                "image_count": {"short": 3, "medium": 18, "long": 6}[name]}

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def change(self, relative, update):
        path = self.root / relative
        value = json.loads(path.read_text())
        update(value)
        self.write(path, value)

    def condition(self, report, variant="mtp1", name="c1_short"):
        return report["variants"][variant]["conditions"][name]

    def test_all_five_conditions_separate_and_warmups_excluded(self):
        report = analysis.analyze_trial(self.root)
        self.assertEqual(report["eligible_mtp_variants_for_review"], ["mtp1", "mtp3"])
        baseline = self.condition(report, "bf16")
        self.assertEqual([r["batch_tokens_per_second"] for r in baseline["measured_repeats"]], [10, 20])
        self.assertEqual(baseline["batch_rate_stats"]["mean"], 15)
        self.assertEqual(baseline["batch_rate_stats"]["median"], 15)
        self.assertAlmostEqual(baseline["pooled_batch_tokens_per_second"], 200 / 15)
        self.assertEqual(baseline["warmup_ttft_s"], [{"repeat": 0, "request_ttft_s": [.01]}])
        self.assertFalse(baseline["warmup_cache_cold_verified"])
        self.assertEqual(len(report["variants"]["bf16"]["conditions"]), 5)
        self.assertEqual(self.condition(report, "bf16", "c1_medium")["comparison_signature"][0]["prompt_tokens"], 31570)
        self.assertEqual(self.condition(report, "bf16", "c1_long")["comparison_signature"][0]["prompt_tokens"], 201714)
        self.assertIsNone(report["selected_variant"])
        self.assertFalse(report["automatic_selection"])

    def test_concurrent_batch_and_request_rates_use_different_denominators(self):
        report = analysis.analyze_trial(self.root)
        condition = self.condition(report, "bf16", "c2_short_medium")
        self.assertAlmostEqual(condition["pooled_batch_tokens_per_second"], 400 / 30)
        self.assertAlmostEqual(condition["request_pooled_tokens_per_second"], 400 / 45)
        self.assertEqual(set(condition["measured_ttft_by_case"]), {"short", "medium"})
        self.assertEqual(condition["measured_ttft_s"]["n"], 4)
        self.assertAlmostEqual(self.condition(report, "mtp1", "c2_short_medium")["vs_bf16"]["batch_throughput_speedup"], 2)

    def test_acceptance_is_measured_pooled_counts_and_gpu_csv_is_parsed(self):
        condition = self.condition(analysis.analyze_trial(self.root))
        counters = condition["measured_speculation"]
        self.assertEqual((counters["draft_tokens"], counters["accepted_tokens"]), (400, 320))
        self.assertEqual(counters["acceptance_ratio"], .8)
        self.assertEqual(condition["gpu_memory"]["max_any_gpu_mib"], 41000)
        self.assertEqual(condition["gpu_memory"]["max_simultaneous_sum_mib"], 81000)

    def test_warmup_mtp_activity_cannot_substitute_for_measured_activity(self):
        for repeat in (1, 2):
            self.change(f"mtp1/concurrency1/batch_00_{repeat:03d}.json",
                        lambda batch: batch.update(speculation={"draft_tokens": 0, "accepted_tokens": 0,
                                                               "acceptance_counters_valid": True}))
        report = analysis.analyze_trial(self.root)
        self.assertFalse(report["variants"]["mtp1"]["eligible_for_review"])
        self.assertEqual(self.condition(report)["measured_speculation"]["draft_tokens"], 0)

    def test_invalid_completed_json_rejects_but_cap_truncation_is_distinct(self):
        relative = "mtp1/concurrency1/batch_00_001.json"
        self.change(relative, lambda batch: batch["requests"][0].update(final_json_schema_valid=False))
        report = analysis.analyze_trial(self.root)
        self.assertFalse(report["variants"]["mtp1"]["eligible_for_review"])
        self.assertTrue(any(e["kind"] == "schema" for e in self.condition(report)["errors"]))
        def truncate(batch):
            row = batch["requests"][0]
            row.update(finish_reason="length", output_tokens=1024)
            row["usage"]["completion_tokens"] = 1024
            batch["aggregate_output_tokens_per_second"] = 1024 / batch["batch_wall_s"]
        self.change(relative, truncate)
        report = analysis.analyze_trial(self.root)
        self.assertTrue(report["variants"]["mtp1"]["eligible_for_review"])
        self.assertEqual(self.condition(report)["measured_repeats"][0]["cap_truncated_responses"], 1)

    def test_counter_reset_and_generation_lag_block_review(self):
        self.change("mtp1/concurrency1/batch_00_001.json", lambda batch: batch.update(
            speculation={"draft_tokens": 10, "accepted_tokens": 20, "acceptance_counters_valid": True},
            metrics_generation_accounting_caught_up=False))
        report = analysis.analyze_trial(self.root)
        self.assertFalse(report["variants"]["mtp1"]["eligible_for_review"])
        self.assertEqual(len(self.condition(report)["measured_repeats"][0]["counter_errors"]), 2)

    def test_missing_repeat_never_uses_warmup_as_measurement(self):
        (self.root / "mtp1/concurrency1/batch_00_002.json").unlink()
        report = analysis.analyze_trial(self.root)
        self.assertEqual(self.condition(report)["measured_repeat_count"], 1)
        self.assertFalse(report["variants"]["mtp1"]["eligible_for_review"])

    def test_cross_variant_case_mismatch_blocks_speedup(self):
        replacement = "f" * 64
        self.change("mtp1/concurrency1/manifest.json", lambda value: value["cases"][0]["source"].update(request_sha256=replacement))
        for repeat in (0, 1, 2):
            def change_source(batch):
                batch["requests"][0]["source_request_sha256"] = replacement
                batch["requests"][0]["case_source"]["request_sha256"] = replacement
            self.change(f"mtp1/concurrency1/batch_00_{repeat:03d}.json", change_source)
        report = analysis.analyze_trial(self.root)
        self.assertTrue(self.condition(report)["complete"])
        self.assertFalse(self.condition(report)["vs_bf16"]["eligible_for_comparison"])
        self.assertNotIn("batch_throughput_speedup", self.condition(report)["vs_bf16"])

    def test_smoke_failure_and_wrong_context_block_review(self):
        self.change("mtp1/vision_schema_smoke.json", lambda value: value.update(ok=False))
        self.change("mtp3/concurrency1/manifest.json", lambda value: value["server"]["startup_args"].update(max_model_len=131072))
        report = analysis.analyze_trial(self.root)
        self.assertEqual(report["eligible_mtp_variants_for_review"], [])

    def test_new_output_only_and_generated_text_is_not_disclosed(self):
        output = self.root / "analysis.json"
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            self.assertEqual(analysis.main(["--trial-dir", str(self.root), "--output", str(output)]), 0)
        text = output.read_text()
        self.assertEqual(text, captured.getvalue())
        self.assertNotIn("DO_NOT_INCLUDE", text)
        self.assertNotIn("final_content", text)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            analysis.main(["--trial-dir", str(self.root), "--output", str(output)])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(output.read_text(), text)

    def test_malformed_nested_evidence_is_reported_without_losing_other_variants(self):
        examples = [
            ("mtp1/concurrency1/batch_00_001.json", lambda batch: batch["requests"][0].update(usage=None)),
            ("mtp1/concurrency1/manifest.json", lambda manifest: manifest["server"].update(startup_args=None)),
        ]
        for relative, change in examples:
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_text()
                try:
                    self.change(relative, change)
                    report = analysis.analyze_trial(self.root)
                    json.dumps(report, allow_nan=False)
                    self.assertFalse(report["variants"]["mtp1"]["eligible_for_review"])
                    self.assertTrue(report["variants"]["mtp3"]["eligible_for_review"])
                    self.assertTrue(self.condition(report)["errors"])
                finally:
                    path.write_text(original)
        for repeat in (0, 1, 2):
            self.change(f"mtp1/concurrency1/batch_00_{repeat:03d}.json",
                        lambda batch: batch.update(gpu_memory_samples=[None]))
        report = analysis.analyze_trial(self.root)
        json.dumps(report, allow_nan=False)
        self.assertEqual(self.condition(report)["gpu_memory"]["sample_errors"], 3)
        self.assertFalse(report["variants"]["mtp1"]["eligible_for_review"])
        self.assertTrue(report["variants"]["mtp3"]["eligible_for_review"])


if __name__ == "__main__":
    unittest.main()
