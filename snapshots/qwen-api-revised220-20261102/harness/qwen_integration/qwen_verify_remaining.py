"""Offline continuation checks using copies of the archived original evidence."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from model_driver import DriverStopped
from qwen_remaining_audit import (
    EPISODE, PRIOR_RUN_ID, audit_prior_run, claim_continuation, verify_evidence,
)
from qwen_remaining_driver import AlignmentOnlyDriver, validate_controller_config
from qwen_server_suite import suite_launcher


class RemainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).resolve().parent.parent
        cls.runtime = cls.source / ("qwen" if (cls.source / "qwen/model_manifest.json").exists() else "qwen_runtime")
        cls.original_audit = audit_prior_run(cls.source, runtime_root=cls.runtime)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for source in self.original_audit["evidence_sha256"]:
            source = Path(source)
            try:
                relative = Path("qwen") / source.relative_to(self.runtime)
            except ValueError:
                relative = source.relative_to(self.source)
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        self.prior = self.root / "results" / PRIOR_RUN_ID

    def tearDown(self):
        self.tmp.cleanup()

    def test_actual_audit_one_time_claim_and_frozen_config(self):
        audit = audit_prior_run(self.root)
        self.assertEqual(audit["prior_calls"][EPISODE], 0)
        output = self.root / "results/qwen3_99999999"
        claim_continuation(audit, output)
        verify_evidence(audit)
        with self.assertRaisesRegex(RuntimeError, "already claimed"):
            claim_continuation(audit, output)
        config = {**audit["controller_config"], "usage_jsonl": str(output / "usage.jsonl"),
                  "base_url": "http://127.0.0.1:12345/v1"}
        validate_controller_config(config, audit)
        with self.assertRaisesRegex(RuntimeError, "changed policy"):
            validate_controller_config({**config, "max_output_tokens": 16384}, audit)

    def test_touched_task_or_new_relay_input_rejected(self):
        audit = audit_prior_run(self.root)
        (self.prior / "align_blocks").mkdir()
        with self.assertRaisesRegex(RuntimeError, "already started"):
            audit_prior_run(self.root)
        (self.prior / "align_blocks").rmdir()
        (self.prior / "relay_audit/requests/new.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            verify_evidence(audit)
        with self.assertRaisesRegex(RuntimeError, "unmatched or extra"):
            audit_prior_run(self.root)

    def test_uncertain_failure_rejected(self):
        path = self.prior / "usage.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[-1]["reason"] = "transport timeout with unknown result"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(RuntimeError, "unreviewed reason"):
            audit_prior_run(self.root)

    def test_wrapper_rejects_attempted_tasks_without_changing_allowed_arguments(self):
        class FakeDriver:
            max_iterations = 8
            def __init__(self): self.received = []
            def next_program(self, **kwargs):
                self.received.append(kwargs)
                return "unchanged-result"
        base = FakeDriver()
        gate = AlignmentOnlyDriver(base)
        values = dict(episode_id=EPISODE, task_instruction="unchanged", public_observation={"images": []},
                      previous_result=None, api_documentation="unchanged docs")
        self.assertEqual(gate.next_program(**values), "unchanged-result")
        self.assertEqual(base.received, [values])
        for episode in ("general_pickup_s0_l0", "stack_bowls_s0_l0", "align_blocks_s1_l0"):
            with self.assertRaises(DriverStopped):
                gate.next_program(**{**values, "episode_id": episode})
        self.assertEqual(len(base.received), 1)

    def test_supervisor_selector_is_opt_in_and_scoped(self):
        path = Path("/integration")
        self.assertEqual(suite_launcher(path), (path / "run_qwen_suite.py", []))
        self.assertEqual(suite_launcher(path, PRIOR_RUN_ID),
                         (path / "run_qwen_remaining.py", ["--prior-run-id", PRIOR_RUN_ID]))
        with self.assertRaisesRegex(RuntimeError, "reviewed original"):
            suite_launcher(path, "qwen3_other")


if __name__ == "__main__":
    unittest.main()
