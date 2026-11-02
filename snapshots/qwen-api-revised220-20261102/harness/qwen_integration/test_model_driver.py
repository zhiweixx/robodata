"""Offline fail-closed accounting/IPC-boundary tests. No network or real keys."""

import copy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_driver import BudgetedAstraDriver, DriverStopped, observation_content


class FakeSDK:
    def __init__(self, *, fail=False, maximum=False):
        self.responses = self
        self.input_tokens = self
        self.fail = fail
        self.maximum = maximum
        self.counts = []
        self.creates = []
        self.output = {"code": "result = get_robot_state()", "summary": "Inspect robot", "give_up": False}
        self.status = "completed"

    def count(self, **payload):
        self.counts.append(copy.deepcopy(payload))
        return {"input_tokens": 128000 if self.maximum else 1000}

    def create(self, **payload):
        self.creates.append(copy.deepcopy(payload))
        if self.fail:
            raise TimeoutError("fake-test-key")
        inp, out = (128000, 8192) if self.maximum else (1000, 500)
        return {"id": "response_offline", "model": "gpt-6-astra", "service_tier": "default",
                "status": self.status, "output_text": json.dumps(self.output),
                "usage": {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                          "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": inp}}}

    def close(self):
        pass


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "usage.jsonl"
        self.sdk = FakeSDK()

    def driver(self, **kwargs):
        result = BudgetedAstraDriver(self.path, ["test_episode"], "get_robot_state() -> public robot state",
                                     _sdk_factory=lambda key: self.sdk, **kwargs)
        patcher = patch.object(result._module, "_read_key", return_value="fake-test-key")
        patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def step(self, driver, feedback=None):
        return driver.next_program("test_episode", "Lift the cup", {"robot_state": {"joint_positions": [0.0]}}, feedback)

    def test_no_credentials_at_construction_and_settings_are_isolated(self):
        driver = self.driver()
        driver._module._read_key.assert_not_called()
        self.assertEqual(driver.client.snapshot()["suite_charged_usd"], "0")
        proposal = self.step(driver)
        self.assertEqual(proposal.iteration, 1)
        self.assertEqual(self.sdk.creates[0]["max_output_tokens"], 8192)
        self.assertEqual(self.sdk.creates[0]["tools"], [])
        self.assertEqual(driver._module.REQUEST_RESERVATION, Decimal("2.0096"))
        self.assertNotIn("fake-test-key", self.path.read_text())

    def test_public_observation_rejects_privileged_fields_before_request(self):
        driver = self.driver()
        for observation in ({"object_poses": {}}, {"robot_state": {"reward": 1}}, {"images": [{"camera": "front", "data_url": "https://example.org/secret.png"}]}):
            with self.assertRaises(ValueError):
                driver.next_program("test_episode", "Lift the cup", observation)
        self.assertFalse(self.sdk.counts)
        self.assertFalse(self.sdk.creates)

    def test_iteration_cap_and_public_feedback(self):
        driver = self.driver(max_iterations=2)
        self.step(driver)
        with self.assertRaises(ValueError):
            self.step(driver)
        self.step(driver, {"ok": True, "result": {"gripper_aperture": 0.04}})
        with self.assertRaises(DriverStopped):
            self.step(driver, {"ok": True})
        self.assertEqual(len(self.sdk.creates), 2)

    def test_budget_reservation_blocks_before_network(self):
        self.sdk.maximum = True
        driver = self.driver(suite_budget_usd="3", max_iterations=4)
        self.step(driver)
        with self.assertRaises(DriverStopped):
            self.step(driver, {"ok": True})
        self.assertEqual(len(self.sdk.creates), 1)
        self.assertEqual(Decimal(driver.client.snapshot()["suite_charged_usd"]), Decimal("2.0096"))

    def test_timeout_retains_reservation_and_stops_no_retry(self):
        self.sdk.fail = True
        driver = self.driver()
        with self.assertRaises(driver._module.SuiteAborted):
            self.step(driver)
        with self.assertRaises(DriverStopped):
            self.step(driver)
        self.assertEqual(len(self.sdk.creates), 1)
        self.assertEqual(Decimal(driver.client.snapshot()["suite_charged_usd"]), Decimal("2.0096"))
        self.assertNotIn("fake-test-key", self.path.read_text())
        with self.assertRaises(DriverStopped):
            self.driver()

    def test_incomplete_output_not_executed_or_retried(self):
        self.sdk.status = "incomplete"
        driver = self.driver()
        with self.assertRaises(DriverStopped):
            self.step(driver)
        with self.assertRaises(DriverStopped):
            self.step(driver)
        self.assertEqual(len(self.sdk.creates), 1)
        events = [json.loads(line)["event"] for line in self.path.read_text().splitlines()]
        self.assertLess(events.index("request_settled"), events.index("driver_stopped"))

    def test_restart_cannot_repeat_measured_episode(self):
        self.step(self.driver())
        with self.assertRaises(DriverStopped):
            self.driver()

    def test_changed_suite_budget_cannot_reuse_ledger(self):
        driver = self.driver()
        with self.assertRaises(Exception) as caught:
            self.driver(suite_budget_usd="50")
        self.assertIn("configuration", str(caught.exception))
        self.assertFalse(self.sdk.creates)

    def test_done_never_contains_code(self):
        self.sdk.output = {"code": "", "summary": "Task appears complete", "give_up": True}
        driver = self.driver()
        self.assertTrue(self.step(driver).give_up)
        with self.assertRaises(DriverStopped):
            self.step(driver, {"ok": True})

    def test_three_episode_eight_call_maximum_fits_authorized_fifty(self):
        self.sdk.maximum = True
        driver = BudgetedAstraDriver(self.path, ["a", "b", "c"], "public API",
                                     suite_budget_usd="50", max_iterations=8,
                                     _sdk_factory=lambda key: self.sdk)
        with patch.object(driver._module, "_read_key", return_value="fake-test-key"):
            for episode in ("a", "b", "c"):
                for index in range(8):
                    driver.next_program(episode, "Lift cup", {}, None if index == 0 else {"ok": True})
        self.assertEqual(len(self.sdk.creates), 24)
        self.assertEqual(Decimal(driver.client.snapshot()["suite_charged_usd"]), Decimal("48.2304"))
        for record in map(json.loads, self.path.read_text().splitlines()):
            if record["event"] == "request_started":
                self.assertLessEqual(Decimal(record["suite_charged_with_reservation_usd"]), Decimal(50))


if __name__ == "__main__":
    unittest.main()
