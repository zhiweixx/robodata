"""Offline process-level file-relay tests; fake SDK, no network or actual keys."""

import multiprocessing
from pathlib import Path
import tempfile
import unittest

from model_driver import BudgetedAstraDriver, DriverStopped
from model_relay import RelayModelClient, _private_directory, _write_json, serve_relay
from test_model_driver import FakeSDK


def fake_controller(base):
    sdk = FakeSDK()
    driver = BudgetedAstraDriver(Path(base) / "usage.jsonl", ["a", "b", "c"],
                                 suite_budget_usd="50", max_iterations=2,
                                 _sdk_factory=lambda key: sdk)
    driver._module._read_key = lambda: "offline-only-not-a-real-key"
    serve_relay(driver, Path(base) / "relay", idle_timeout_s=10)


class RelayTests(unittest.TestCase):
    def test_persistent_controller_accepts_three_sequential_clients(self):
        with tempfile.TemporaryDirectory() as folder:
            context = multiprocessing.get_context("spawn")
            process = context.Process(target=fake_controller, args=(folder,))
            process.start()
            try:
                for episode in ("a", "b", "c"):
                    client = RelayModelClient(Path(folder) / "relay", timeout_s=10)
                    for index in range(2):
                        proposal = client.next_program(
                            episode, "Pick up cup", {"robot_state": {"joints": [0]}},
                            previous_result=None if index == 0 else {"ok": True},
                            api_documentation="get_robot_state() is a public flat tool; no SAM3 available",
                        )
                        self.assertEqual(proposal.iteration, index + 1)
                client.stop()
                process.join(5)
                self.assertEqual(process.exitcode, 0)
                self.assertEqual(len(list((Path(folder) / "relay" / "responses").glob("*.json"))), 7)
                self.assertNotIn("offline-only-not-a-real-key", (Path(folder) / "usage.jsonl").read_text())
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(5)

    def test_missing_response_is_never_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            relay = _private_directory(Path(folder) / "relay")
            _write_json(relay / "ready.json", {"ready": True})
            client = RelayModelClient(relay, timeout_s=1)
            with self.assertRaises(DriverStopped):
                client.next_program("a", "Lift cup", {}, api_documentation="public API")
            with self.assertRaises(DriverStopped):
                client.next_program("a", "Lift cup", {}, api_documentation="public API")
            self.assertEqual(len(list((relay / "requests").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
