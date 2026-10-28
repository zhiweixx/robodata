"""Public trajectory-length receipts, using the actual ASPIRE executor on CPU."""
from __future__ import annotations

import unittest

from cap_harness.contracts import ExecutionResult
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.registry import ToolRegistry
from robodata.execution.receipt_executor import ReceiptProgramExecutor, measured_details


class WaypointReceiptTests(unittest.TestCase):
    def test_automatic_failure_receipt_preserves_required_and_allowed_steps(self):
        calls = []

        def move_to_pose():
            calls.append("called")
            return ExecutionResult(
                ok=False, steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST, "trajectory exceeds max_steps",
                    details={"waypoint_count": 124, "max_steps": 120,
                             "reward": 1, "object_poses": {"mug": [1, 2, 3]},
                             "hidden_state": {"waypoint_count": 999}}))

        registry = ToolRegistry()
        registry.register(move_to_pose, name="move_to_pose", layer="shared",
                          capability="motion", public=True)
        outcome = ReceiptProgramExecutor(registry).execute_program("move_to_pose()")

        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(calls, ["called"])
        self.assertEqual(len(outcome.calls), 1)
        trace = outcome.calls[0]
        self.assertFalse(trace.ok)
        self.assertEqual(trace.failure_code, "invalid_request")
        self.assertEqual(trace.outputs_summary["steps_executed"], 0)
        self.assertEqual(trace.outputs_summary["error_details"],
                         {"waypoint_count": 124, "max_steps": 120})
        self.assertTrue(trace.outputs_summary["error_recoverable"])

    def test_nonnegative_integer_counts_are_retained(self):
        for count in (0, 1, 124):
            with self.subTest(count=count):
                self.assertEqual(measured_details({"waypoint_count": count}),
                                 {"waypoint_count": count})

    def test_invalid_count_types_and_nested_data_are_not_coerced(self):
        for value in (True, False, -1, 124.0, 1.5, float("nan"), float("inf"),
                      "124", None, [124], {"reward": 1}, {"waypoint_count": 124}):
            with self.subTest(value=value):
                self.assertEqual(measured_details({"waypoint_count": value,
                                                  "max_steps": 120,
                                                  "private": {"waypoint_count": 124}}),
                                 {"max_steps": 120})


if __name__ == "__main__":
    unittest.main()
