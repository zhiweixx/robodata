"""CPU regression checks: real ASPIRE executor, synthetic public sensors/robot.

No model, Isaac Sim, GPU or native task scorer is loaded by these tests.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np

from cap_harness.contracts import ExecutionResult, Pose, Trajectory
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.registry import ToolRegistry
from cap_harness.runtime import ProgramExecutor
from robodata.execution.aspire_glue import build_api, execute_program, implementation_provenance
from robodata.execution.deployment_docs import PIXEL_TO_WORLD_REFERENCE, deployed_api_documentation
from robodata.execution.model_episode import plain, public_observation
from robodata.execution.receipt_executor import ReceiptProgramExecutor, measured_details
from robodata.environments.robodojo.robodojo_adapter import RoboDojoAdapter
from tests.runtime.fixtures import Env


def execute(api, source):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "program.py"
        path.write_text(source)
        return execute_program(api, path)


class ContractTests(unittest.TestCase):
    def test_public_state_json_and_executable_field_agree(self):
        adapter = RoboDojoAdapter(Env())
        state = adapter.get_robot_state()
        public = public_observation(adapter)["robot_state"]
        self.assertNotIn("measured_gripper_positions", public)
        self.assertEqual(public["gripper_positions"], dict(state.gripper_positions))
        outcome = execute(build_api(adapter),
            'result = dict(get_robot_state().gripper_positions)')
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(dict(outcome.result), public["gripper_positions"])
        self.assertEqual(adapter.native_env.take_action_cnt, [0])

    def test_reference_100_pixels_uses_one_capture_and_zero_actions(self):
        env = Env()
        captures = []
        original = env.get_obs
        def capture():
            captures.append(1)
            return original()
        env.get_obs = capture
        adapter = RoboDojoAdapter(env)
        code = PIXEL_TO_WORLD_REFERENCE + '''
observation = get_observation()
camera = observation.cameras["cam_head"]
points = [pixel_to_world(camera, 1 + i % 10, 1 + i // 10) for i in range(100)]
result = {"points": points}
'''
        outcome = execute(build_api(adapter), code)
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(len(captures), 1)
        self.assertEqual([c.tool for c in outcome.calls], ["get_observation"])
        self.assertEqual(env.take_action_cnt, [0])
        expected = [[(1+i%10-16)/100., (1+i//10-12)/80., 1.] for i in range(100)]
        np.testing.assert_allclose(outcome.result["points"], expected)

    def test_reference_refresh_after_motion_is_explicit(self):
        env = Env()
        adapter = RoboDojoAdapter(env)
        code = PIXEL_TO_WORLD_REFERENCE + '''
before = get_observation()
state = get_robot_state()
step(RobotAction({"right": ArmCommand(mode="joint_position", target=state.joint_positions["right"], embodiment="robodojo")}))
after = get_observation()
result = {"before_s": before.cameras["cam_head"].timestamp_s,
          "after_s": after.cameras["cam_head"].timestamp_s,
          "point": pixel_to_world(after.cameras["cam_head"], 16, 12)}
'''
        outcome = execute(build_api(adapter), code)
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(env.take_action_cnt, [1])
        self.assertEqual([c.tool for c in outcome.calls].count("get_observation"), 2)
        self.assertEqual(outcome.result["before_s"], 0.)
        self.assertEqual(outcome.result["after_s"], .04)

    def test_geometry_retains_skew_rotation_and_validation(self):
        namespace = {}
        exec(PIXEL_TO_WORLD_REFERENCE, namespace)
        project = namespace["pixel_to_world"]
        camera = NS(depth_m=np.full((8, 9), 2.),
            intrinsics=np.array([[100., 3., 4.], [2., 80., 3.], [0., 0., 1.]]),
            camera_pose=Pose([1., 2., 3.], [2**-.5, 0., 0., 2**-.5], "world"))
        optical = 2.*np.linalg.solve(camera.intrinsics, [6., 5., 1.])
        expected = camera.camera_pose.as_matrix() @ np.r_[optical, 1.]
        np.testing.assert_allclose(project(camera, 6, 5), expected[:3], atol=1e-12)
        with self.assertRaisesRegex(ValueError, "outside"):
            project(camera, 9, 5)
        camera.depth_m[5, 6] = 0.
        with self.assertRaisesRegex(ValueError, "invalid metric depth"):
            project(camera, 6, 5)

    def test_docs_do_not_change_full_budget_contract(self):
        from robodata.eval.robodojo.full_deployment_docs import full_api_documentation
        api = build_api(RoboDojoAdapter(Env()))
        docs = full_api_documentation(api)
        self.assertIn("There is no harness cap on program count", docs)
        self.assertNotIn("300s Python watchdog", docs)
        self.assertIn("gripper_physical_joint_limits_m", docs)
        self.assertIn("not the total space between the fingers", docs)
        self.assertIn('pose.position, not pose["position"]', docs)


class ReceiptTests(unittest.TestCase):
    def make_registry(self, result):
        calls = []
        def move_to_pose():
            calls.append("one original call")
            return result
        registry = ToolRegistry()
        registry.register(move_to_pose, name="move_to_pose", layer="shared",
                          capability="motion", public=True)
        return registry, calls

    def test_error_receipt_preserves_measured_details_without_private_data(self):
        result = ExecutionResult(ok=False, steps_executed=13,
            final_errors={"right": .02},
            error=ApiError(ErrorCode.CONTROLLER_FAILED, "pose outside tolerance", details={
                "position_error_m": .02, "position_tolerance_m": .008,
                "orientation_error_rad": .3, "orientation_tolerance_rad": .17,
                "reward": 1., "object_poses": {"a": [1,2,3]},
                "joint_error_rad": {"right": .12, "reward": 777., "left": {"reward": 4.}},
                "targets": {"left": .5, "right": "sensitive content"},
                "position_error_m_hidden": 9., "raw_state": [1.,2.]}),
            diagnostics={"trajectory_executed": True, "joint_target_converged": True,
                "position_error_m": .02, "private": {"secret": 8},
                "pixels": np.ones((2,2)), "unknown": {"reward": 4.}})
        registry, calls = self.make_registry(result)
        outcome = ReceiptProgramExecutor(registry).execute_program(
            "motion = move_to_pose()\nresult = {\"motion_ok\": motion.ok}")
        self.assertTrue(outcome.ok)
        self.assertFalse(outcome.result["motion_ok"])
        self.assertEqual(calls, ["one original call"])
        self.assertEqual(len(outcome.calls), 1)
        trace = outcome.calls[0]
        self.assertFalse(trace.ok)
        self.assertEqual(trace.error_message, "pose outside tolerance")
        self.assertEqual(trace.failure_code, "controller_failed")
        self.assertEqual(trace.outputs_summary["steps_executed"], 13)
        details = trace.outputs_summary["error_details"]
        self.assertEqual(details["position_error_m"], .02)
        self.assertEqual(details["position_tolerance_m"], .008)
        self.assertEqual(details["orientation_error_rad"], .3)
        self.assertEqual(details["orientation_tolerance_rad"], .17)
        self.assertEqual(details["joint_error_rad"], {"right": .12})
        self.assertEqual(details["targets"], {"left": .5})
        encoded = json.dumps(plain(trace.outputs_summary))
        for prohibited in ["reward", "object_pose", "secret", "private", "raw_state",
                           "final_observation", "pixels", "sensitive content", "hidden"]:
            self.assertNotIn(prohibited, encoded)

    def test_untrusted_nested_or_nonfinite_measurements_are_omitted(self):
        details = measured_details({"position_error_m": {"reward": 2.},
            "orientation_error_rad": "object_pose", "joint_tolerance_rad": float("nan"),
            "max_steps": True, "completion": "success_predicate satisfied",
            "trajectory_executed": {"secret": True},
            "measured": {"right": float("inf"), "left": .4}})
        self.assertEqual(details, {"measured": {"left": .4}})

    def test_success_semantics_equal_upstream(self):
        result = ExecutionResult(ok=True, steps_executed=7, final_errors={"right": .01},
            diagnostics={"position_error_m": .01, "orientation_error_rad": .02})
        outcomes = []
        for executor in (ProgramExecutor, ReceiptProgramExecutor):
            registry, calls = self.make_registry(result)
            outcome = executor(registry).execute_program(
                "motion = move_to_pose()\nresult = {\"motion_ok\": motion.ok, \"steps\": motion.steps_executed}")
            self.assertEqual(calls, ["one original call"])
            outcomes.append(outcome)
        self.assertEqual(outcomes[0].result, outcomes[1].result)
        self.assertEqual(outcomes[0].ok, outcomes[1].ok)
        self.assertEqual(outcomes[0].error, outcomes[1].error)
        for field in ("ok", "tool", "failure_code", "error_message", "inputs_summary"):
            self.assertEqual(getattr(outcomes[0].calls[0], field), getattr(outcomes[1].calls[0], field))
        self.assertEqual(outcomes[1].calls[0].outputs_summary["measured_diagnostics"],
                         {"position_error_m": .01, "orientation_error_rad": .02})

    def test_validator_and_builtins_remain_restricted(self):
        registry, calls = self.make_registry(ExecutionResult(ok=True, steps_executed=0))
        executor = ReceiptProgramExecutor(registry)
        for source in ("import os", "result = move_to_pose.__closure__",
                       "result = getattr(move_to_pose, 'anything')"):
            outcome = executor.execute_program(source)
            self.assertFalse(outcome.ok)
        self.assertEqual(calls, [])

    def test_glue_records_receipt_extension_hash(self):
        provenance = implementation_provenance()
        self.assertEqual(len(provenance["receipt_extension"]["sha256"]), 64)
        self.assertIn("receipt_executor.py", provenance["receipt_extension"]["path"])
        self.assertEqual([p["name"] for p in provenance["upstream_patches"]],
                         ["0001-robodojo-contracts.patch", "0002-trajectory-gripper-cache.patch"])
        self.assertTrue(all(len(p["sha256"]) == 64 for p in provenance["upstream_patches"]))

    def test_joint_error_labels_preserve_actions_and_terminal_status(self):
        class LaggingEnv(Env):
            def take_action(self, action):
                super().take_action(action)
                self.robot_manager.robot_key[1].data.joint_pos[0] -= .1

        for case in ("converged", "lagging", "budget_end", "early_end"):
            with self.subTest(case=case):
                env = LaggingEnv() if case == "lagging" else Env()
                if case == "budget_end":
                    env.step_lim = 2
                if case == "early_end":
                    original = env.take_action
                    def ending(action):
                        original(action)
                        env.end_flag[0] = True
                    env.take_action = ending
                adapter = RoboDojoAdapter(env)
                state = adapter.get_robot_state()
                trajectory = Trajectory(joint_positions=np.zeros((2,6)), dt_s=.04,
                    joint_names=state.joint_names["right"], planner="test",
                    collision_aware=False, expected_start=state, arm="right",
                    embodiment="robodojo")
                result = adapter.execute_trajectory(trajectory)
                self.assertEqual(result.steps_executed, env.take_action_cnt[0])
                if case == "converged":
                    self.assertTrue(result.ok)
                    self.assertEqual(result.status.value, "success")
                    self.assertEqual(result.steps_executed, 2)
                    self.assertEqual(result.diagnostics["joint_error_rad"], {"right": 0.})
                    self.assertEqual(result.diagnostics["joint_tolerance_rad"], .05)
                elif case == "lagging":
                    self.assertFalse(result.ok)
                    self.assertEqual(result.status.value, "controller_failed")
                    self.assertEqual(result.steps_executed, 2)
                    self.assertEqual(result.error.code, ErrorCode.CONTROLLER_FAILED)
                    self.assertAlmostEqual(result.error.details["joint_error_rad"]["right"], .1)
                    self.assertEqual(result.error.details["joint_tolerance_rad"], .05)
                else:
                    self.assertFalse(result.ok)
                    self.assertEqual(result.truncated, case == "budget_end")
                    self.assertEqual(result.terminated, case == "early_end")
                    self.assertEqual(result.status.value,
                                     "truncated" if case == "budget_end" else "terminated")


if __name__ == "__main__":
    unittest.main()
