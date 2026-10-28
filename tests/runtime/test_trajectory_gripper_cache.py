"""CPU regression checks for issued gripper commands after failed trajectories.

These exercise the real CapApi, RoboDojo adapter and planner bridge. No model,
GPU, simulator, or physical robot is started.
"""
from __future__ import annotations

import unittest

import numpy as np

from cap_harness.contracts import ExecutionResult, StepResult, SynchronizedTrajectory, Trajectory
from cap_harness.errors import ApiError, ErrorCode
from robodata.execution.aspire_glue import build_api
from robodata.environments.robodojo.robodojo_adapter import RoboDojoAdapter
from tests.runtime.fixtures import Env


class ContactEnv(Env):
    """The command can close fully while measured aperture stalls on an object."""
    def take_action(self, action):
        super().take_action(action)
        for arm, value in self.physical_gripper.items():
            self.physical_gripper[arm] = max(.35, value)


class PrefixFailureAdapter(RoboDojoAdapter):
    prefix = 1

    def step(self, action):
        if self._env.take_action_cnt[0] >= self.prefix:
            return StepResult(ok=False, error=ApiError(
                ErrorCode.CONTROLLER_FAILED, "next waypoint rejected before dispatch"))
        return super().step(action)


class TrajectoryGripperCacheTests(unittest.TestCase):
    def setup_runtime(self, env=None, adapter_class=RoboDojoAdapter):
        env = ContactEnv() if env is None else env
        adapter = adapter_class(env)
        api = build_api(adapter)
        # Prime the existing cache through public planning, without executing.
        for arm in adapter.arms:
            self.assert_plan_grip(api, env, arm, 1.)
        return env, adapter, api

    def assert_plan_grip(self, api, env, arm, expected):
        before = env.take_action_cnt[0]
        state = api.get_robot_state()
        plan = api.plan_motion(state.joint_positions[arm], arm=arm)
        self.assertTrue(plan.ok, plan.error)
        np.testing.assert_allclose(plan.trajectory.gripper_positions, expected)
        self.assertEqual(env.take_action_cnt[0], before, "reading/planning must not issue actions")

    def trajectory(self, adapter, *, synchronized=False, with_grippers=True):
        state = adapter.get_robot_state()
        common = dict(dt_s=adapter.control_period_s, planner="test", collision_aware=False,
                      expected_start=state, embodiment="robodojo")
        if synchronized:
            return SynchronizedTrajectory(
                joint_positions={arm: np.tile(state.joint_positions[arm], (3, 1)) for arm in state.arms},
                joint_names=dict(state.joint_names),
                gripper_positions=({"left": [.1, .6, .7], "right": [.2, .5, 0.]}
                                   if with_grippers else None), **common)
        return Trajectory(joint_positions=np.tile(state.joint_positions["right"], (3, 1)),
            joint_names=state.joint_names["right"], arm="right",
            gripper_positions=[.2, .5, 0.] if with_grippers else None, **common)

    def test_partial_single_uses_last_issued_command_not_future_target_or_aperture(self):
        env, adapter, api = self.setup_runtime(adapter_class=PrefixFailureAdapter)
        result = api.execute_trajectory(self.trajectory(adapter))
        self.assertFalse(result.ok)
        self.assertEqual(result.steps_executed, 1)
        self.assertEqual(adapter.get_robot_state().gripper_positions["right"], .35)
        self.assert_plan_grip(api, env, "right", .2)
        self.assert_plan_grip(api, env, "left", 1.)

    def test_partial_synchronized_retains_each_arms_issued_target(self):
        env, adapter, api = self.setup_runtime(adapter_class=PrefixFailureAdapter)
        adapter.prefix = 2
        result = api.execute_trajectory(self.trajectory(adapter, synchronized=True))
        self.assertFalse(result.ok)
        self.assertEqual(result.steps_executed, 2)
        self.assert_plan_grip(api, env, "left", .6)
        self.assert_plan_grip(api, env, "right", .5)

    def test_terminal_and_truncated_execution_keep_last_command(self):
        for truncated in (False, True):
            for synchronized in (False, True):
                with self.subTest(truncated=truncated, synchronized=synchronized):
                    env = ContactEnv()
                    if truncated:
                        env.step_lim = 1
                    else:
                        original = env.take_action
                        def terminate(action):
                            original(action)
                            env.end_flag[0] = True
                        env.take_action = terminate
                    env, adapter, api = self.setup_runtime(env)
                    result = api.execute_trajectory(self.trajectory(adapter, synchronized=synchronized))
                    self.assertFalse(result.ok)
                    self.assertEqual(result.steps_executed, 1)
                    self.assertEqual(result.truncated, truncated)
                    self.assertEqual(result.terminated, not truncated)
                    self.assert_plan_grip(api, env, "right", .2)
                    self.assert_plan_grip(api, env, "left", .1 if synchronized else 1.)

    def test_final_tracking_failure_keeps_close_command_for_next_plan(self):
        class LaggingEnv(ContactEnv):
            def take_action(self, action):
                super().take_action(action)
                self.robot_manager.robot_key[1].data.joint_pos[0] -= .1
        for synchronized in (False, True):
            with self.subTest(synchronized=synchronized):
                env, adapter, api = self.setup_runtime(LaggingEnv())
                result = api.execute_trajectory(self.trajectory(adapter, synchronized=synchronized))
                self.assertEqual(result.error.code, ErrorCode.CONTROLLER_FAILED)
                self.assertEqual(result.steps_executed, 3)
                self.assert_plan_grip(api, env, "right", 0.)
                self.assert_plan_grip(api, env, "left", .7 if synchronized else 1.)

    def test_successful_single_and_synchronized_execution_still_updates(self):
        for synchronized in (False, True):
            with self.subTest(synchronized=synchronized):
                env, adapter, api = self.setup_runtime()
                result = api.execute_trajectory(self.trajectory(adapter, synchronized=synchronized))
                self.assertTrue(result.ok, result.error)
                self.assert_plan_grip(api, env, "right", 0.)
                self.assert_plan_grip(api, env, "left", .7 if synchronized else 1.)

    def test_zero_step_rejection_and_stale_plan_do_not_change_cache(self):
        for reason in ("adapter_rejection", "stale", "invalid_option"):
            with self.subTest(reason=reason):
                env, adapter, api = self.setup_runtime(adapter_class=PrefixFailureAdapter)
                adapter.prefix = 0
                trajectory = self.trajectory(adapter)
                if reason == "stale":
                    env.robot_manager.robot_key[1].data.joint_pos[0] = .3
                result = api.execute_trajectory(trajectory,
                    stop_on_termination=False if reason == "invalid_option" else True)
                self.assertFalse(result.ok)
                self.assertEqual(result.steps_executed, 0)
                self.assertEqual(env.take_action_cnt[0], 0)
                self.assert_plan_grip(api, env, "right", 1.)

    def test_exception_after_native_acceptance_uses_adapter_evidence_not_zero_count(self):
        env, adapter, api = self.setup_runtime()
        trajectory = self.trajectory(adapter)
        observe = adapter.get_observation
        def failing_observation():
            raise RuntimeError("sensor failed after command acceptance")
        adapter.get_observation = failing_observation
        result = api.execute_trajectory(trajectory)
        self.assertEqual(result.error.code, ErrorCode.EXECUTION_FAILED)
        self.assertEqual(result.steps_executed, 0)  # Exception path cannot report a prefix.
        self.assertEqual(env.take_action_cnt[0], 1)
        adapter.get_observation = observe
        self.assert_plan_grip(api, env, "right", .2)

    def test_no_gripper_column_does_not_infer_targets_from_aperture(self):
        for synchronized in (False, True):
            with self.subTest(synchronized=synchronized):
                env, adapter, api = self.setup_runtime(adapter_class=PrefixFailureAdapter)
                trajectory = self.trajectory(adapter, synchronized=synchronized, with_grippers=False)
                def unexpected_read():
                    self.fail("trajectory without gripper commands needs no cache refresh")
                adapter.get_commanded_gripper_positions = unexpected_read
                result = api.execute_trajectory(trajectory)
                self.assertFalse(result.ok)
                self.assert_plan_grip(api, env, "right", 1.)

    def test_snapshot_is_independent_and_contains_only_normalized_command_targets(self):
        env, adapter, api = self.setup_runtime(adapter_class=PrefixFailureAdapter)
        api.execute_trajectory(self.trajectory(adapter))
        snapshot = adapter.get_commanded_gripper_positions()
        self.assertEqual(snapshot, {"left": 1., "right": .2})
        snapshot["right"] = .9
        snapshot["unknown"] = 0.
        self.assertEqual(adapter.get_commanded_gripper_positions(), {"left": 1., "right": .2})
        self.assert_plan_grip(api, env, "right", .2)

    def test_invalid_optional_evidence_does_not_mask_failure_or_poison_cache(self):
        invalid = [None, {"right": .2}, {"left": 1., "right": .2, "unknown": 0.}]
        invalid.extend({"left": 1., "right": value} for value in
                       (True, np.bool_(True), "0.2", None, float("nan"), float("inf"), -1., 1.1, [0.2]))
        for evidence in invalid:
            with self.subTest(evidence=evidence):
                env, adapter, api = self.setup_runtime(adapter_class=PrefixFailureAdapter)
                adapter.get_commanded_gripper_positions = lambda: evidence
                result = api.execute_trajectory(self.trajectory(adapter))
                self.assertEqual(result.error.code, ErrorCode.CONTROLLER_FAILED)
                self.assert_plan_grip(api, env, "right", 1.)

    def test_evidence_reader_exception_does_not_hide_execution_error(self):
        env, adapter, api = self.setup_runtime(adapter_class=PrefixFailureAdapter)
        def failing_reader():
            raise RuntimeError("evidence unavailable")
        adapter.get_commanded_gripper_positions = failing_reader
        result = api.execute_trajectory(self.trajectory(adapter))
        self.assertEqual(result.error.code, ErrorCode.CONTROLLER_FAILED)
        self.assert_plan_grip(api, env, "right", 1.)

    def test_adapter_without_evidence_preserves_success_only_fallback(self):
        for prefix in (1, 99):
            with self.subTest(prefix=prefix):
                env, adapter, api = self.setup_runtime(adapter_class=PrefixFailureAdapter)
                adapter.prefix = prefix
                adapter.get_commanded_gripper_positions = None
                result = api.execute_trajectory(self.trajectory(adapter))
                self.assertEqual(result.ok, prefix == 99)
                self.assert_plan_grip(api, env, "right", 0. if result.ok else 1.)


if __name__ == "__main__":
    unittest.main()
