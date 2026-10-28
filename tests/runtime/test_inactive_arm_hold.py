"""CPU regressions for native command latching; no model or simulator is loaded."""
from __future__ import annotations

import unittest

import numpy as np

from cap_harness.contracts import ArmCommand, RobotAction
from robodata.environments.robodojo.robodojo_adapter import RoboDojoAdapter
from tests.runtime.fixtures import Env


def joint_action(arm, position):
    return RobotAction({arm: ArmCommand(mode="joint_position",
        target=np.full(6, position), embodiment="robodojo")})


class DisturbedEnv(Env):
    def __init__(self):
        super().__init__()
        self.commands = []

    def take_action(self, action):
        self.commands.append({key: value.copy() for key, value in action.items()})
        super().take_action(action)
        # A persistent load leaves the measured joints below their commanded
        # values. Re-sampling these as hold targets would integrate the error.
        for entity in self.robot_manager.robot_key:
            entity.data.joint_pos[0] -= .01


class InactiveArmHoldTests(unittest.TestCase):
    def test_disturbance_does_not_accumulate_into_inactive_arm_target(self):
        env = DisturbedEnv()
        env.robot_manager.robot_key[0].data.joint_pos[0] = .25
        adapter = RoboDojoAdapter(env)

        for _ in range(10):
            self.assertTrue(adapter.step(joint_action("right", .1)).ok)

        for command in env.commands:
            np.testing.assert_allclose(command["left_arm_joint_state"], .25)
        np.testing.assert_allclose(adapter.get_robot_state().joint_positions["left"], .24)
        self.assertEqual(env.take_action_cnt, [10])

    def test_explicit_joint_command_becomes_next_inactive_hold_target(self):
        env = DisturbedEnv()
        adapter = RoboDojoAdapter(env)
        adapter.step(joint_action("left", .4))
        adapter.step(joint_action("right", -.2))
        adapter.step(joint_action("left", .6))

        np.testing.assert_allclose(env.commands[1]["left_arm_joint_state"], .4)
        np.testing.assert_allclose(env.commands[2]["right_arm_joint_state"], -.2)
        np.testing.assert_allclose(env.commands[2]["left_arm_joint_state"], .6)
        self.assertEqual(env.take_action_cnt, [3])

    def test_native_command_buffer_mutation_cannot_change_latched_target(self):
        env = Env()
        adapter = RoboDojoAdapter(env)
        action = joint_action("right", .3)
        adapter.step(action)
        # Native code can retain or alter its own command buffer. It must not
        # own either the adapter's persistent target or the public action.
        env.last_action["right_arm_joint_state"][:] = 9.
        np.testing.assert_allclose(action.arms["right"].target, .3)
        adapter.step(joint_action("left", .1))
        np.testing.assert_allclose(env.last_action["right_arm_joint_state"], .3)

    def test_rejected_action_does_not_partially_update_joint_targets(self):
        env = Env()
        adapter = RoboDojoAdapter(env)
        invalid = RobotAction({
            "right": joint_action("right", .8).arms["right"],
            "unknown": joint_action("unknown", .2).arms["unknown"],
        })
        with self.assertRaisesRegex(ValueError, "unknown arm"):
            adapter.step(invalid)
        self.assertEqual(env.take_action_cnt, [0])
        adapter.step(joint_action("left", .1))
        np.testing.assert_allclose(env.last_action["right_arm_joint_state"], 0.)

    def test_reset_initializes_holds_from_new_episode_state(self):
        class ResettableEnv(Env):
            def reset(self, seed):
                self.env_seeds = seed
                self.take_action_cnt = [0]
                self.end_flag = [False]
                for index, entity in enumerate(self.robot_manager.robot_key):
                    entity.data.joint_pos[0] = .2 + .1 * index

        env = ResettableEnv()
        adapter = RoboDojoAdapter(env)
        adapter.step(joint_action("left", .8))
        adapter.step(joint_action("right", .9))
        adapter.reset("general_pickup", 7)
        # Disturb the newly reset pose before the first action, so this also
        # checks that the hold was captured at reset rather than at first step.
        env.robot_manager.robot_key[0].data.joint_pos[0] -= .02
        adapter.step(joint_action("right", .4))
        np.testing.assert_allclose(env.last_action["left_arm_joint_state"], .2)
        self.assertEqual(env.env_seeds, [7])
        self.assertEqual(env.take_action_cnt, [1])

    def test_native_rejection_keeps_previous_targets(self):
        env = Env()
        adapter = RoboDojoAdapter(env)
        take_action = env.take_action
        def reject(action):
            raise RuntimeError("native command rejected before stepping")
        env.take_action = reject
        with self.assertRaisesRegex(RuntimeError, "native command rejected"):
            adapter.step(joint_action("right", .8))
        env.take_action = take_action
        adapter.step(joint_action("left", .1))
        np.testing.assert_allclose(env.last_action["right_arm_joint_state"], 0.)
        self.assertEqual(env.take_action_cnt, [1])


if __name__ == "__main__":
    unittest.main()
