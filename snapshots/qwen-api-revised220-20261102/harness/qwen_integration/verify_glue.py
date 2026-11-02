"""CPU contract check only; this does not claim a RoboDojo rollout succeeded."""
import numpy as np

from cap_harness.contracts import (
    CameraObservation, Observation, Pose, RobotPlanningContext, RobotState, TaskContext,
)
from cap_harness.registry import ToolRegistry
from cap_harness.runtime import ProgramExecutor
from aspire_glue import EXTENSION_ALLOWLIST, build_api, render_api_docs


class FakeAdapter:
    embodiment = "robodojo"
    control_period_s = .04
    native_dt = .04

    def __init__(self):
        self.state = RobotState(
            joint_positions={a: np.zeros(6) for a in ("left", "right")},
            joint_velocities={a: np.zeros(6) for a in ("left", "right")},
            end_effector_poses={a: Pose([0, 0, 1], [1, 0, 0, 0], "world")
                                for a in ("left", "right")},
            gripper_positions={"left": 1., "right": 1.},
            base_frame="world", embodiment="robodojo",
        )

    def get_robot_state(self):
        return self.state

    def get_observation(self):
        camera = CameraObservation(
            np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4)), np.eye(3),
            "cam_head", Pose([0, 0, 0], [1, 0, 0, 0], "world"),
        )
        return Observation({"cam_head": camera}, self.state)

    def native_plan_to_pose(self, state, target, **options):
        return dict(ok=True, joint_positions=np.zeros((2, 6)),
                    dt_s=self.native_dt, collision_aware=False)

    def get_task_context(self):
        return TaskContext("fake", 0, "fake", "hold", "contract-check")

    def resolve_arm(self, arm):
        return {"primary": "right", "secondary": "left"}.get(arm, arm)

    def get_planning_context(self):
        return RobotPlanningContext(
            "robodojo", "dual_arx_x5", self.state.joint_names,
            {a: np.eye(4) for a in self.state.arms},
            {a: "flange" for a in self.state.arms},
        )


def main():
    adapter = FakeAdapter()
    api = build_api(adapter)
    goal = Pose([0, 0, 1], [1, 0, 0, 0], "world")
    planned = api.plan_motion(goal, arm="right")
    assert planned.ok, planned.error
    assert planned.trajectory.joint_positions.shape == (2, 6)
    assert planned.trajectory.planner == "robodojo-native"
    assert planned.trajectory.collision_aware is False
    assert "move_to_pose" in render_api_docs(api)
    adapter.native_dt = .05
    assert not api.plan_motion(goal, arm="right").ok
    assert not api.plan_motion(Pose([0, 0, 1], [1, 0, 0, 0], "camera"), arm="right").ok
    registry = api.register_tools(ToolRegistry(public_extension_allowlist=EXTENSION_ALLOWLIST))
    executor = ProgramExecutor(registry)
    result = executor.execute_program(
        "s=get_robot_state(); result={'joint_count':len(s.joint_positions['right']), 'frame':s.base_frame}"
    )
    assert result.ok, result.error
    assert result.result["joint_count"] == 6
    assert not executor.execute_program("import os").ok
    print("PASS actual ASPIRE CapApi/ProgramExecutor: six-joint plans, wrong-frame/timestep "
          "rejection, registry and restricted imports. Fake adapter only; no simulator claim.")


if __name__ == "__main__":
    main()
