# Executed by the upstream ASPIRE restricted ProgramExecutor, without API calls.
task = get_task_context()
observation = get_observation()
state = get_robot_state()
if state.embodiment != "robodojo" or len(observation.cameras) != 3:
    raise ValueError("unexpected robot/camera contract")
# Empty-hand jaw motion at the initial pose; measurements must verify movement.
initial_aperture = float(state.gripper_positions["right"])
closed = set_gripper(0.0, arm="right")
closed_aperture = float(get_robot_state().gripper_positions["right"])
if not closed.ok or closed_aperture > 0.03 or initial_aperture - closed_aperture < 0.5:
    raise ValueError("empty-hand close did not measurably reach closed aperture")
half_opened = set_gripper(0.5, arm="right")
half_aperture = float(get_robot_state().gripper_positions["right"])
if not half_opened.ok or abs(half_aperture - 0.5) > 0.03:
    raise ValueError("half-open command did not reach half physical aperture")
opened = open_gripper(arm="right")
opened_aperture = float(get_robot_state().gripper_positions["right"])
if not opened.ok or opened_aperture < 0.97 or opened_aperture - closed_aperture < 0.5:
    raise ValueError("empty-hand open did not measurably reach open aperture")
state = get_robot_state()
hold = step(RobotAction({
    "right": ArmCommand(mode="joint_position", target=state.joint_positions["right"], embodiment="robodojo"),
    "left": ArmCommand(mode="joint_position", target=state.joint_positions["left"], embodiment="robodojo"),
}))
if not hold.ok:
    raise ValueError("one-period hold failed")
before = get_robot_state()
pose = before.end_effector_poses["right"]
target_xyz = pose.position.copy()
target_xyz[2] = target_xyz[2] + 0.01
target = Pose(position=target_xyz, quaternion_wxyz=pose.quaternion_wxyz, frame=pose.frame)
motion = move_to_pose(target, arm="right", tolerance=0.005, max_steps=60)
if not motion.ok:
    raise ValueError("small lift failed: " + str(motion.error))
after = get_robot_state()
delta_z = float(after.end_effector_poses["right"].position[2] - pose.position[2])
if delta_z < 0.005 or delta_z > 0.015:
    raise ValueError("measured lift differs from requested 1cm")
result = {"smoke_ok": True, "task_language": task.language,
    "camera_names": list(observation.cameras), "measured_lift_m": delta_z,
    "initial_aperture": initial_aperture, "closed_aperture": closed_aperture,
    "half_aperture": half_aperture, "half_steps": half_opened.steps_executed,
    "opened_aperture": opened_aperture,
    "close_steps": closed.steps_executed, "open_steps": opened.steps_executed,
    "motion_steps": motion.steps_executed,
    "scope": "empty-hand close/half/open, hold and1cm lift; not an attempted task solve"}
