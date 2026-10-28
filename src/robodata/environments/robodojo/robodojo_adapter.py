"""In-process RoboDojo boundary for ASPIRE; simulator state stays private.

Public poses use the native X5 flange orientation and the grasp-point origin
(0.1501 m along flange +X). They are NOT Panda/contact-graspnet tool frames.
Native CuRobo currently checks its configured table and self collisions only.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
import numpy as np

from cap_harness.contracts import (ArmCommand, CameraObservation, ExecutionResult,
    Observation, Pose, RobotAction, RobotPlanningContext, RobotState, StepResult,
    SynchronizedTrajectory, TaskContext, Trajectory)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import matrix_to_pose
from robodata.environments.robodojo.native_retime import retime_native_path
from robodata.execution.native_stdout import route_native_progress


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class RoboDojoAdapter:
    embodiment = "robodojo"
    arms = ("left", "right")
    BASE_FRAME = "world"
    TCP_OFFSET_M = 0.1501
    gripper_holds_intermediate = True

    def __init__(self, task_env, *, recorder=None):
        if task_env.num_envs != 1:
            raise ValueError("pilot bridge supports exactly one environment")
        self._env = task_env
        self._manager = task_env.robot_manager
        self._robots = {a: self._manager.get_robot_by_arm_name(f"{a}_arm") for a in self.arms}
        self._base_poses = {}
        self._base_quaternion_norms = {}
        for arm,robot in self._robots.items():
            base=as_numpy(robot.entity_origin_pose).astype(np.float64)
            if base.shape != (7,) or not np.isfinite(base).all():
                raise ValueError(f"invalid native base pose for {arm}")
            norm=float(np.linalg.norm(base[3:7]))
            # Native dual_x5.yml rounds sqrt(.5) to.707. Normalize this small
            # representation error once, before strict ASPIRE Pose validation,
            # and share exactly the same frame with the native planner.
            if abs(norm-1.0)>1e-3:
                raise ValueError(f"native base quaternion for {arm} is not near unit length: {norm}")
            self._base_quaternion_norms[arm]=norm
            self._base_poses[arm]=Pose(base[:3],base[3:7]/norm,self.BASE_FRAME)
        self._gripper_limits = {}
        for arm,robot in self._robots.items():
            entity=self._manager.robot_key[self._manager.robot_list.index(robot)]
            hard_limits=as_numpy(entity.root_physx_view.get_dof_limits())
            low,high=map(float,hard_limits[0,robot.gripper_joint_indices[0]])
            if not math.isfinite(low) or not math.isfinite(high) or not high>low:
                raise ValueError(f"invalid physical gripper joint limits for {arm}")
            self._gripper_limits[arm]=(low,high)
        self.control_frequency = float(task_env.obs_manager.collect_freq)
        if not math.isclose(self.control_frequency, 25.0):
            raise ValueError("pilot bridge requires the checked 25 Hz action configuration")
        self.control_period_s = 1.0 / self.control_frequency
        self._recorder = recorder
        self._closed = False
        self._task_context = None
        self._last_observation = None
        self._last_calibration = {}
        self._home = {a: self._manager.get_joint(r, [0])[0].copy() for a, r in self._robots.items()}
        state = self.get_robot_state()
        self._joint_targets = {a: state.joint_positions[a].copy() for a in self.arms}
        self._gripper_targets = dict(state.gripper_positions)

    @property
    def native_env(self):
        """Runtime-only handle; never registered in the generated-code tools."""
        return self._env

    @property
    def current_time_s(self):
        return int(self._env.take_action_cnt[0]) * self.control_period_s

    @property
    def dof(self):
        return 6

    def resolve_arm(self, arm):
        return {"primary": "right", "secondary": "left"}.get(arm, arm)

    def reset(self, task_ref, seed):
        task_name = task_ref if isinstance(task_ref, str) else getattr(task_ref, "task_name", None)
        if task_name not in (None, self._env.task_name):
            raise ValueError("construct a new native task for a different task name")
        self._env.reset(seed=[int(seed)])
        self._task_context = None
        state = self.get_robot_state()
        self._joint_targets = {a: state.joint_positions[a].copy() for a in self.arms}
        return self.get_observation()

    def _tcp_pose(self, flange):
        flange = np.asarray(flange, dtype=np.float64)
        p = Pose(position=flange[:3], quaternion_wxyz=flange[3:7], frame=self.BASE_FRAME)
        T = p.as_matrix().copy()
        T[:3, 3] += T[:3, 0] * self.TCP_OFFSET_M
        return matrix_to_pose(T, frame=self.BASE_FRAME)

    def _flange_array(self, tcp):
        if tcp.frame != self.BASE_FRAME:
            raise ValueError("native planning target must be in world frame")
        T = tcp.as_matrix()
        return np.concatenate((tcp.position - T[:3, 0] * self.TCP_OFFSET_M, tcp.quaternion_wxyz))

    def get_robot_state(self):
        joints, velocities, poses, grippers, names = {}, {}, {}, {}, {}
        for arm, robot in self._robots.items():
            entity = self._manager.robot_key[self._manager.robot_list.index(robot)]
            joints[arm] = self._manager.get_joint(robot, [0])[0]
            velocities[arm] = as_numpy(entity.data.joint_vel[0, robot.arm_joint_indices])
            poses[arm] = self._tcp_pose(self._manager.get_real_endpose(robot, [0], is_relative=False)[0])
            physical = float(self._manager.get_end_effector_real_val(robot, [0])[0][0])
            # Command scale includes negative preload on X5 (-.01m), whereas
            # the physical prismatic joint stops at0m. Do not normalize actual
            # aperture by an unreachable command endpoint.
            low, high = self._gripper_limits[arm]
            fraction = (physical - low) / (high - low)
            if robot.gripper_move["sign"] != 1:
                fraction = 1.0 - fraction
            grippers[arm] = float(np.clip(fraction, 0.0, 1.0))
            names[arm] = tuple(robot.arm_joints_name)
        return RobotState(joint_positions=joints, joint_velocities=velocities,
            end_effector_poses=poses, gripper_positions=grippers,
            base_frame=self.BASE_FRAME, joint_names=names, timestamp_s=self.current_time_s,
            embodiment=self.embodiment)

    def get_commanded_gripper_positions(self):
        """Runtime-only command evidence, not measured aperture or a public tool.

        ``step`` updates these targets after native action acceptance, before
        observation/tracking checks can fail. Return an independent snapshot.
        """
        return dict(self._gripper_targets)

    def get_task_context(self):
        if self._task_context is None:
            raw = self._env.get_obs()
            self._set_context(raw)
        return self._task_context

    def _set_context(self, raw):
        instruction = raw.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("RoboDojo did not supply authoritative task language")
        family = "Generalization" if self._env.task_name == "stack_bowls" else "Open"
        task_index = {"general_pickup":0,"stack_bowls":1,"align_blocks":2}.get(self._env.task_name,0)
        self._task_context = TaskContext(suite="RoboDojo", task_id=task_index,
            task_name=self._env.task_name, language=instruction, family=family,
            metadata={"robot": "dual_arx_x5", "control_hz": self.control_frequency,
                "end_effector_reference": "native flange orientation; grasp-point origin +0.1501m flange X"})

    def get_observation(self):
        raw = self._env.get_obs()
        self._set_context(raw)
        manager = self._env.camera_manager
        cameras, calibration = {}, {}
        for camera_index, name in enumerate(manager.camera_names[0]):
            view = raw["vision"][name]
            if "depth" not in view:
                raise ValueError(f"metric depth missing from {name}; configure annotators before reset")
            rgb = as_numpy(view["color"])[..., :3]
            if rgb.dtype != np.uint8:
                raise ValueError("native camera RGB must be uint8; no silent channel/scale conversion")
            depth = as_numpy(view["depth"]).squeeze().astype(np.float64)
            depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
            camera = manager.cameras[0][camera_index]
            # Read actual sensor calibration, not the config helper's fx==fy assumption.
            K = as_numpy(camera.get_intrinsics_matrix()).astype(np.float64)
            position, quaternion = camera.get_world_pose(camera_axes="ros")
            camera_pose = Pose(position=as_numpy(position), quaternion_wxyz=as_numpy(quaternion), frame=self.BASE_FRAME)
            cameras[name] = CameraObservation(rgb=rgb, depth_m=depth, intrinsics=K,
                frame=f"camera/{name}", camera_pose=camera_pose, timestamp_s=self.current_time_s)
            calibration[name] = {"shape": list(rgb.shape), "valid_depth_fraction": float(np.mean(depth > 0)),
                "depth_min_positive_m": float(depth[depth > 0].min()) if np.any(depth > 0) else None,
                "depth_max_m": float(depth.max()), "intrinsics": K.tolist(),
                "world_from_camera": camera_pose.as_matrix().tolist()}
        self._last_calibration = calibration
        result = Observation(cameras=cameras, robot_state=self.get_robot_state(),
            task_context=self._task_context, timestamp_s=self.current_time_s)
        self._last_observation = result
        return result

    def calibration_sanity(self):
        """Cross-check CV backprojection against Isaac's own projection helper."""
        observation = self.get_observation()
        report = {}
        manager = self._env.camera_manager
        for index, name in enumerate(manager.camera_names[0]):
            obs = observation.cameras[name]
            height, width = obs.depth_m.shape
            uv = np.array([[.5*width, .5*height], [.3*width, .4*height], [.7*width, .6*height]])
            # Unit optical depth: calibration check does not query object ground truth.
            rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(obs.intrinsics).T
            T = obs.camera_pose.as_matrix()
            world = rays @ T[:3, :3].T + T[:3, 3]
            camera = manager.cameras[0][index]
            # Isaac sensors follow the active backend. Match the public pose's
            # tensor dtype/device instead of passing NumPy into Torch's pad().
            backend_position, _ = camera.get_world_pose(camera_axes="ros")
            backend_world = (backend_position.new_tensor(world)
                if hasattr(backend_position,"new_tensor") else world)
            projected = as_numpy(camera.get_image_coords_from_world_points(backend_world))
            error_px = float(np.max(np.linalg.norm(projected - uv, axis=1)))
            report[name] = {**self._last_calibration[name], "roundtrip_max_error_px": error_px}
            if not math.isfinite(error_px) or error_px > 0.1:
                raise ValueError(f"camera convention/calibration mismatch for {name}: {error_px}px")
            if self._last_calibration[name]["valid_depth_fraction"] <= 0.01:
                raise ValueError(f"insufficient finite positive depth for {name}")
        return report

    def get_planning_context(self):
        names, transforms, links = {}, {}, {}
        for arm, robot in self._robots.items():
            names[arm] = tuple(robot.arm_joints_name)
            transforms[arm] = self._base_poses[arm].as_matrix()
            links[arm] = robot.ee_link_name
        return RobotPlanningContext(embodiment=self.embodiment, model="dual_arx_x5",
            joint_names=names, base_transforms=transforms, end_effector_links=links)

    def _terminal(self):
        ended = bool(self._env.is_episode_end())
        success = ended and bool(self._env.success[0])
        truncated = ended and not success and self._env.take_action_cnt[0] >= self._env.step_lim
        return ended and not truncated, truncated

    def _native_gripper_target(self, arm, public_position):
        """Map physical aperture fraction to the benchmark's command range.

        Endpoints retain its intended close preload/open command. Intermediate
        targets use reachable physical positions, consistent with RobotState.
        """
        p=float(public_position)
        if p in (0.0,1.0):
            return p
        robot=self._robots[arm]
        physical_low,physical_high=self._gripper_limits[arm]
        fraction=p if robot.gripper_move["sign"]==1 else 1-p
        q=physical_low+fraction*(physical_high-physical_low)
        command_low,command_high=map(float,robot.gripper_scale)
        native=(q-command_low)/(command_high-command_low)
        if robot.gripper_move["sign"]!=1:
            native=1-native
        if not 0<=native<=1:
            raise ValueError("physical gripper target lies outside native command range")
        return native

    def step(self, action):
        if action.embodiment != self.embodiment:
            raise ValueError("action embodiment mismatch")
        terminated, truncated = self._terminal()
        if terminated or truncated:
            return StepResult(ok=False, observation=self.get_observation(), terminated=terminated,
                truncated=truncated, error=ApiError(ErrorCode.TRUNCATED if truncated else ErrorCode.TERMINATED,
                    "native episode already ended", recoverable=False))
        # Unspecified arms hold their last commanded joints. Re-reading the
        # measured positions here would turn each disturbance into a new target.
        commands = {a: (self._joint_targets[a].copy(), self._gripper_targets[a]) for a in self.arms}
        for arm, command in action.arms.items():
            if arm not in commands:
                raise ValueError(f"unknown arm {arm}")
            if tuple(command.target.shape) != (6,):
                raise ValueError("X5 joint target must contain 6 values")
            grip = commands[arm][1] if command.gripper_position is None else command.gripper_position
            commands[arm] = (command.target.copy(), grip)
        native = {}
        for arm, (joints, gripper) in commands.items():
            native[f"{arm}_arm_joint_state"] = np.array(joints, copy=True)
            native[f"{arm}_ee_joint_state"] = np.array([self._native_gripper_target(arm,gripper)])
        before = int(self._env.take_action_cnt[0])
        if self._recorder is not None:
            self._recorder.before_step(action)
        with route_native_progress():
            self._env.take_action(native)
        self._joint_targets = {a: v[0].copy() for a, v in commands.items()}
        self._gripper_targets = {a: float(v[1]) for a,v in commands.items()}
        after = int(self._env.take_action_cnt[0])
        if after != before + 1:
            raise RuntimeError("native action did not advance exactly one control period")
        observation = self.get_observation()
        terminated, truncated = self._terminal()
        result = StepResult(ok=True, observation=observation, terminated=terminated, truncated=truncated)
        if self._recorder is not None:
            self._recorder.after_step(action, result)
        return result

    def _execution(self, result, count, errors=None):
        terminal = result.terminated or result.truncated
        error = result.error
        if terminal:
            error = ApiError(ErrorCode.TRUNCATED if result.truncated else ErrorCode.TERMINATED,
                "native episode ended during execution", recoverable=False)
        return ExecutionResult(ok=result.ok and not terminal, steps_executed=count,
            final_observation=result.observation, terminated=result.terminated,
            truncated=result.truncated, error=error, final_errors=errors or {})

    def execute_trajectory(self, trajectory):
        if trajectory.embodiment != self.embodiment or not math.isclose(trajectory.dt_s, self.control_period_s, abs_tol=1e-9):
            raise ValueError("trajectory must use robodojo embodiment at 25Hz; no implicit retiming")
        state = self.get_robot_state()
        synchronized = isinstance(trajectory, SynchronizedTrajectory)
        positions = dict(trajectory.joint_positions) if synchronized else {trajectory.arm: trajectory.joint_positions}
        names = dict(trajectory.joint_names) if synchronized else {trajectory.arm: trajectory.joint_names}
        grippers = dict(trajectory.gripper_positions or {}) if synchronized else ({trajectory.arm: trajectory.gripper_positions} if trajectory.gripper_positions is not None else {})
        for arm in positions:
            if arm not in state.arms or tuple(names[arm]) != tuple(state.joint_names[arm]):
                raise ValueError("trajectory arm or joint ordering mismatch")
            if np.max(np.abs(state.joint_positions[arm] - trajectory.expected_start.joint_positions[arm])) > .05:
                return ExecutionResult(ok=False, steps_executed=0, error=ApiError(ErrorCode.STALE_PLAN, "start state differs by more than .05rad"))
        count = len(next(iter(positions.values())))
        final = None
        for index in range(count):
            action = RobotAction({arm: ArmCommand(mode="joint_position", target=values[index],
                gripper_position=float(grippers[arm][index]) if arm in grippers else None,
                embodiment=self.embodiment) for arm, values in positions.items()})
            final = self.step(action)
            if not final.ok or final.terminated or final.truncated:
                return self._execution(final, index + int(final.ok))
        errors = {arm: float(np.max(np.abs(final.observation.robot_state.joint_positions[arm] - values[-1]))) for arm, values in positions.items()}
        if max(errors.values()) > .05:
            return ExecutionResult(ok=False, steps_executed=count, final_observation=final.observation,
                error=ApiError(ErrorCode.CONTROLLER_FAILED, "final joint error exceeds .05rad",
                    details={"joint_error_rad": errors, "joint_tolerance_rad": .05}),
                final_errors=errors)
        result = self._execution(final, count, errors)
        # Label the existing measurement; preserve execution/tolerance semantics.
        return replace(result,
            diagnostics={**result.diagnostics,
                "joint_error_rad": errors, "joint_tolerance_rad": .05})

    def set_grippers(self, positions):
        if set(positions) != set(self.arms):
            raise ValueError("set_grippers requires both arms explicitly")
        # This plural atomic API promises exactly one control period. The
        # singular set_gripper/close_gripper primitives own aperture settlement.
        state=self.get_robot_state()
        result=self.step(RobotAction({a:ArmCommand(mode="joint_position",
            target=state.joint_positions[a],gripper_position=p,embodiment=self.embodiment)
            for a,p in positions.items()}))
        return self._execution(result,int(result.ok))

    def set_gripper(self, position, *, arm="primary"):
        arm = self.resolve_arm(arm)
        return self._drive_grippers({arm:position})

    def close_gripper(self, *, arm="primary"):
        return self._drive_grippers({self.resolve_arm(arm):0.0}, accept_stall=True)

    def _drive_grippers(self, positions, *, accept_stall=False):
        if not positions or set(positions)-set(self.arms):
            raise ValueError("unknown arm")
        positions={a:float(p) for a,p in positions.items()}
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in positions.values()):
            raise ValueError("gripper targets must be finite normalized positions")
        start=self.get_robot_state()
        recent={a:[] for a in positions}
        max_steps,tolerance,stall_window=25,.03,5
        final=None
        for index in range(max_steps):
            final=self.step(RobotAction({a:ArmCommand(mode="joint_position",
                target=start.joint_positions[a],gripper_position=p,embodiment=self.embodiment)
                for a,p in positions.items()}))
            if not final.ok or final.terminated or final.truncated:
                return self._execution(final,index+int(final.ok))
            measured={a:final.observation.robot_state.gripper_positions[a] for a in positions}
            for a in positions:
                recent[a].append(measured[a])
            errors={a:abs(measured[a]-positions[a]) for a in positions}
            reached=all(e<=tolerance for e in errors.values())
            stalled=accept_stall and all(positions[a]==0 and len(recent[a])>=stall_window
                and max(recent[a][-stall_window:])-min(recent[a][-stall_window:])<=.005
                for a in positions)
            if reached or stalled:
                return ExecutionResult(ok=True,steps_executed=index+1,
                    final_observation=final.observation,final_errors=errors,
                    diagnostics={"completion":"target_reached" if reached else "stable_aperture_stall",
                        "target_positions":positions,"measured_positions":measured,
                        "tolerance":tolerance,"stall_window_steps":stall_window,
                        "note":"aperture settlement is not proof of object contact or grasp"})
        return ExecutionResult(ok=False,steps_executed=max_steps,
            final_observation=final.observation,final_errors=errors,
            error=ApiError(ErrorCode.TIMEOUT,"gripper did not reach measured aperture target within25ticks",
                details={"targets":positions,"measured":measured,"max_steps":max_steps}))

    def _native_result(self, raw, arm, start):
        if raw.get("status") != "Success":
            return {"ok": False, "diagnostics": {"native_status": str(raw.get("status"))}}
        path = np.asarray(raw["position"], dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 6 or not np.isfinite(path).all():
            raise ValueError("native planner returned invalid joint path")
        native_dt = float(self._manager.planner[self._robots[arm].robot_name].dt)
        sampled,retiming=retime_native_path(path,start.joint_positions[arm],
            native_dt_s=native_dt,control_dt_s=self.control_period_s)
        return {"ok": True, "joint_positions": sampled, "dt_s": self.control_period_s,
            "joint_names": start.joint_names[arm], "collision_aware": True,
            "diagnostics": {"backend": "RoboDojo native CuRobo", "collision_scope": "configured table and self collision only; no object ground-truth obstacles",
                "native_waypoints":len(path),"native_dt_s":native_dt,"control_waypoints":len(sampled),
                **retiming,
                "target_origin":"grasp point +0.1501m flange X", "target_orientation":"native X5 flange; ContactGraspNet convention uncalibrated"}}

    def native_plan_to_pose(self, robot_state, target_pose, *, arm="primary", gripper_position=None, **timing):
        arm = self.resolve_arm(arm)
        robot = self._robots[arm]
        planner = self._manager.planner[robot.robot_name]
        base=self._base_poses[arm]
        raw = planner.plan_path(robot_state.joint_positions[arm], self._flange_array(target_pose),
            real_robot_pose=np.concatenate((base.position,base.quaternion_wxyz)).tolist())
        return self._native_result(raw, arm, robot_state)

    def native_plan_to_joints(self, robot_state, target_joints, *, arm="primary", gripper_position=None, **timing):
        arm = self.resolve_arm(arm)
        planner = self._manager.planner[self._robots[arm].robot_name]
        return self._native_result(planner.plan_joint(robot_state.joint_positions[arm], target_joints), arm, robot_state)

    def get_controller_metadata(self):
        return {"embodiment":self.embodiment,"arms":list(self.arms),"control_hz":self.control_frequency,
            "end_effector_position":"grasp point center, world metres",
            "orientation":"native X5 flange WXYZ quaternion; flange +X points along tool approach",
            "straight_down_quaternion_wxyz":[.5,-.5,.5,.5],
            "gripper":"normalized0closed1open; state is measured finger position, not width in metres; omitted command remains latched",
            "gripper_physical_joint_limits_m":dict(self._gripper_limits),
            "gripper_mapping":"public positions normalize measured finger joint travel; per-joint limits are not total jaw opening; interior targets mapped to native scale; closed endpoint preserves native preload",
            "native_base_quaternion_input_norms":dict(self._base_quaternion_norms),
            "gripper_settlement":"set_gripper waits at most25ticks for aperture error<=.03; close_gripper also accepts5stableticks, not grasp proof; plural set_grippers is exactly1tick",
            "last_gripper_commands":dict(self._gripper_targets),
            "motion_planner":"native table+self collision; public point-cloud scene not consumed",
            "unsupported":["SAM3 segmentation","ContactGraspNet grasps","standalone IK","synchronized motion planning"],
            "home_joint_positions":{a:q.tolist() for a,q in self._home.items()}}

    def get_task_metadata(self):
        task=self.get_task_context()
        return {"task_name":task.task_name,"instruction":task.language,
            "step_limit":int(self._env.step_lim),"remaining_actions":int(self._env.step_lim-self._env.take_action_cnt[0])}

    def go_home(self, *, arm="primary"):
        arm=self.resolve_arm(arm)
        start=self.get_robot_state()
        raw=self.native_plan_to_joints(start,self._home[arm],arm=arm)
        if not raw.get("ok"):
            return ExecutionResult(ok=False,steps_executed=0,error=ApiError(ErrorCode.PLANNING_FAILED,"native home plan failed"))
        trajectory=Trajectory(joint_positions=raw["joint_positions"],dt_s=raw["dt_s"],
            joint_names=start.joint_names[arm],planner="robodojo-native",collision_aware=raw["collision_aware"],
            expected_start=start,arm=arm,embodiment=self.embodiment)
        return self.execute_trajectory(trajectory)

    def official_status(self):
        """Evaluator-only report; never included in public observations/tool results."""
        ended = bool(self._env.is_episode_end())
        reward = float(self._env.reward_manager.get_reward()[0])
        return {"official_success": bool(reward > 1 - 1e-3), "reward":reward,
            "episode_ended":ended,"environment_steps":int(self._env.take_action_cnt[0]),
            "step_limit":int(self._env.step_lim)}

    def close(self):
        # Existing RoboDojo main owns simulator teardown and video finalization.
        self._closed = True
