"""Connect the upstream ASPIRE runtime to the separate RoboDojo adapter.

This module does not implement a replacement code executor or perception model.
It instantiates upstream CapApi and a ProgramExecutor subclass that preserves
public measured execution details in receipts (without changing tool execution),
Sam3Provider, ContactGraspNetProvider, and TracedProvider directly. The only new
planning backend wraps RoboDojo's own robot-specific planner via adapter hooks.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import inspect
from importlib.resources import files
from typing import Any

import numpy as np
import requests

from cap_harness.api import CapApi
from cap_harness.artifacts import RunRecorder, TracedProvider
from cap_harness.contracts import (
    JOINT_DIMENSIONS,
    MotionStrategy,
    PlanResult,
    Pose,
    RobotState,
    SynchronizedPlanResult,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.providers.graspnet.client import ContactGraspNetProvider
from cap_harness.providers.sam3.client import Sam3Provider
from cap_harness.registry import ToolRegistry
from cap_harness.run import run_program_with_api
from robodata.execution.receipt_executor import run_program_with_receipts
from robodata.perception.perception_bridge import (
    RoboDojoContactGraspNetProvider,
    RoboDojoPerceptionApi,
    X5GraspCalibration,
    configure_perception_metadata,
)

UPSTREAM_COMMIT = "fa00f0ef2bc1d91d594b9aa502c22133c9f538f7"
PLANNER_NAME = "robodojo-native"
EXTENSION_ALLOWLIST = frozenset(
    {"robodojo.get_task_metadata", "robodojo.get_controller_metadata"}
)


class RoboDojoNativePlanner:
    """Adapt native robot plans without replacing or executing their trajectories.

    The adapter owns simulator/frame conversion. Each callback returns a mapping
    with ok, joint_positions (N,6), dt_s, joint_names, collision_aware, diagnostics.
    No target object pose, success predicate, or simulator contact truth is read
    by this provider. collision_aware is never inferred from a planner's name.
    """

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    @staticmethod
    def _failure(message: str, code: ErrorCode = ErrorCode.PLANNING_FAILED) -> PlanResult:
        return PlanResult(ok=False, error=ApiError(code, message))

    def _plan(self, method: str, state: RobotState, goal: Any, *, arm: str,
              gripper_position: float | None, **options: Any) -> PlanResult:
        if state.embodiment != "robodojo" or arm not in state.arms:
            return self._failure("native planner needs a RoboDojo state and known arm",
                                 ErrorCode.INVALID_REQUEST)
        # Scene/context are passed by the real CapApi. The native planner owns
        # its own configured scene, so do not claim these public points are its
        # collision model. Preserve that distinction in trace diagnostics.
        context = options.pop("context", None)
        scene = options.pop("scene", None)
        if any(value is not None for value in options.values()):
            return self._failure(
                "custom timing options are not supported by this native-planner bridge",
                ErrorCode.UNSUPPORTED,
            )
        if context is not None and context.model != "dual_arx_x5":
            return self._failure("planning context is not dual_arx_x5",
                                 ErrorCode.INVALID_REQUEST)
        if isinstance(goal, Pose) and goal.frame != state.base_frame:
            return self._failure("target and robot state must share world frame",
                                 ErrorCode.INVALID_REQUEST)
        callback = getattr(self.adapter, method, None)
        if not callable(callback):
            return self._failure(f"adapter does not implement {method}", ErrorCode.UNSUPPORTED)
        try:
            raw = callback(state, goal, arm=arm, gripper_position=gripper_position, **options)
            if not raw.get("ok", False):
                return self._failure(str(raw.get("error", "native planner found no plan")))
            joints = np.asarray(raw["joint_positions"], dtype=np.float64)
            if joints.ndim != 2 or joints.shape[1] != 6 or not len(joints):
                return self._failure("native trajectory must be nonempty N x 6")
            if not np.isfinite(joints).all():
                return self._failure("native trajectory contains nonfinite joints")
            dt = float(raw["dt_s"])
            period = float(self.adapter.control_period_s)
            if not np.isfinite(dt) or not np.isclose(dt, period, rtol=0, atol=1e-9):
                return self._failure("native trajectory dt must equal simulator control period")
            names = tuple(raw.get("joint_names", state.joint_names[arm]))
            if names != tuple(state.joint_names[arm]):
                return self._failure("native trajectory joint order differs from observation")
            collision_aware = raw.get("collision_aware")
            if type(collision_aware) is not bool:
                return self._failure("adapter must explicitly declare collision_aware")
            # Measured aperture is not the command: a held object can keep the
            # fingers apart while the controller still commands closure. Omit
            # the target unless the caller supplies its latched command.
            gripper_targets = (None if gripper_position is None
                               else np.full(len(joints), float(gripper_position)))
            trajectory = Trajectory(
                joint_positions=joints,
                dt_s=dt,
                joint_names=names,
                planner=PLANNER_NAME,
                collision_aware=collision_aware,
                expected_start=state,
                arm=arm,
                gripper_positions=gripper_targets,
                embodiment="robodojo",
            )
            diagnostics = dict(raw.get("diagnostics", {}))
            perception = getattr(self.adapter, "_robodojo_perception", {})
            if perception.get("contact_graspnet"):
                diagnostics["target_orientation"] = (
                    "native X5 flange; Contact-GraspNet proposals converted to native +X approach")
                diagnostics["grasp_calibration"] = perception["grasp_calibration"]
            diagnostics.update(provider=PLANNER_NAME,
                               public_planning_scene_supplied=scene is not None,
                               public_planning_scene_consumed=False,
                               planner_is_upstream_aspire_curobo_service=False)
            return PlanResult(ok=True, trajectory=trajectory, diagnostics=diagnostics)
        except Exception as exc:
            return self._failure(f"native adapter plan failed: {type(exc).__name__}: {exc}")

    def plan_to_pose(self, robot_state: RobotState, target_pose: Pose, *,
                     arm: str = "right", gripper_position: float | None = None,
                     **options: Any) -> PlanResult:
        return self._plan("native_plan_to_pose", robot_state, target_pose, arm=arm,
                          gripper_position=gripper_position, **options)

    def plan_to_joints(self, robot_state: RobotState, target_joints: np.ndarray, *,
                       arm: str = "right", gripper_position: float | None = None,
                       **options: Any) -> PlanResult:
        return self._plan("native_plan_to_joints", robot_state, target_joints, arm=arm,
                          gripper_position=gripper_position, **options)

    def plan_synchronized_motion(self, *args: Any, **kwargs: Any) -> SynchronizedPlanResult:
        return SynchronizedPlanResult(
            ok=False,
            error=ApiError(ErrorCode.UNSUPPORTED,
                           "jointly collision-checked bimanual planning is not implemented"),
        )


def build_api(adapter: Any, *, recorder: RunRecorder | None = None,
              sam3_url: str | None = None, graspnet_url: str | None = None,
              grasp_calibration: X5GraspCalibration | None = None,
              default_camera: str = "cam_head") -> CapApi:
    """Build authentic ASPIRE API; omit unavailable providers, never fake them."""
    if JOINT_DIMENSIONS.get("robodojo") != 6:
        raise RuntimeError("Apply integration/patches/0001-robodojo-contracts.patch first")
    if bool(graspnet_url) != (grasp_calibration is not None):
        raise ValueError("Contact-GraspNet URL and explicit X5 grasp calibration must be supplied together")

    def local_session():
        # Services are node-local. Cluster proxy settings are needed by Isaac
        # asset loading, but must not reroute sensor payloads sent to providers.
        session = requests.Session()
        session.trust_env = False
        return session

    def traced(provider: Any, name: str) -> Any:
        return TracedProvider(provider, recorder, name) if recorder is not None else provider

    planner = traced(RoboDojoNativePlanner(adapter), PLANNER_NAME)
    segmentation = (traced(Sam3Provider(sam3_url, session=local_session()), "sam3") if sam3_url else None)
    grasps = ({"contact-graspnet": traced(RoboDojoContactGraspNetProvider(
                    ContactGraspNetProvider(graspnet_url, session=local_session()), grasp_calibration),
                    "contact_graspnet")} if graspnet_url else {})
    api_class = RoboDojoPerceptionApi if graspnet_url else CapApi
    api = api_class(
        adapter,
        segmentation_provider=segmentation,
        grasp_providers=grasps,
        trajectory_planning_providers={PLANNER_NAME: planner},
        integrated_pose_planning_providers={PLANNER_NAME: planner},
        default_camera=default_camera,
        default_motion_strategy=MotionStrategy(trajectory_planner=PLANNER_NAME,
                                               pose_planner=PLANNER_NAME),
        recorder=recorder,
    )
    api._robodojo_perception = {"sam3": bool(sam3_url), "contact_graspnet": bool(graspnet_url),
        "grasp_calibration": None if grasp_calibration is None else grasp_calibration.as_dict()}
    configure_perception_metadata(adapter, sam3=bool(sam3_url),
                                  contact_graspnet=bool(graspnet_url), calibration=grasp_calibration)
    return api


def execute_program(api: CapApi, program_path: str | Path,
                    recorder: RunRecorder | None = None):
    """The caller owns reset, recording callbacks, evaluation and finalization."""
    return run_program_with_receipts(
        api=api, program_path=Path(program_path),
        extension_allowlist=EXTENSION_ALLOWLIST, recorder=recorder,
    )


def render_api_docs(api: CapApi) -> str:
    """Render the actual deployed ASPIRE registry for the coding-agent prompt."""
    registry = api.register_tools(ToolRegistry(public_extension_allowlist=EXTENSION_ALLOWLIST))
    return registry.render_docs()


def implementation_provenance() -> dict[str, Any]:
    """Record which real upstream files ran, separate from the integration code."""
    from cap_harness import contracts, registry, runtime
    modules = {"api": inspect.getmodule(CapApi), "contracts": contracts,
               "registry": registry, "runtime": runtime,
               "run": inspect.getmodule(run_program_with_api)}
    receipt_module = inspect.getmodule(run_program_with_receipts)
    perception_module = inspect.getmodule(RoboDojoContactGraspNetProvider)
    return {
        "expected_upstream_commit": UPSTREAM_COMMIT,
        "integration_backend": PLANNER_NAME,
        "upstream_patches": [
            {"name": name, "sha256": hashlib.sha256(
                (files("cap_harness") / "robodata_patches" / name).read_bytes()).hexdigest()}
            for name in ("0001-robodojo-contracts.patch", "0002-trajectory-gripper-cache.patch")
        ],
        "receipt_extension": {
            "path": str(Path(receipt_module.__file__).resolve()),
            "sha256": hashlib.sha256(Path(receipt_module.__file__).read_bytes()).hexdigest(),
            "scope": "public measured execution details only; inherits upstream program executor",
        },
        "perception_extension": {
            "path": str(Path(perception_module.__file__).resolve()),
            "sha256": hashlib.sha256(Path(perception_module.__file__).read_bytes()).hexdigest(),
            "scope": "optional upstream providers plus geometry-only CGN-to-X5 adaptation",
        },
        "upstream_modules": {
            name: {"path": str(Path(module.__file__).resolve()),
                   "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
            for name, module in modules.items()
        },
        "scope": "ASPIRE runtime/API with custom RoboDojo adapter and native planner",
    }
