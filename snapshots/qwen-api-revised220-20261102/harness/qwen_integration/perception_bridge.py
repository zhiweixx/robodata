"""Geometry-only adaptation of ASPIRE perception to the public X5 tool frame.

The learned models are the upstream SAM3 and Contact-GraspNet providers. This
module neither predicts grasps nor reads simulator object/contact state. It
converts Contact-GraspNet's Panda-frame proposals into this deployment's X5
grasp-center poses; calibration must be supplied from robot geometry evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Mapping

import numpy as np

from cap_harness.api import CapApi
from cap_harness.contracts import GraspCandidate, GraspSet
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import matrix_to_pose, select_top_down_grasp


def configure_perception_metadata(adapter, *, sam3: bool, contact_graspnet: bool,
                                  calibration: "X5GraspCalibration | None"):
    """Expose provider configuration without modifying native robot controls.

    The episode also queries the original adapter directly for its public
    observation, so this narrowly wraps that instance's metadata method. No
    wrapper is installed in the baseline profile.
    """
    if not sam3 and not contact_graspnet:
        return
    original = getattr(adapter, "_robodojo_original_controller_metadata", None)
    if original is None:
        original = adapter.get_controller_metadata
        adapter._robodojo_original_controller_metadata = original
    adapter._robodojo_perception = {
        "sam3": bool(sam3), "contact_graspnet": bool(contact_graspnet),
        "grasp_calibration": None if calibration is None else calibration.as_dict(),
    }

    def metadata(instance):
        result = dict(instance._robodojo_original_controller_metadata())
        providers = instance._robodojo_perception
        enabled = set()
        if providers["sam3"]:
            enabled.add("SAM3 segmentation")
        if providers["contact_graspnet"]:
            enabled.add("ContactGraspNet grasps")
        result["unsupported"] = [name for name in result.get("unsupported", [])
                                 if name not in enabled]
        result["perception_providers"] = dict(providers)
        return result

    adapter.get_controller_metadata = MethodType(metadata, adapter)


@dataclass(frozen=True)
class X5GraspCalibration:
    """Native-frame direction along which the opposing X5 fingers separate.

    The public X5 approach is already defined as native +X by RoboDojoAdapter.
    ``source_id`` identifies the robot-geometry evidence; it is not learned from
    evaluation scenes. An optional aperture bound must be a verified total jaw
    clearance, never an individual prismatic joint's travel.
    """

    native_closing_direction: tuple[float, float, float]
    source_id: str
    maximum_contact_separation_m: float | None = None

    def __post_init__(self):
        direction = np.asarray(self.native_closing_direction, dtype=float)
        if (direction.shape != (3,) or not np.isfinite(direction).all()
                or not np.isclose(np.linalg.norm(direction), 1., atol=1e-8)
                or not np.isclose(direction[0], 0., atol=1e-8)):
            raise ValueError("native closing direction must be unit and perpendicular to native +X")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("grasp calibration requires robot geometry source_id")
        if self.maximum_contact_separation_m is not None:
            bound = self.maximum_contact_separation_m
            if not np.isfinite(bound) or bound <= 0:
                raise ValueError("maximum contact separation must be finite and positive")
        object.__setattr__(self, "native_closing_direction", tuple(float(v) for v in direction))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]):
        allowed = {"native_closing_direction", "source_id", "maximum_contact_separation_m"}
        if not isinstance(value, Mapping) or set(value) - allowed:
            raise ValueError("invalid X5 grasp calibration fields")
        return cls(**dict(value))

    def as_dict(self):
        return {"native_closing_direction": list(self.native_closing_direction),
                "source_id": self.source_id,
                "maximum_contact_separation_m": self.maximum_contact_separation_m}

    def raw_from_native_rotation(self):
        # Raw Contact-GraspNet: X=jaw baseline, Z=positive approach. Map the
        # native approach and measured jaw axis onto those same physical axes.
        native_x = np.array([1., 0., 0.])
        native_jaw = np.asarray(self.native_closing_direction)
        native_basis = np.column_stack([native_x, native_jaw, np.cross(native_x, native_jaw)])
        raw_basis = np.column_stack([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        return raw_basis @ native_basis.T


def to_x5_grasp(candidate: GraspCandidate, calibration: X5GraspCalibration):
    """Convert one genuine upstream candidate, preserving score and world frame.

    Pinned CGN ``utils.build_6d_grasp`` defines raw translation as
    contact + thickness/2 * raw_X - gripper_depth * raw_Z. ASPIRE's client
    right-multiplies its rotation by local Z +90 degrees. Undo that rotation,
    recover contact separation by projection, and place the X5 TCP at the
    predicted contact midpoint. This also handles a checkpoint center_to_tip
    shift along raw_Z without guessing the Panda-to-X5 translation offset.
    """
    if candidate.metadata.get("orientation_correction") != "right_multiply_local_z_90_deg":
        raise ValueError("unexpected Contact-GraspNet frame convention")
    contact = np.asarray(candidate.metadata.get("contact_point"), dtype=float)
    if contact.shape != (3,) or not np.isfinite(contact).all():
        raise ValueError("Contact-GraspNet predicted contact_point is required")
    corrected = candidate.pose.as_matrix()
    correction_inverse = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    raw_rotation = corrected[:3, :3] @ correction_inverse
    difference = candidate.pose.position - contact
    half_separation = float(np.dot(difference, raw_rotation[:, 0]))
    if half_separation < -1e-6:
        raise ValueError("predicted contact separation is negative")
    if abs(float(np.dot(difference, raw_rotation[:, 1]))) > 1e-4:
        raise ValueError("predicted contact and grasp pose violate the CGN baseline convention")
    separation = 2. * max(0., half_separation)
    if (calibration.maximum_contact_separation_m is not None
            and separation > calibration.maximum_contact_separation_m + 1e-6):
        raise ValueError("predicted contact separation exceeds calibrated X5 jaw clearance")
    transform = np.eye(4)
    transform[:3, :3] = raw_rotation @ calibration.raw_from_native_rotation()
    transform[:3, 3] = contact + max(0., half_separation) * raw_rotation[:, 0]
    metadata = {**dict(candidate.metadata),
                "source_provider": "contact_graspnet",
                "source_pose_position": tuple(float(v) for v in candidate.pose.position),
                "source_pose_quaternion_wxyz": tuple(float(v) for v in candidate.pose.quaternion_wxyz),
                "source_pose_convention": "raw Contact-GraspNet right-multiplied local Z +90deg",
                "pose_convention": "X5 grasp-center origin; native flange orientation; approach +X",
                "width_source": "predicted contact separation reconstructed from CGN pose/contact",
                "width_is_commanded_opening": False,
                "source_gripper_depth_m": -float(np.dot(difference, raw_rotation[:, 2])),
                "grasp_calibration_source": calibration.source_id}
    # Do not propagate a source approach distance as an X5-specific motion rule.
    return GraspCandidate(pose=matrix_to_pose(transform, frame=candidate.frame),
                          score=candidate.score, width_m=separation,
                          approach_distance_m=0., metadata=metadata)


class RoboDojoContactGraspNetProvider:
    """Keep upstream inference, adapt only its robot-specific output geometry."""

    def __init__(self, provider, calibration: X5GraspCalibration):
        if not isinstance(calibration, X5GraspCalibration):
            raise ValueError("explicit X5GraspCalibration is required")
        self.provider = provider
        self.calibration = calibration

    @property
    def base_url(self):
        return self.provider.base_url

    def health(self):
        return self.provider.health()

    def generate_grasps(self, point_cloud, **options):
        result = self.provider.generate_grasps(point_cloud, **options)
        if not result.ok:
            return result
        grasps, rejected = [], []
        for candidate in result.grasps:
            try:
                grasps.append(to_x5_grasp(candidate, self.calibration))
            except (TypeError, ValueError) as exc:
                rejected.append(str(exc))
        diagnostics = {**dict(result.diagnostics), "robot_geometry_adaptation": "contact_graspnet_to_x5",
                       "grasp_calibration": self.calibration.as_dict(),
                       "source_candidate_count": len(result.grasps),
                       "rejected_candidate_count": len(rejected),
                       "rejected_reasons": rejected}
        if not grasps:
            return GraspSet(ok=False, error=ApiError(ErrorCode.NO_GRASP,
                "No Contact-GraspNet candidates satisfy the X5 output geometry contract",
                details=diagnostics), diagnostics=diagnostics)
        return GraspSet(ok=True, grasps=grasps, diagnostics=diagnostics)


class RoboDojoPerceptionApi(CapApi):
    """Adapt only top-down grasp selection to the public X5 approach axis."""

    def select_grasp(self, grasps: GraspSet, strategy: str = "top_down") -> GraspCandidate | None:
        if strategy == "top_down":
            return select_top_down_grasp(grasps, approach_axis=0)
        return super().select_grasp(grasps, strategy=strategy)
