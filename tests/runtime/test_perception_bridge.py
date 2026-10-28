"""CPU contract tests; these are not perception accuracy or robot grasp trials."""
from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np

from cap_harness.api import CapApi
from cap_harness.contracts import GraspCandidate, GraspSet, PointCloud, Segmentation, SegmentationSet
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import matrix_to_pose
from cap_harness.providers.graspnet.client import ContactGraspNetProvider
from robodata.execution.aspire_glue import RoboDojoNativePlanner, build_api, execute_program
from robodata.execution.deployment_docs import deployed_api_documentation
from robodata.perception.perception_bridge import (RoboDojoContactGraspNetProvider, RoboDojoPerceptionApi,
                               X5GraspCalibration, to_x5_grasp)
from robodata.environments.robodojo.robodojo_adapter import RoboDojoAdapter
from tests.runtime.fixtures import Env


CALIBRATION = X5GraspCalibration((0., 1., 0.), "synthetic CPU-test X5 geometry")


def upstream_candidate(rotation=None, contact=None, separation=.04, depth=.1034, score=.8):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    contact = np.array([.3, .2, .8]) if contact is None else np.asarray(contact)
    raw = np.eye(4)
    raw[:3, :3] = rotation
    raw[:3, 3] = contact + separation/2*rotation[:, 0] - depth*rotation[:, 2]
    provider = ContactGraspNetProvider("http://127.0.0.1:8115")
    return provider._decode_candidates(raw[None], np.array([score]), contact[None],
                                       frame="world", max_candidates=None)[0]


class GeometryTests(unittest.TestCase):
    def test_native_approach_and_jaw_axes_match_raw_grasp(self):
        source = upstream_candidate()
        converted = to_x5_grasp(source, CALIBRATION)
        matrix = converted.pose.as_matrix()
        np.testing.assert_allclose(matrix[:3, 0], [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(matrix[:3, 1], [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(matrix[:3, 2], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(matrix[:3, 3], [.32, .2, .8], atol=1e-12)
        self.assertEqual(converted.frame, "world")
        self.assertEqual(converted.score, source.score)
        self.assertAlmostEqual(converted.width_m, .04)
        self.assertNotEqual(converted.width_m, source.width_m)  # upstream default is .08
        self.assertFalse(converted.metadata["width_is_commanded_opening"])

    def test_conversion_commutes_with_arbitrary_world_transform(self):
        axis = np.array([.3, -.8, .5]); axis /= np.linalg.norm(axis)
        skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        rotation = np.eye(3) + np.sin(.73)*skew + (1-np.cos(.73))*(skew@skew)
        contact = np.array([1.5, -.2, .6])
        source = upstream_candidate(rotation, contact, separation=.027)
        converted = to_x5_grasp(source, CALIBRATION)
        np.testing.assert_allclose(converted.pose.position, contact+.0135*rotation[:, 0], atol=1e-12)
        np.testing.assert_allclose(converted.pose.as_matrix()[:3, 0], rotation[:, 2], atol=1e-12)
        np.testing.assert_allclose(converted.pose.as_matrix()[:3, 1], rotation[:, 0], atol=1e-12)
        self.assertAlmostEqual(converted.width_m, .027)

    def test_checkpoint_depth_shift_does_not_move_predicted_contact_center(self):
        near = to_x5_grasp(upstream_candidate(depth=.1034), CALIBRATION)
        far = to_x5_grasp(upstream_candidate(depth=.1184), CALIBRATION)
        np.testing.assert_allclose(near.transform, far.transform, atol=1e-12)
        self.assertAlmostEqual(far.metadata["source_gripper_depth_m"], .1184)

    def test_calibration_rejects_invented_or_nonorthogonal_axes(self):
        for direction in [(1, 0, 0), (0, 2, 0), (0, float("nan"), 0)]:
            with self.assertRaises(ValueError):
                X5GraspCalibration(direction, "test")
        with self.assertRaises(ValueError):
            X5GraspCalibration((0, 1, 0), "")
        with self.assertRaises(ValueError):
            X5GraspCalibration.from_mapping({"native_closing_direction": [0, 1, 0],
                                            "source_id": "test", "unverified_offset": .1})

    def test_missing_predicted_contact_or_unknown_convention_is_rejected(self):
        source = upstream_candidate()
        for metadata in [{}, {"orientation_correction": "right_multiply_local_z_90_deg"}]:
            candidate = GraspCandidate(source.pose, source.score, source.width_m, metadata=metadata)
            with self.assertRaises(ValueError):
                to_x5_grasp(candidate, CALIBRATION)

    def test_bad_contact_geometry_and_verified_aperture_bound(self):
        with self.assertRaisesRegex(ValueError, "negative"):
            to_x5_grasp(upstream_candidate(separation=-.02), CALIBRATION)
        limited = X5GraspCalibration((0, 1, 0), "verified total jaw-clearance fixture", .03)
        with self.assertRaisesRegex(ValueError, "clearance"):
            to_x5_grasp(upstream_candidate(separation=.04), limited)
        source = upstream_candidate()
        metadata = {**dict(source.metadata), "contact_point": [.3, .21, .8]}
        invalid = GraspCandidate(source.pose, source.score, source.width_m, metadata=metadata)
        with self.assertRaisesRegex(ValueError, "baseline convention"):
            to_x5_grasp(invalid, CALIBRATION)


class ProviderTests(unittest.TestCase):
    def test_controller_metadata_only_changes_enabled_provider_fields(self):
        adapter = RoboDojoAdapter(Env())
        before = adapter.get_controller_metadata()
        build_api(adapter)
        self.assertEqual(adapter.get_controller_metadata(), before)
        self.assertNotIn("get_controller_metadata", adapter.__dict__)
        build_api(adapter, sam3_url="http://127.0.0.1:8114")
        partial = adapter.get_controller_metadata()
        self.assertNotIn("SAM3 segmentation", partial["unsupported"])
        self.assertIn("ContactGraspNet grasps", partial["unsupported"])
        build_api(adapter, sam3_url="http://127.0.0.1:8114",
                  graspnet_url="http://127.0.0.1:8115", grasp_calibration=CALIBRATION)
        after = adapter.get_controller_metadata()
        self.assertEqual(after["unsupported"], ["standalone IK", "synchronized motion planning"])
        self.assertEqual(after["perception_providers"]["grasp_calibration"], CALIBRATION.as_dict())
        for key in before.keys() - {"unsupported"}:
            self.assertEqual(after[key], before[key])

    def test_plan_diagnostics_match_frame_without_changing_motion(self):
        adapter = RoboDojoAdapter(Env())
        state = adapter.get_robot_state()
        planner = RoboDojoNativePlanner(adapter)
        goal = state.end_effector_poses["right"]
        baseline = planner.plan_to_pose(state, goal, arm="right")
        self.assertIn("uncalibrated", baseline.diagnostics["target_orientation"])
        build_api(adapter, graspnet_url="http://127.0.0.1:8115", grasp_calibration=CALIBRATION)
        treatment = planner.plan_to_pose(state, goal, arm="right")
        self.assertTrue(baseline.ok and treatment.ok)
        self.assertNotIn("uncalibrated", treatment.diagnostics["target_orientation"])
        np.testing.assert_array_equal(baseline.trajectory.joint_positions, treatment.trajectory.joint_positions)
        self.assertEqual(treatment.trajectory.dt_s, baseline.trajectory.dt_s)
        self.assertFalse(treatment.diagnostics["public_planning_scene_consumed"])
        self.assertEqual(adapter.native_env.take_action_cnt, [0])

    def test_provider_preserves_failure_and_options(self):
        error = GraspSet(ok=False, error=ApiError(ErrorCode.NO_GRASP, "none"))
        class Provider:
            def generate_grasps(self, cloud, **options):
                self.got = (cloud, options)
                return error
        source = Provider()
        wrapper = RoboDojoContactGraspNetProvider(source, CALIBRATION)
        cloud = PointCloud(np.array([[0, 0, 1.]]), "world")
        self.assertIs(wrapper.generate_grasps(cloud, max_candidates=7), error)
        self.assertIs(source.got[0], cloud)
        self.assertEqual(source.got[1], {"max_candidates": 7})

    def test_provider_records_rejected_candidates_without_fabricating_grasps(self):
        class Provider:
            def generate_grasps(self, cloud, **options):
                return GraspSet(ok=True, grasps=[upstream_candidate(separation=-.02)])
        result = RoboDojoContactGraspNetProvider(Provider(), CALIBRATION).generate_grasps(None)
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics["rejected_candidate_count"], 1)
        self.assertEqual(result.grasps, ())

    def test_top_down_selection_uses_x5_x_axis(self):
        # Raw +Z points down, raw X remains horizontal.
        down = np.diag([1., -1., -1.])
        candidate = to_x5_grasp(upstream_candidate(down), CALIBRATION)
        api = RoboDojoPerceptionApi(RoboDojoAdapter(Env()))
        chosen = api.select_grasp(GraspSet(ok=True, grasps=[candidate]), "top_down")
        self.assertIs(chosen, candidate)
        # The upstream Z-axis selector would reject this converted pose.
        baseline = CapApi(RoboDojoAdapter(Env()))
        self.assertIsNone(baseline.select_grasp(GraspSet(ok=True, grasps=[candidate]), "top_down"))

    def test_explicit_wiring_preserves_baseline_and_does_not_start_services(self):
        adapter = RoboDojoAdapter(Env())
        baseline = build_api(adapter)
        self.assertIs(type(baseline), CapApi)
        self.assertIsNone(baseline._segmentation_provider)
        self.assertEqual(baseline._grasp_providers, {})
        with self.assertRaises(ValueError):
            build_api(adapter, graspnet_url="http://127.0.0.1:8115")
        with patch("requests.Session.request", side_effect=AssertionError("no network in construction")):
            api = build_api(adapter, sam3_url="http://127.0.0.1:8114",
                            graspnet_url="http://127.0.0.1:8115", grasp_calibration=CALIBRATION)
        self.assertIs(type(api), RoboDojoPerceptionApi)
        self.assertFalse(api._segmentation_provider._http.session.trust_env)
        self.assertFalse(api._grasp_providers["contact-graspnet"].provider._http.session.trust_env)
        self.assertEqual(api.default_motion_strategy, baseline.default_motion_strategy)
        self.assertEqual(adapter.native_env.take_action_cnt, [0])

    def test_perception_docs_explain_availability_and_native_frames_only(self):
        baseline = build_api(RoboDojoAdapter(Env()))
        self.assertIn("There is no SAM3", deployed_api_documentation(baseline))
        api = build_api(RoboDojoAdapter(Env()), sam3_url="http://127.0.0.1:8114",
                        graspnet_url="http://127.0.0.1:8115", grasp_calibration=CALIBRATION)
        docs = deployed_api_documentation(api)
        self.assertNotIn("There is no SAM3", docs)
        self.assertIn("already converted", docs)
        self.assertIn("predicted contact separation", docs)
        self.assertIn("same native RoboDojo planner", docs)
        from robodata.eval.robodojo.full_deployment_docs import full_api_documentation
        self.assertIn("There is no harness cap on program count", full_api_documentation(api))
        self.assertNotIn("300s Python watchdog", full_api_documentation(api))

    def test_generated_program_can_compose_segmentation_grasp_and_selection(self):
        class Segmenter:
            def segment_text(self, camera, query):
                mask = np.zeros(camera.depth_m.shape, dtype=bool)
                mask[10:14, 14:18] = True
                return SegmentationSet(ok=True, segmentations=[Segmentation(
                    mask=mask, label=query, score=.9, camera_name=camera.frame, frame=camera.frame)])
        class Grasper:
            def generate_grasps(self, cloud, **options):
                self.cloud, self.options = cloud, options
                return GraspSet(ok=True, grasps=[upstream_candidate(np.diag([1., -1., -1.]))])
        source = Grasper()
        env = Env()
        api = RoboDojoPerceptionApi(RoboDojoAdapter(env), segmentation_provider=Segmenter(),
            grasp_providers={"contact-graspnet": RoboDojoContactGraspNetProvider(source, CALIBRATION)})
        code = '''
masks = segment_text("cam_head", "test object")
grasps = generate_grasps("cam_head", masks.segmentations[0])
chosen = select_grasp(grasps, strategy="top_down")
result = {"position": chosen.pose.position.tolist(), "width": chosen.width_m}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "program.py"
            path.write_text(code)
            outcome = execute_program(api, path)
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(source.cloud.frame, "world")
        self.assertEqual(source.options["scene_point_cloud"].frame, "world")
        self.assertEqual(len(source.cloud.points), 16)
        self.assertAlmostEqual(outcome.result["width"], .04)
        self.assertEqual([call.tool for call in outcome.calls],
                         ["segment_text", "generate_grasps", "select_grasp"])
        self.assertEqual(env.take_action_cnt, [0])


if __name__ == "__main__":
    unittest.main()
