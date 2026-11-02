"""CPU contract tests only; no learned inference, simulator or network evidence."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cap_harness.contracts import Segmentation, SegmentationSet
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.providers.graspnet.client import ContactGraspNetProvider
from full_episode import FullEvaluationInterrupted
from perception_bridge import X5GraspCalibration
import perception_smoke_entry as smoke
from verify_adapter_offline import Env


CALIBRATION = X5GraspCalibration((0., 1., 0.), f"X5A.urdf sha256:{smoke.X5_URDF_SHA256} joint7/8")
CONFIG = {"sam3_url": "http://127.0.0.1:18114", "graspnet_url": "http://127.0.0.1:18115",
          "grasp_calibration": CALIBRATION.as_dict(), "provenance": {"cpu_test_only": True}}


class SmokeTests(unittest.TestCase):
    def test_only_the_exposed_smoke_state_is_accepted(self):
        valid = {"task": "general_pickup", "seed": 0, "layout_ids": "0",
                 "mode": "interface_smoke", "count": "1"}
        smoke.checked_smoke_identity(**valid)
        for key, changed in (("task", "stack_blocks"), ("seed", 1), ("layout_ids", "1"),
                             ("layout_ids", "0,1"), ("mode", "evaluation"), ("count", "native")):
            with self.subTest(key=key, changed=changed), self.assertRaises(ValueError):
                smoke.checked_smoke_identity(**{**valid, key: changed})

    def test_query_comes_from_public_instruction_or_generic_fallback(self):
        self.assertEqual(smoke.instruction_query("Pick up the mint green scissors by 10 cm."),
                         ("mint green scissors", "public_task_instruction"))
        self.assertEqual(smoke.instruction_query("Please lift a blue mug."),
                         ("blue mug", "public_task_instruction"))
        self.assertEqual(smoke.instruction_query("Complete the manipulation task."),
                         ("object", "generic_object_query"))

    def test_service_config_requires_local_ports_and_reviewed_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "services.json"
            path.write_text(json.dumps(CONFIG))
            value, calibration = smoke.load_service_config(path)
            self.assertEqual(value, CONFIG)
            self.assertEqual(calibration, CALIBRATION)
            for field, replacement in (("sam3_url", "https://example.com:8114"),
                                       ("sam3_url", "http://user:pass@127.0.0.1:8114"),
                                       ("graspnet_url", "http://127.0.0.1"),
                                       ("grasp_calibration", {**CALIBRATION.as_dict(), "source_id": "unverified"})):
                path.write_text(json.dumps({**CONFIG, field: replacement}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    smoke.load_service_config(path)

    def exercise(self, output, *, no_mask=False, unreachable=False):
        env = Env()
        env.eval_seed = 0
        env.eval_num = 1
        calls = []

        def segment(provider, camera, query):
            calls.append(("sam3", camera.rgb.copy(), query))
            if no_mask:
                return SegmentationSet(ok=False, error=ApiError(ErrorCode.PERCEPTION_FAILED, "test no mask"))
            mask = np.zeros(camera.depth_m.shape, dtype=bool)
            mask[7:14, 9:17] = True
            return SegmentationSet(ok=True, segmentations=[Segmentation(
                mask=mask, label=query, score=.9, camera_name=camera.frame, frame=camera.frame)])

        def grasp(provider, cloud, *, scene_point_cloud, max_candidates):
            from cap_harness.contracts import GraspSet
            calls.append(("cgn", cloud.points.copy(), scene_point_cloud.points.copy(), cloud.frame))
            rotation = np.diag([1., -1., -1.])
            contact = cloud.points[len(cloud.points) // 2]
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = contact + .01 * rotation[:, 0] - .1034 * rotation[:, 2]
            candidates = provider._decode_candidates(transform[None], np.array([.8]), contact[None],
                                                      frame=cloud.frame, max_candidates=max_candidates)
            return GraspSet(ok=True, grasps=candidates)

        if unreachable:
            for planner in env.robot_manager.planner.values():
                planner.plan_path = lambda *args, **kwargs: {"status": "Unreachable"}
        with patch("full_adapter.task_records", return_value={"general_pickup": {"task_id": 0, "dimension": "Open"}}), \
             patch("cap_harness.providers.sam3.client.Sam3Provider.health", return_value=True), \
             patch("cap_harness.providers.graspnet.client.ContactGraspNetProvider.health", return_value=True), \
             patch("cap_harness.providers.sam3.client.Sam3Provider.segment_text", segment), \
             patch.object(ContactGraspNetProvider, "generate_grasps", grasp), \
             patch("requests.Session.request", side_effect=AssertionError("CPU test must never use network")):
            if no_mask:
                with self.assertRaises(FullEvaluationInterrupted):
                    smoke.run_perception_smoke(env, output, CONFIG, CALIBRATION)
                report = json.loads((Path(output) / "smoke_report.json").read_text())
            else:
                report = smoke.run_perception_smoke(env, output, CONFIG, CALIBRATION)
        return env, calls, report

    def test_real_upstream_projection_and_conversion_only_execute_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            env, calls, report = self.exercise(directory)
            self.assertTrue(report["ok"])
            self.assertEqual([row[0] for row in calls], ["sam3", "cgn"])
            self.assertEqual(calls[0][2], "mint green scissors")
            self.assertEqual(calls[1][3], "world")
            self.assertEqual(len(calls[1][1]), 56)
            self.assertEqual(len(calls[1][2]), 24 * 32)
            self.assertEqual(report["model_requests"], 0)
            self.assertTrue(report["frame_conversion_ok"])
            self.assertTrue(report["native_planning_ok"])
            self.assertTrue(report["candidate_plan"]["ok"])
            self.assertFalse(report["predicted_grasp_executed"])
            self.assertEqual(env.take_action_cnt, [1])
            np.testing.assert_array_equal(env.last_action["right_arm_joint_state"], np.zeros(6))
            candidates = json.loads((Path(directory) / "grasp_candidates.json").read_text())
            self.assertEqual(candidates[0]["frame"], "world")
            self.assertAlmostEqual(candidates[0]["width_m"], .02)
            self.assertFalse(candidates[0]["metadata"]["width_is_commanded_opening"])
            for name in ("public_sensor_input.npz", "sam3_masks.npz", "public_point_clouds.npz",
                         "selected_mask.png", "selected_mask_overlay.png", "candidate_plan.npz"):
                self.assertTrue((Path(directory) / name).is_file(), name)

    def test_unreachable_prediction_is_distinct_from_working_native_planner(self):
        with tempfile.TemporaryDirectory() as directory:
            env, _, report = self.exercise(directory, unreachable=True)
            self.assertTrue(report["ok"])
            self.assertFalse(report["candidate_plan"]["ok"])
            self.assertFalse(report["candidate_plan"]["executed"])
            self.assertTrue(report["native_planning_ok"])
            self.assertEqual(env.take_action_cnt, [1])

    def test_empty_segmentation_is_saved_and_stops_before_grasp_or_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            env, calls, report = self.exercise(directory, no_mask=True)
            self.assertFalse(report["ok"])
            self.assertFalse(report["segmentation"]["ok"])
            self.assertIn("no nonempty", report["error"])
            self.assertEqual([row[0] for row in calls], ["sam3"])
            self.assertEqual(env.take_action_cnt, [0])
            self.assertTrue((Path(directory) / "public_sensor_input.npz").is_file())

    def test_vendor_source_changes_fail_before_inference(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.py").write_text("known convention")
            hashes = {"source.py": hashlib.sha256(b"known convention").hexdigest()}
            with patch.object(smoke, "CGN_FILES", hashes):
                self.assertEqual(smoke.verify_vendor_source(root)["files_sha256"], hashes)
                (root / "source.py").write_text("changed convention")
                with self.assertRaisesRegex(ValueError, "differs"):
                    smoke.verify_vendor_source(root)


if __name__ == "__main__":
    unittest.main()
