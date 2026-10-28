"""Actual pinned CGN postprocessing; only neural outputs/device wrappers are mocked."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

import robodata.perception.cgn_singleton_fix as fix


FIXTURE = Path(__file__).with_name("test_fixtures") / "cgn_postprocessing.py.txt"


class Tensor:
    def __init__(self, value): self.value = value
    def type(self, dtype): return self
    def to(self, device): return self
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value.copy()


class SingletonFixTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "estimator.py"
        self.source = FIXTURE.read_text()
        self.path.write_text(self.source)
        self.module = ModuleType("test_cgn_vendor")
        self.module.__file__ = str(self.path)
        self.module.__dict__.update(np=np,
            torch=NS(from_numpy=Tensor, float32=np.float32),
            preprocess_pc_for_inference=lambda pc, *a, **kw: (pc, np.zeros(3)),
            regularize_pc_point_count=lambda pc, count: pc)
        exec(compile(self.source, str(self.path), "exec"), vars(self.module))
        # Only the dependency-file guards are substituted for this test excerpt.
        # Production accepts the full original vendor SHA256 exclusively.
        with patch.multiple(fix,
                UPSTREAM_SOURCE_SHA256=hashlib.sha256(self.source.encode()).hexdigest(),
                PATCHED_SOURCE_SHA256=hashlib.sha256(fix._rewrite_source(self.source).encode()).hexdigest()):
            self.corrected, self.evidence = fix.corrected_estimator_class(self.module)
        self.points = np.array([[.2, .3, .4], [.3, .4, .5]], dtype=np.float32)

    def tearDown(self):
        self.folder.cleanup()

    def estimator(self, selected, *, patched=True):
        cls = self.corrected if patched else self.module.GraspEstimator
        obj = cls()
        obj.device, obj._num_input_points = "cpu", 2
        obj._contact_grasp_cfg = {"DATA": {"gripper_width": .08, "num_point": 2, "raw_num_points": 2},
            "TEST": {"extra_opening": .005, "max_farthest_points": 2, "num_samples": 2,
                     "first_thres": .4, "second_thres": .2, "with_replacement": False,
                     "center_to_tip": .02, "filter_thres": .0001}}
        obj.forward_calls, obj.selection_calls = [], []
        def forward(batch):
            obj.forward_calls.append(batch.value.copy())
            return {"pred_grasps_cam": Tensor(np.tile(np.eye(4), (1, 2, 1, 1))),
                    "pred_scores": Tensor(np.array([[.9, .8]])),
                    "pred_points": Tensor(self.points[None]),
                    "offset_pred": Tensor(np.array([[.02, .09]]))}
        def select(*args, **kwargs):
            obj.selection_calls.append((args, kwargs))
            return np.array(selected, dtype=np.int32)
        obj.model, obj.select_grasps = forward, select
        return obj

    def test_empty_singleton_and_multiple_outputs_keep_candidate_axes(self):
        for indices in ([], [0], [1], [0, 1], [1, 0], [0, 0]):
            with self.subTest(indices=indices):
                obj = self.estimator(indices)
                grasps, scores, points, widths = obj.predict_grasps(self.points, convert_cam_coords=False)
                n = len(indices)
                self.assertEqual(grasps.shape, (n, 4, 4))
                self.assertEqual(scores.shape, (n,))
                self.assertEqual(points.shape, (n, 3))
                self.assertEqual(widths.shape, (n,))
                np.testing.assert_array_equal(scores, np.array([.9, .8])[indices])
                np.testing.assert_array_equal(points, self.points[indices])
                np.testing.assert_allclose(widths, np.array([.025, .08])[indices])
                self.assertEqual(len(obj.forward_calls), 1)
                args, kwargs = obj.selection_calls[0]
                self.assertEqual(args[2:], (2, 2, .4, .2))
                self.assertEqual(kwargs, {"with_replacement": False})

    def test_both_singleton_failures_reproduce_upstream_and_are_fixed(self):
        for index in (0, 1):
            with self.subTest(index=index):
                old, new = self.estimator([index], patched=False), self.estimator([index])
                old_out = old.predict_grasps(self.points, convert_cam_coords=False)
                new_out = new.predict_grasps(self.points, convert_cam_coords=False)
                old_filter = old.filter_segment(old_out[2], self.points, thres=.0001)
                self.assertEqual(np.asarray(old_filter).size, 0)
                new_filter = new.filter_segment(new_out[2], self.points, thres=.0001)
                np.testing.assert_array_equal(new_filter[0], [0])
                self.assertEqual(len(old_out[0]), 0 if index == 0 else 1)
                self.assertEqual(len(new_out[0]), 1)

    def test_scene_filter_preserves_singleton_and_still_rejects_outside_contacts(self):
        scene = self.points * [-1, -1, 1]  # public camera frame after native conversion
        for index in (0, 1):
            for offset, expected in ((0., 1), (1., 0)):
                with self.subTest(index=index, offset=offset):
                    obj = self.estimator([index])
                    with contextlib.redirect_stdout(io.StringIO()):
                        grasps, scores, contacts, widths = obj.predict_scene_grasps(
                            scene, pc_segments={"object": scene + offset},
                            local_regions=False, filter_grasps=True)
                    self.assertEqual(grasps["object"].shape, (expected, 4, 4))
                    self.assertEqual(scores["object"].shape, (expected,))
                    self.assertEqual(contacts["object"].shape, (expected, 3))
                    self.assertEqual(widths["object"].shape, (expected,))

    def test_multiple_candidates_are_numerically_identical_to_upstream(self):
        for convert in (False, True):
            for selected in ([0, 1], [1, 0], [1, 1]):
                with self.subTest(convert=convert, selected=selected):
                    old, new = self.estimator(selected, patched=False), self.estimator(selected)
                    before = old.predict_grasps(self.points, convert_cam_coords=convert, forward_passes=2)
                    after = new.predict_grasps(self.points, convert_cam_coords=convert, forward_passes=2)
                    for a, b in zip(before, after):
                        np.testing.assert_array_equal(a, b)
                    np.testing.assert_array_equal(old.forward_calls[0], new.forward_calls[0])
                    self.assertEqual(new.forward_calls[0].shape, (2, 2, 3))

    def test_dependency_file_and_original_class_are_not_modified(self):
        self.assertEqual(self.path.read_text(), self.source)
        self.assertIsNot(self.corrected, self.module.GraspEstimator)
        self.assertIs(self.corrected.predict_scene_grasps, self.module.GraspEstimator.predict_scene_grasps)
        self.assertIs(self.corrected.filter_segment, self.module.GraspEstimator.filter_segment)
        self.assertEqual(self.evidence["patch_module_sha256"],
                         hashlib.sha256(Path(fix.__file__).read_bytes()).hexdigest())

    def test_unknown_vendor_source_and_wrong_patch_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "pinned singleton-fix input"):
            fix.corrected_estimator_class(self.module)
        with patch.object(fix, "UPSTREAM_SOURCE_SHA256", hashlib.sha256(self.source.encode()).hexdigest()):
            with self.assertRaisesRegex(ValueError, "output differs"):
                fix.corrected_estimator_class(self.module)

    def test_missing_or_ambiguous_patch_targets_are_rejected(self):
        target = "if not np.any(selection_idcs):"
        for altered in (self.source.replace(target, "if False:"), self.source + "\n# " + target):
            with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
                fix._rewrite_source(altered)


if __name__ == "__main__":
    unittest.main()
