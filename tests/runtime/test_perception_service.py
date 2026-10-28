"""CPU tests for service evidence/readiness, never GPU or model accuracy tests."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import robodata.perception.perception_service as runner


class ServiceTests(unittest.TestCase):
    def test_manifest_requires_pinned_revision_size_and_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            weight = Path(folder) / "sam3.pt"
            weight.write_bytes(b"fake checkpoint for CPU file-validation test")
            digest = hashlib.sha256(weight.read_bytes()).hexdigest()
            manifest = Path(folder) / "manifest.json"
            data = {"repo": "facebook/sam3", "revision": runner.SAM3_WEIGHT_REVISION,
                    "files": [{"file": "sam3.pt", "path": str(weight),
                               "bytes": weight.stat().st_size, "sha256": digest}]}
            manifest.write_text(json.dumps(data))
            with patch.object(runner, "SAM3_WEIGHT_SHA256", digest):
                path, evidence = runner.checked_sam3_manifest(manifest)
                self.assertEqual(path, weight.resolve())
                self.assertEqual(evidence["checkpoint_sha256"], digest)
                weight.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    runner.checked_sam3_manifest(manifest)
            data["revision"] = "main"
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "pinned model revision"):
                runner.checked_sam3_manifest(manifest)

    def test_vendor_commit_and_import_location_are_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            vendor = Path(folder) / "sam3"; vendor.mkdir()
            with patch.object(runner.subprocess, "check_output", return_value="other\n"):
                with self.assertRaisesRegex(ValueError, "vendor commit"):
                    runner.checked_vendor(Path(folder), "sam3", runner.SAM3_VENDOR_COMMIT)
            with self.assertRaisesRegex(RuntimeError, "outside the pinned vendor"):
                runner.assert_module_under(NS(__file__="/tmp/elsewhere.py", __name__="sam3"), vendor)

    def test_sam3_passes_local_checkpoint_and_disables_downloads(self):
        vendor = Path("/fake/vendor/sam3")
        model, processor = Mock(), Mock()
        model.parameters.return_value = [NS(device="cuda:0", numel=lambda: 7)]
        builder = NS(__name__="sam3.model_builder", __file__=str(vendor / "sam3/model_builder.py"),
                     build_sam3_image_model=Mock(return_value=model))
        service = NS(_MODEL=None, _PROCESSOR=None, torch=Mock(), Sam3Processor=Mock(return_value=processor))
        imports = {"sam3.model_builder": builder, "cap_harness.providers.sam3.service": service}
        with patch.object(runner, "checked_vendor", return_value=vendor), \
             patch.object(runner, "checked_sam3_manifest", return_value=(Path("/weights/sam3.pt"), {})), \
             patch.object(runner.importlib, "import_module", side_effect=imports.__getitem__), \
             patch.object(runner.sys, "path", []):
            loaded, evidence = runner.load_sam3(Path("/fake/vendor"), Path("/manifest"))
        builder.build_sam3_image_model.assert_called_once_with(
            checkpoint_path="/weights/sam3.pt", load_from_HF=False, device="cuda",
            enable_inst_interactivity=True, eval_mode=True)
        self.assertIs(loaded._MODEL, model)
        self.assertIs(loaded._PROCESSOR, processor)
        self.assertEqual(evidence["vendor_commit"], runner.SAM3_VENDOR_COMMIT)

    def _graspnet_mocks(self, state_dict):
        vendor = Path("/fake/vendor/contact_graspnet_pytorch")
        model = Mock()
        model.parameters.return_value = [NS(device="cuda:0", numel=lambda: 7)]
        estimator = NS(model=model, device="cuda:0")
        estimator_module = NS(__file__=str(vendor / "contact_graspnet_pytorch/contact_grasp_estimator.py"),
                              __name__="contact_graspnet_pytorch.contact_grasp_estimator",
                              GraspEstimator=Mock(return_value=estimator))

        class CheckpointIO:
            def __init__(self, **kwargs):
                self.model = kwargs["model"]
            def load(self, filename):
                return self.parse_state_dict(state_dict)
            def parse_state_dict(self, state):
                self.model.load_state_dict(state["model"])
                return {}

        checkpoint_module = NS(__file__=str(vendor / "contact_graspnet_pytorch/checkpoints.py"),
                               __name__="contact_graspnet_pytorch.checkpoints", CheckpointIO=CheckpointIO)
        service = NS(_ESTIMATOR=None, _load_config=Mock(return_value={}))
        imports = {"contact_graspnet_pytorch.contact_grasp_estimator": estimator_module,
                   "contact_graspnet_pytorch.checkpoints": checkpoint_module,
                   "cap_harness.providers.graspnet.service": service}
        return vendor, model, service, imports

    def test_graspnet_requires_actual_model_state_load(self):
        for state, expected in [({}, False), ({"model": {"layer": "tensor"}}, True)]:
            vendor, model, service, imports = self._graspnet_mocks(state)
            with self.subTest(expected=expected), \
                 patch.object(runner, "checked_vendor", return_value=vendor), \
                 patch.object(runner, "checked_file", return_value={}), \
                 patch.object(runner, "corrected_estimator_class", return_value=(
                     imports["contact_graspnet_pytorch.contact_grasp_estimator"].GraspEstimator,
                     {"id": "test-cgn-patch"})) as corrected, \
                 patch.object(runner.importlib, "import_module", side_effect=imports.__getitem__), \
                 patch.object(runner.sys, "path", []):
                if expected:
                    _, evidence = runner.load_graspnet(Path("/fake/vendor"))
                    corrected.assert_called_once_with(imports["contact_graspnet_pytorch.contact_grasp_estimator"])
                    self.assertEqual(evidence["estimator_patch"], {"id": "test-cgn-patch"})
                    model.load_state_dict.assert_called_once_with(state["model"])
                    model.eval.assert_called_once()
                    self.assertIsNotNone(service._ESTIMATOR)
                else:
                    with self.assertRaisesRegex(ValueError, "no model state_dict"):
                        runner.load_graspnet(Path("/fake/vendor"))
                    self.assertIsNone(service._ESTIMATOR)

    def test_health_follows_loaded_model_and_replaces_liveness(self):
        class App:
            router = NS(routes=[NS(path="/healthz"), NS(path="/segment")])
            state = NS(request_limit=1)
            def get(self, path):
                def register(fn):
                    self.handler = fn
                    self.router.routes.append(NS(path=path))
                    return fn
                return register
        class Response:
            def __init__(self, content, status_code):
                self.content, self.status_code = content, status_code
        app = App()
        service = NS(app=app, _MODEL=None, _PROCESSOR=None)
        runner.install_loaded_health(service, "sam3", {"checkpoint_sha256": "test"})
        self.assertEqual([r.path for r in app.router.routes].count("/healthz"), 1)
        with patch.dict("sys.modules", {"fastapi.responses": NS(JSONResponse=Response)}):
            before = asyncio.run(app.handler())
            self.assertEqual(before.status_code, 503)
            self.assertFalse(before.content["loaded"])
            service._MODEL, service._PROCESSOR = object(), object()
            after = asyncio.run(app.handler())
            self.assertEqual(after.status_code, 200)
            self.assertTrue(after.content["loaded"])
            self.assertEqual(after.content["checkpoint_sha256"], "test")

    def test_cpu_fallback_or_mixed_device_parameters_are_rejected(self):
        for devices in [[], ["cpu"], ["cuda:0", "cpu"], ["cuda:1"]]:
            model = NS(parameters=lambda: [NS(device=d, numel=lambda: 2) for d in devices])
            with self.subTest(devices=devices), self.assertRaisesRegex(RuntimeError, "assigned GPU"):
                runner.checked_parameter_devices(model)

    def test_startup_seeds_python_numpy_and_torch_without_model_calls(self):
        numpy, torch = Mock(), Mock()
        with patch.dict("sys.modules", {"numpy": numpy, "torch": torch}), \
             patch.object(runner.random, "seed") as python_seed:
            runner.seed_service(0)
        python_seed.assert_called_once_with(0)
        numpy.random.seed.assert_called_once_with(0)
        torch.manual_seed.assert_called_once_with(0)
        torch.cuda.manual_seed_all.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
