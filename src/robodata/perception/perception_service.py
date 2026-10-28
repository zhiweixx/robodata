"""Run pinned ASPIRE perception services with verified, local-only weights.

The service HTTP handlers and learned inference are upstream implementations.
This entry point makes checkpoint selection and model readiness explicit; it
does not train, search for skills, inspect task state, or invoke paid models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import random
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from robodata.perception.cgn_singleton_fix import corrected_estimator_class


SAM3_VENDOR_COMMIT = "6fe87d64a5beb9084923d7a9e002741178635b09"
CGN_VENDOR_COMMIT = "da3dcfb2f53e43b186083ee4a9d1e232f73efc98"
SAM3_WEIGHT_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
SAM3_WEIGHT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
CGN_WEIGHT_SHA256 = "39fc3439d5814043ba64e0715c127c2e8aca6ea376c22614e11af1bc89317762"
CGN_CONFIG_SHA256 = "7ae6cc15c726e8ec2c46b0c799d58ca95caeb9ce9f261aaaf3e467712df76ac5"
LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path: Path, expected: str, size: int | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file() or (size is not None and path.stat().st_size != size):
        raise ValueError(f"Checkpoint file/size mismatch: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def checked_vendor(root: Path, name: str, expected_commit: str) -> Path:
    vendor = (root / name).expanduser().resolve(strict=True)
    actual = subprocess.check_output(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected_commit:
        raise ValueError(f"Unexpected {name} vendor commit: {actual}")
    return vendor


def checked_sam3_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest = json.loads(path.read_text())
    if manifest.get("repo") != "facebook/sam3" or manifest.get("revision") != SAM3_WEIGHT_REVISION:
        raise ValueError("SAM3 manifest does not identify the pinned model revision")
    entries = manifest.get("files", [])
    if not isinstance(entries, list):
        raise ValueError("SAM3 manifest files must be a list")
    files = {}
    for row in entries:
        name = row["file"]
        if name in files:
            raise ValueError("Duplicate SAM3 manifest filename")
        files[name] = checked_file(Path(row["path"]), row["sha256"], int(row["bytes"]))
    if "sam3.pt" not in files or files["sam3.pt"]["sha256"] != SAM3_WEIGHT_SHA256:
        raise ValueError("SAM3 weights are not the pinned checkpoint")
    return Path(files["sam3.pt"]["path"]), {
        "repo": manifest["repo"], "revision": manifest["revision"], "files": files,
        "manifest_path": str(path.resolve()), "manifest_sha256": sha256_file(path),
        "checkpoint_sha256": SAM3_WEIGHT_SHA256,
    }


def assert_module_under(module, vendor: Path):
    if not Path(module.__file__).resolve().is_relative_to(vendor):
        raise RuntimeError(f"Imported {module.__name__} outside the pinned vendor: {module.__file__}")


def checked_parameter_devices(model) -> dict[str, Any]:
    parameters = list(model.parameters())
    devices = sorted({str(parameter.device) for parameter in parameters})
    if not parameters or devices != ["cuda:0"]:
        raise RuntimeError(f"Perception model parameters must be on assigned GPU cuda:0, got {devices}")
    return {"parameter_devices": devices, "parameter_count": sum(p.numel() for p in parameters)}


def seed_service(seed: int):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sam3(vendor_root: Path, manifest_path: Path):
    vendor = checked_vendor(vendor_root, "sam3", SAM3_VENDOR_COMMIT)
    checkpoint, evidence = checked_sam3_manifest(manifest_path)
    sys.path.insert(0, str(vendor))
    builder = importlib.import_module("sam3.model_builder")
    assert_module_under(builder, vendor)
    service = importlib.import_module("cap_harness.providers.sam3.service")
    # The process sees exactly its allocated provider GPU via CUDA_VISIBLE_DEVICES.
    # Do not rely on model-builder defaults, HF credentials, or network downloads.
    service._DEVICE = "cuda"
    service.torch.cuda.set_device(0)
    service.torch.backends.cuda.matmul.allow_tf32 = True
    service.torch.backends.cudnn.allow_tf32 = True
    model = builder.build_sam3_image_model(
        checkpoint_path=str(checkpoint), load_from_HF=False, device="cuda",
        enable_inst_interactivity=True, eval_mode=True)
    devices = checked_parameter_devices(model)
    service._MODEL = model
    service._PROCESSOR = service.Sam3Processor(model, device="cuda", confidence_threshold=0.0)
    return service, {**evidence, **devices, "vendor_commit": SAM3_VENDOR_COMMIT, "vendor_path": str(vendor)}


def load_graspnet(vendor_root: Path):
    vendor = checked_vendor(vendor_root, "contact_graspnet_pytorch", CGN_VENDOR_COMMIT)
    checkpoint_root = vendor / "checkpoints/contact_graspnet"
    checkpoint = checked_file(checkpoint_root / "checkpoints/model.pt", CGN_WEIGHT_SHA256)
    config = checked_file(checkpoint_root / "config.yaml", CGN_CONFIG_SHA256)
    sys.path[:0] = [str(vendor), str(vendor / "Pointnet_Pointnet2_pytorch")]
    estimator_module = importlib.import_module("contact_graspnet_pytorch.contact_grasp_estimator")
    checkpoint_module = importlib.import_module("contact_graspnet_pytorch.checkpoints")
    assert_module_under(estimator_module, vendor)
    assert_module_under(checkpoint_module, vendor)
    service = importlib.import_module("cap_harness.providers.graspnet.service")

    class RequiredModelCheckpoint(checkpoint_module.CheckpointIO):
        def parse_state_dict(self, state_dict):
            # Upstream otherwise only prints a warning when model weights are
            # absent, which is insufficient evidence of a loaded model.
            if not isinstance(state_dict, dict) or not isinstance(state_dict.get("model"), dict):
                raise ValueError("Contact-GraspNet checkpoint has no model state_dict")
            result = super().parse_state_dict(state_dict)
            self.model_loaded = True
            return result

    # Keep the installed vendor tree intact for historical reproducibility.
    # Only candidate-count/array-rank postprocessing differs in this subclass.
    estimator_class, estimator_patch = corrected_estimator_class(estimator_module)
    estimator = estimator_class(service._load_config(checkpoint_root))
    if str(estimator.device) != "cuda:0":
        raise RuntimeError(f"Contact-GraspNet must load on the assigned GPU, got {estimator.device}")
    loader = RequiredModelCheckpoint(checkpoint_dir=checkpoint_root / "checkpoints",
                                     model=estimator.model)
    loader.load("model.pt")  # Missing files or incompatible state dicts are fatal.
    if not getattr(loader, "model_loaded", False):
        raise RuntimeError("Contact-GraspNet checkpoint loader did not load model weights")
    estimator.model.eval()
    devices = checked_parameter_devices(estimator.model)
    service._ESTIMATOR = estimator
    return service, {**devices, "vendor_commit": CGN_VENDOR_COMMIT, "vendor_path": str(vendor),
                     "estimator_patch": estimator_patch,
                     "checkpoint": checkpoint, "config": config,
                     "checkpoint_sha256": CGN_WEIGHT_SHA256}


def loaded_status(service, kind: str) -> bool:
    if kind == "sam3":
        return service._MODEL is not None and service._PROCESSOR is not None
    return service._ESTIMATOR is not None


def install_loaded_health(service, kind: str, evidence: dict[str, Any]):
    # Replace the upstream liveness-only route; never report readiness merely
    # because FastAPI initialized or a TCP port was bound.
    app = service.app
    app.router.routes[:] = [route for route in app.router.routes
                            if getattr(route, "path", None) != "/healthz"]
    metadata = {**evidence, "kind": kind, "device": "cuda:0", "offline_weights": True}

    @app.get("/healthz")
    async def healthz():
        from fastapi.responses import JSONResponse
        loaded = loaded_status(service, kind)
        return JSONResponse({**metadata, "ok": loaded, "loaded": loaded,
                             "model_loaded": loaded, "request_limit": app.state.request_limit},
                            status_code=200 if loaded else 503)

    app.state.checkpoint_evidence = metadata
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("sam3", "graspnet"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--sam3-manifest", type=Path)
    parser.add_argument("--vendor-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        parser.error("perception services must bind to the compute node's loopback interface")
    if args.kind == "sam3" and args.sam3_manifest is None:
        parser.error("--sam3-manifest is required for SAM3")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CAP_HARNESS_SERVICE_CONCURRENCY"] = "1"
    seed_service(0)
    started = time.monotonic()
    service, evidence = (load_sam3(args.vendor_root, args.sam3_manifest) if args.kind == "sam3"
                         else load_graspnet(args.vendor_root))
    evidence["load_wall_seconds"] = time.monotonic() - started
    evidence["random_seed"] = 0
    evidence["random_seed_scope"] = "Python/NumPy/Torch service startup; request sequence advances RNG state"
    app = install_loaded_health(service, args.kind, evidence)
    LOGGER.info("Perception model loaded: %s", json.dumps(app.state.checkpoint_evidence, sort_keys=True))
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
