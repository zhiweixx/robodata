"""Narrow, hash-guarded correction of pinned CGN candidate postprocessing.

No vendor file is edited. The service instantiates a subclass with only the
corrected predict_grasps method; its network, selection, filtering, thresholds,
coordinate transforms and checkpoint loader are inherited unchanged.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path


FIX_ID = "cgn-singleton-shapes-v1"
UPSTREAM_SOURCE_SHA256 = "d063d4379f838fdabf82aa970161b048f32519203ef5b980f47600f7b9345356"
PATCHED_SOURCE_SHA256 = "efef788871342af28ca7b77afbfb0b930b2111596818f5e8b96800087f4ed68d"
_REPLACEMENTS = (
    ("if not np.any(selection_idcs):", "if len(selection_idcs) == 0:"),
    ("pred_points[selection_idcs].squeeze()", "pred_points[selection_idcs]"),
    ("gripper_openings[selection_idcs].squeeze()", "gripper_openings[selection_idcs]"),
)


def expected_patch_evidence(patch_module_sha256):
    return {"id": FIX_ID, "method": "GraspEstimator.predict_grasps",
            "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
            "patched_source_sha256": PATCHED_SOURCE_SHA256,
            "patch_module_sha256": patch_module_sha256}


def _rewrite_source(source):
    """Three exact substitutions; refuse missing or ambiguous patch targets."""
    for original, replacement in _REPLACEMENTS:
        if source.count(original) != 1:
            raise ValueError("CGN singleton fix target is missing or ambiguous")
        source = source.replace(original, replacement)
    return source


def _replacement_method(source, module_globals, filename):
    """Compile only the method; do not re-execute vendor imports/module setup."""
    tree = ast.parse(source, filename=filename)
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "GraspEstimator"]
    methods = [node for cls in classes for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name == "predict_grasps"]
    if len(classes) != 1 or len(methods) != 1:
        raise ValueError("CGN singleton fix requires exactly one predict_grasps method")
    namespace = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), filename, "exec"),
         module_globals, namespace)
    return namespace["predict_grasps"]


def corrected_estimator_class(module):
    """Return a corrected subclass and exact patch evidence before model loading."""
    path = Path(module.__file__).resolve(strict=True)
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != UPSTREAM_SOURCE_SHA256:
        raise ValueError("CGN estimator source differs from pinned singleton-fix input")
    patched = _rewrite_source(original.decode("utf-8"))
    if hashlib.sha256(patched.encode("utf-8")).hexdigest() != PATCHED_SOURCE_SHA256:
        raise ValueError("CGN singleton-fix output differs from pinned bytes")
    method = _replacement_method(patched, vars(module), str(path) + "::" + FIX_ID)
    corrected = type("SingletonSafeGraspEstimator", (module.GraspEstimator,),
                     {"predict_grasps": method, "__module__": __name__})
    evidence = expected_patch_evidence(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return corrected, evidence
