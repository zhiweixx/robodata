"""Content identities for a portable, immutable execution source tree.

A digest identifies bytes, not trust. Callers must also verify that imported
modules resolve to the inventoried tree, and retain the record with results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

SCHEMA = "robodata.source.v1"
SOURCE_ROOTS = ("src", "third_party/aspire/src", "deploy", "patches", "configs")
ROOT_FILES = ("pyproject.toml", "third_party/aspire/pyproject.toml")


def excluded_source_path(relative: Path | str) -> bool:
    """Keep generated caches and private machine bindings out of snapshots."""
    relative = Path(relative)
    return (any(part in {"__pycache__", ".git"} or part.startswith(".venv")
                or part.endswith(".egg-info")
                for part in relative.parts)
            or relative.suffix == ".pyc"
            or (relative.parts and relative.parts[0] == "configs"
                and any(part.startswith("local") for part in relative.parts[1:])))


def sha256_file(path: Path | str) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def freeze_source(repo_root: Path | str) -> dict:
    root = Path(repo_root).resolve(strict=True)
    paths = []
    for folder in SOURCE_ROOTS:
        base = root / folder
        if base.is_symlink():
            raise ValueError(f"source must be a regular in-tree path: {folder}")
        if base.exists():
            for path in base.rglob("*"):
                if excluded_source_path(path.relative_to(root)):
                    continue
                if path.is_symlink():
                    raise ValueError(f"source must be a regular in-tree path: {path.relative_to(root)}")
                if path.is_file():
                    # Pin patches, lockfiles and packaged data too: extension
                    # allowlists silently miss execution-affecting inputs.
                    paths.append(path)
    paths.extend(root / name for name in ROOT_FILES if (root / name).is_file())
    files = {}
    for path in sorted(set(paths)):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"source must be a regular in-tree file: {path.relative_to(root)}")
        files[path.relative_to(root).as_posix()] = sha256_file(path)
    if not files or "src/robodata/__init__.py" not in files:
        raise ValueError("not a RoboData source checkout")
    return {"schema": SCHEMA, "files_sha256": files,
            "tree_sha256": canonical_sha256(files)}


def verify_source(repo_root: Path | str, record: Mapping) -> dict:
    if record.get("schema") != SCHEMA:
        raise ValueError("unsupported source inventory schema")
    files = record.get("files_sha256")
    if not isinstance(files, dict) or record.get("tree_sha256") != canonical_sha256(files):
        raise ValueError("source inventory digest mismatch")
    actual = freeze_source(repo_root)
    if actual != dict(record):
        changed = sorted(set(files) ^ set(actual["files_sha256"]) |
                         {key for key in files.keys() & actual["files_sha256"].keys()
                          if files[key] != actual["files_sha256"][key]})
        raise ValueError("execution source drift: " + ", ".join(changed[:12]))
    return actual


def verify_imports(repo_root: Path | str, record: Mapping,
                   executed_files: Mapping[str, str | Path]) -> dict:
    """Reject a same-named module loaded from another checkout or mutable path."""
    root = Path(repo_root).resolve(strict=True)
    verify_source(root, record)
    evidence = {}
    for relative, loaded in executed_files.items():
        expected = root / relative
        actual = Path(loaded).resolve(strict=True)
        if relative not in record["files_sha256"] or actual != expected.resolve(strict=True):
            raise ValueError(f"module origin differs from frozen source: {relative}")
        if sha256_file(actual) != record["files_sha256"][relative]:
            raise ValueError(f"loaded source differs from inventory: {relative}")
        evidence[relative] = record["files_sha256"][relative]
    return evidence
