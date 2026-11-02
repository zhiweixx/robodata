"""Bind the authorized prompt/schema revision to its immutable source baseline.

CPU-only: this module reads source files and hashes, never credentials or runtime
state. The revised harness is deliberately not described as byte-identical to
the historical Qwen/Terra harness used to select paired states.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

REVISION = "robodojo_task_reasoning_v2"
TASK_CONTRACT = "robodojo_public_task_contract_v2"
PROPOSAL_SCHEMA = "aspire_program_give_up_v2"
REPLAY_POLICY = "complete_response_output_v1"
RUNTIME_CHANGES = frozenset({
    "README.md", "qwen_integration/qwen_verify_local_driver.py", "qwen_integration/verify_model_episode_offline.py",
    "qwen_integration/model_driver.py", "qwen_integration/local_model_driver.py",
    "qwen_integration/model_episode.py", "qwen_integration/deployment_docs.py",
    "qwen_integration/qwen_model_smoke.py", "qwen_integration/run_suite.py",
    "qwen_integration/run_qwen_suite.py", "qwen_full/full_adapter.py",
    "qwen_full/full_episode.py", "qwen_full/full_deployment_docs.py",
    "qwen_full/full_model_driver.py", "qwen_full/full_model_smoke.py", "qwen_full/mtp_replay_benchmark.py",
})
REQUIRED_CHANGES = frozenset({"qwen_integration/model_driver.py", "qwen_full/full_adapter.py",
    "qwen_full/full_episode.py", "qwen_full/full_deployment_docs.py", "qwen_full/full_model_driver.py"})
RECIPE_ROOT = "qwen_full/official_task_recipes/"
REQUIRED_ADDITIONS = frozenset({"qwen_full/task_prompt_contract.py", "qwen_full/test_task_prompt_contract.py",
    RECIPE_ROOT + "LICENSE", RECIPE_ROOT + "PROVENANCE.json"})


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or str(p) != name or "\\" in name:
        raise ValueError("invalid source path")
    return p


def _allowed_change(name):
    p = _relative(name)
    return name in RUNTIME_CHANGES or (p.parts[0] in {"qwen_full", "qwen_integration"}
        and p.name.startswith("test_") and p.suffix == ".py")


def validate_revision(manifest, baseline_profile=None):
    revision = manifest.get("harness_revision", {})
    expected = {"version": REVISION, "task_contract": TASK_CONTRACT,
        "proposal_schema": PROPOSAL_SCHEMA, "reasoning_replay": REPLAY_POLICY,
        "reasoning_context": "provider_default"}
    if any(revision.get(k) != v for k, v in expected.items()):
        raise ValueError("fresh task/reasoning protocol revision required; historical resume is forbidden")
    baseline = revision.get("baseline_source_profile", {})
    if baseline_profile is not None and baseline != baseline_profile:
        raise ValueError("revision baseline differs from paired source evidence")
    old = baseline.get("files_sha256", {})
    profile = manifest.get("source_profile", {})
    files = profile.get("files_sha256", {})
    if (len(old) != 536 or canonical_sha256(old) != baseline.get("files_manifest_sha256")
            or profile.get("git_commit") is not None
            or profile.get("parent_git_commit") != baseline.get("git_commit")
            or profile.get("parent_files_manifest_sha256") != baseline.get("files_manifest_sha256")
            or profile.get("protocol_revision") != REVISION
            or canonical_sha256(files) != profile.get("files_manifest_sha256")):
        raise ValueError("revised source provenance is incomplete or relabels historical Git source")
    if not set(old) <= set(files):
        raise ValueError("revision deleted historical harness files")
    changed = {name for name in old if files[name] != old[name]}
    added = set(files) - set(old)
    recipes = {n for n in added if n.startswith(RECIPE_ROOT) and n.endswith(".md")
               and len(PurePosixPath(n).parts) == 3}
    if (not REQUIRED_CHANGES <= changed or any(not _allowed_change(n) for n in changed)
            or len(recipes) != 42 or added != REQUIRED_ADDITIONS | recipes):
        raise ValueError("harness changes exceed the authorized task/schema revision or omit required recipes")
    for name in files:
        _relative(name)
    return revision


def build_revised_profile(source_root, baseline_profile, campaign):
    source = Path(source_root).resolve(strict=True)
    baseline = copy.deepcopy(baseline_profile)
    old = baseline["files_sha256"]
    additions = set(REQUIRED_ADDITIONS)
    additions.update(p.relative_to(source).as_posix()
        for p in (source / RECIPE_ROOT).glob("*.md"))
    files = {}
    for name in sorted(set(old) | additions):
        _relative(name)
        p = source / name
        if p.is_symlink() or not p.is_file() or not p.resolve(strict=True).is_relative_to(source):
            raise ValueError("missing or unsafe revised source: " + name)
        files[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    profile = {"relative_root": "harness_profiles/" + campaign, "git_commit": None,
        "parent_git_commit": baseline["git_commit"],
        "parent_files_manifest_sha256": baseline["files_manifest_sha256"],
        "protocol_revision": REVISION, "files_sha256": files,
        "files_manifest_sha256": canonical_sha256(files)}
    revision = {"version": REVISION, "task_contract": TASK_CONTRACT,
        "proposal_schema": PROPOSAL_SCHEMA, "reasoning_replay": REPLAY_POLICY,
        "reasoning_context": "provider_default", "baseline_source_profile": baseline}
    validate_revision({"source_profile": profile, "harness_revision": revision}, baseline)
    return profile, revision


def copy_revised_profile(source_root, destination, profile):
    source = Path(source_root).resolve(strict=True)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, digest in profile["files_sha256"].items():
        _relative(name)
        original = source / name
        if original.is_symlink() or not original.resolve(strict=True).is_relative_to(source):
            raise ValueError("unsafe revised source")
        data = original.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("revised source changed while freezing: " + name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        target.chmod(original.stat().st_mode & 0o777)
