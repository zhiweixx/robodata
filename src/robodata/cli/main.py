"""Lightweight public CLI. No model or simulator is loaded by planning/status."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from robodata.data.provenance import (freeze_source, verify_source, canonical_sha256,
                                     excluded_source_path)

SNAPSHOT_DIRECTORIES = ("src", "third_party/aspire", "deploy", "patches", "configs")


def repository_root(explicit=None):
    root = Path(explicit).expanduser().resolve() if explicit else Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").is_file() or not (root / "deploy").is_dir():
        raise ValueError("use an editable source checkout, or pass --repo /path/to/robodata")
    return root


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_cluster_config(repo, path):
    """Use the selected checkout's config validator, including path expansion."""
    spec = importlib.util.spec_from_file_location("_robodata_cluster_config", repo / "deploy/common.py")
    if spec is None or spec.loader is None:
        raise ValueError("selected checkout has no deployment config loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_config(path)


def publish_campaign(temporary, output):
    """Reserve a new destination atomically; publish the ready plan last.

    A concurrent planner or an existing empty directory must never be replaced.
    An interrupted publication has no plan.json and cannot start evaluation.
    """
    output.mkdir(mode=0o700)
    for item in temporary.iterdir():
        if item.name != "plan.json":
            item.rename(output / item.name)
    (temporary / "plan.json").rename(output / "plan.json")
    temporary.rmdir()


def plan_campaign(repo, manifest_path, config_path, output, shards=None):
    from robodata.eval.robodojo.manifest import load_manifest, plan_shards
    repo = Path(repo).resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output == repo or any(output.is_relative_to(repo / name) for name in SNAPSHOT_DIRECTORIES):
        raise ValueError("campaign output must not be inside source snapshot directories")
    if output.exists():
        raise FileExistsError("campaign already exists; use its frozen plan and resume")
    manifest = load_manifest(manifest_path)
    config = load_cluster_config(repo, config_path)
    assignments = plan_shards(manifest, shards)
    original_inventory = freeze_source(repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".robodata-plan-", dir=output.parent))
    try:
        frozen = temporary / "source"
        frozen.mkdir()
        def ignore(directory, names):
            return [name for name in names
                    if excluded_source_path((Path(directory) / name).relative_to(repo))]
        for name in SNAPSHOT_DIRECTORIES:
            source = repo / name
            if source.exists():
                shutil.copytree(source, frozen / name, ignore=ignore)
        for name in ("pyproject.toml", "README.md", "AGENTS.md", "LICENSE", "NOTICE"):
            if (repo / name).is_file():
                shutil.copy2(repo / name, frozen / name)
        inventory = freeze_source(frozen)
        verify_source(frozen, inventory)
        verify_source(repo, original_inventory)
        if inventory != original_inventory:
            raise ValueError("source changed while campaign snapshot was copied")
        write_json(temporary / "manifest.json", manifest)
        write_json(temporary / "cluster_config.json", config)
        write_json(temporary / "source_inventory.json", inventory)
        group_ids = [g["group_id"] for g in manifest["groups"]]
        plan = {"schema": 1, "groups": group_ids, "assignments": assignments,
                "manifest_sha256": canonical_sha256(manifest),
                "cluster_config_sha256": canonical_sha256(config),
                "source_tree_sha256": inventory["tree_sha256"]}
        write_json(temporary / "plan.json", plan)
        publish_campaign(temporary, output)
        return {"campaign_dir": str(output), "groups": len(group_ids),
                "shards": len(assignments), "source_tree_sha256": inventory["tree_sha256"]}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="robodata", description=__doc__)
    parser.add_argument("--repo", help="source checkout (defaults to editable installation)")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("doctor", "fetch", "setup", "build-container", "submit"):
        item = sub.add_parser(command, add_help=False,
            help="deployment tools; pass --help for options")
        item.add_argument("arguments", nargs=argparse.REMAINDER)
    item = sub.add_parser("make-manifest", help="select distinct official states per task family")
    item.add_argument("--states-per-task", type=int, required=True,
                      help="requested distinct states per family; even number from 2 to 50")
    item.add_argument("--exclude-family", action="append", default=[],
                      help="optional family to exclude (repeat for multiple families)")
    item.add_argument("--output", required=True, help="new JSON manifest path; never overwrites")
    item = sub.add_parser("plan", help="freeze source/config and assign disjoint native groups")
    item.add_argument("--manifest", required=True)
    item.add_argument("--config", required=True)
    item.add_argument("--output", required=True, help="new campaign directory")
    item.add_argument("--shards", type=int, help="whole-group shards; defaults to one per selected configuration")
    item = sub.add_parser("run", help="run one planned native group in a Slurm allocation")
    item.add_argument("--campaign-dir", required=True)
    item.add_argument("--group-id", required=True)
    item.add_argument("--resume", action="store_true")
    for command in ("status", "audit"):
        item = sub.add_parser(command, help="inspect native results without model execution")
        item.add_argument("--campaign-dir", required=True)
        item.add_argument("--output", help="optional JSON report destination")
    args, unknown = parser.parse_known_args(argv)
    try:
        if args.command in {"doctor", "fetch", "setup", "build-container", "submit"}:
            root = repository_root(args.repo)
            # Preserve the complete command tail; argparse.REMAINDER alone does
            # not retain option ordering when the first token is an option.
            tail = argv[argv.index(args.command) + 1:]
            return subprocess.call([sys.executable, str(root / "deploy/manage.py"),
                                    args.command, *tail])
        if unknown:
            parser.error("unrecognized arguments: " + " ".join(unknown))
        if args.command == "make-manifest":
            from robodata.eval.robodojo.state_subset import build_state_subset
            manifest = build_state_subset(args.states_per_task, args.exclude_family)
            path = Path(args.output).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
            with path.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
            value = {"manifest": str(path), "task_families": manifest["family_count"],
                     "configuration_groups": manifest["group_count"],
                     "requested_states_per_task": manifest["per_family_target"],
                     "target_episodes": manifest["total_target_episodes"],
                     "candidate_states": manifest["total_candidate_episodes"],
                     "finite_layout_exceptions": manifest["exception_family_targets"],
                     "excluded_families": manifest["excluded_families"]}
        elif args.command == "plan":
            value = plan_campaign(repository_root(args.repo), args.manifest,
                                  args.config, args.output, args.shards)
        elif args.command == "run":
            from robodata.eval.robodojo.runner import run_campaign_group
            directory = Path(args.campaign_dir).expanduser().resolve(strict=True)
            return run_campaign_group(directory, args.group_id, resume=args.resume)
        else:
            from robodata.eval.robodojo.audit import audit_campaign
            value = audit_campaign(Path(args.campaign_dir), full=args.command == "audit")
            if args.output:
                write_json(args.output, value)
        print(json.dumps(value, indent=2, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"robodata: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
