#!/usr/bin/env python3
"""Build the native RoboDojo sweep without importing Isaac Sim or task code.

The Lighthouse host asset root is currently:
  /srv/ROBODATA_RUNTIME/RoboDojo/
  .cache/robodojo_assets_repo/Assets
The isaac6 checkout's Assets symlink points to /assets, a container mount, so
host invocations must pass the actual host directory with --assets-root.

Layout IDs deliberately follow SeedManager: enumerate filenames sorted by
their numeric suffix. A filename suffix is not necessarily its layout ID.
All candidates are authorized; target_episodes is the number scored, allowing
native invalid-layout replacement. Nothing here sets EVAL_LAYOUT_IDS.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


class ManifestError(ValueError):
    pass


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            try:
                values.append(ast.literal_eval(node.value))
            except (ValueError, TypeError) as exc:
                raise ManifestError(f"{path}: {name} must be a literal") from exc
    if len(values) != 1:
        raise ManifestError(f"{path}: expected one literal assignment for {name}")
    return values[0]


def scalar(value: str):
    """Read the scalar-only official _task.yml subset; fail on schema drift."""
    value = value.strip()
    if value.startswith(('"', "'")):
        try:
            result = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ManifestError(f"unsupported YAML scalar: {value!r}") from exc
        if not isinstance(result, str):
            raise ManifestError("quoted YAML scalar must be a string")
        return result
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", value):
        return value
    raise ManifestError(f"unsupported YAML scalar: {value!r}")


def read_native_options(path: Path):
    """Avoid requiring PyYAML on the Lighthouse login host."""
    result = {}
    stack = [(-2, result)]
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"( *)([A-Za-z_][A-Za-z0-9_-]*):\s*(.*?)\s*", raw)
        if not match or "\t" in raw:
            raise ManifestError(f"{path}:{lineno}: unsupported native configuration syntax")
        indent, key, value = len(match[1]), match[2], match[3]
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack or indent != stack[-1][0] + 2:
            raise ManifestError(f"{path}:{lineno}: unexpected YAML indentation")
        parent = stack[-1][1]
        if key in parent:
            raise ManifestError(f"{path}:{lineno}: duplicate configuration key {key}")
        parent[key] = scalar(value) if value else {}
        if not value:
            stack.append((indent, parent[key]))
    if set(result) != {"common", "tasks"} or not all(isinstance(v, dict) for v in result.values()):
        raise ManifestError(f"{path}: expected common/tasks mappings")
    if any(isinstance(v, dict) for v in result["common"].values()) or any(
        not isinstance(v, dict) or any(isinstance(item, dict) for item in v.values())
        for v in result["tasks"].values()
    ):
        raise ManifestError(f"{path}: native task options must be scalar mappings")
    return result


def native_step_limit(path: Path, task: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    all_classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    if task not in all_classes:
        raise ManifestError(f"{path}: missing or duplicate task class {task}")
    # Official tasks commonly inherit their budget from a same-file *Common
    # mixin; importing the task to follow this would load Isaac dependencies.
    reachable = set()
    def visit(name):
        if name in reachable:
            return
        reachable.add(name)
        for base in all_classes[name].bases:
            if isinstance(base, ast.Name) and base.id in all_classes:
                visit(base.id)
    visit(task)
    limits = []
    for node in (node for name in reachable for node in ast.walk(all_classes[name])):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            if (isinstance(target, ast.Attribute) and target.attr == "step_lim"
                    and isinstance(target.value, ast.Name) and target.value.id == "self"):
                try:
                    limits.append(ast.literal_eval(node.value))
                except (ValueError, TypeError) as exc:
                    raise ManifestError(f"{path}: dynamic step_lim requires review") from exc
    if len(limits) != 1 or type(limits[0]) is not int or limits[0] <= 0:
        raise ManifestError(f"{path}: expected one positive literal self.step_lim")
    return limits[0]


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkout_head(root: Path):
    # Do not accidentally report an enclosing, unrelated workspace Git repo.
    if not (root / ".git").exists():
        return None
    completed = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def build_manifest(robodojo_root, assets_root, *, priority=()):
    root = Path(robodojo_root).expanduser().resolve(strict=True)
    assets = Path(assets_root).expanduser().resolve(strict=True)
    inventory_path = root / "scripts/internal/task_inventory.py"
    summary_path = root / "scripts/internal/summarize_result.py"
    options_path = root / "task/RoboDojo/config/_task.yml"
    task_dir = root / "task/RoboDojo/tasks"
    config_dir = root / "task/RoboDojo/config"
    dimensions = literal_assignment(inventory_path, "DIMENSION_TASKS")
    summary_dimensions = literal_assignment(summary_path, "DIMENSIONS")
    seeds = literal_assignment(summary_path, "EXPECTED_SEEDS")
    standalone = literal_assignment(summary_path, "STANDALONE_EPISODES")
    paired = literal_assignment(summary_path, "PAIRED_HALF_EPISODES")
    expected_dimensions = {"generalization": 12, "precision": 8, "long-horizon": 8, "memory": 6, "open": 8}
    if not isinstance(dimensions, dict) or set(dimensions) != set(expected_dimensions):
        raise ManifestError("official capability dimensions changed; review full-evaluation scope")
    if any(not isinstance(names, (list, tuple)) or len(names) != expected_dimensions[dim]
           or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in names)
           for dim, names in dimensions.items()):
        raise ManifestError("official task-family counts or task names changed")
    if ({key.lower(): set(value) for key, value in summary_dimensions.items()}
            != {key: set(value) for key, value in dimensions.items()}):
        raise ManifestError("inventory and summary disagree on capability dimensions")
    if seeds != [0, 1, 2] or standalone != 50 or paired != 25:
        raise ManifestError("official seed/count protocol changed; review full-evaluation scope")
    family_dimensions = {task: dim for dim, names in dimensions.items() for task in names}
    if len(family_dimensions) != 42:
        raise ManifestError("duplicate task family in official dimensions")
    config_names = sorted(set(family_dimensions) | {task + "_random" for task in dimensions["generalization"]})
    if len(config_names) != 54:
        raise ManifestError("expected 54 runnable configurations")
    options = read_native_options(options_path)
    if set(options["tasks"]) - set(config_names):
        raise ManifestError("native overrides include unknown task configurations")
    actual_tasks = {p.stem for p in task_dir.glob("*.py") if p.stem != "__init__"}
    if actual_tasks != set(config_names):
        raise ManifestError(f"task modules disagree with expected configurations: {sorted(actual_tasks ^ set(config_names))}")
    layout_root = assets / "Eval_Layout/RoboDojo/arx_x5"
    sources = [inventory_path, summary_path, options_path]
    records = {}
    for task in config_names:
        family = task.removesuffix("_random")
        dimension = family_dimensions[family]
        native = dict(options["common"], **options["tasks"].get(task, {}))
        target = paired if dimension == "generalization" else standalone
        if type(native.get("eval_nums")) is not int or native["eval_nums"] != target:
            raise ManifestError(f"{task}: native eval_nums disagrees with official summary ({target})")
        task_source = task_dir / (task + ".py")
        task_config = config_dir / (task + ".yml")
        if not task_config.is_file():
            raise ManifestError(f"{task}: missing native task configuration")
        step_limit = native_step_limit(task_source, task)
        sources.extend([task_source, task_config])
        records[task] = {"family": family, "dimension": dimension,
                         "variant": "random" if task.endswith("_random") else "standard",
                         "step_limit": step_limit, "target_episodes": target,
                         "native_options": native}
    groups = []
    episode_ids = []
    episode_seeds = {}
    for seed in seeds:
        seed_dir = layout_root / str(seed)
        if not seed_dir.is_dir():
            raise ManifestError(f"missing layout seed directory: {seed_dir}")
        for task in config_names:
            pattern = re.compile(re.escape(task) + r"_([0-9]+)\.json")
            files = [(int(match[1]), path) for path in seed_dir.iterdir()
                     if path.is_file() and (match := pattern.fullmatch(path.name))]
            files.sort(key=lambda item: item[0])
            if len({suffix for suffix, _ in files}) != len(files):
                raise ManifestError(f"{task} seed {seed}: ambiguous duplicate numeric layout suffix")
            record = records[task]
            if len(files) < record["target_episodes"]:
                raise ManifestError(f"{task} seed {seed}: {len(files)} candidates < {record['target_episodes']} native target")
            ids = [f"{task}_s{seed}_l{idx}" for idx in range(len(files))]
            episode_ids.extend(ids)
            episode_seeds.update({episode_id: seed for episode_id in ids})
            groups.append({"group_id": f"{task}_s{seed}", "task": task, "eval_seed": seed,
                           **record, "candidate_count": len(files),
                           "candidate_layout_ids": list(range(len(files))), "episode_ids": ids,
                           "layouts": [{"layout_id": idx, "filename": path.name,
                                        "filename_suffix": suffix, "source_path": str(path)}
                                       for idx, (suffix, path) in enumerate(files)]})
    priority = list(priority)
    if len(set(priority)) != len(priority):
        raise ManifestError("priority entries must be unique")
    choices = set(config_names) | {group["group_id"] for group in groups}
    if set(priority) - choices:
        raise ManifestError(f"unknown priority entries: {sorted(set(priority) - choices)}")
    for group in groups:
        group["priority"] = min((idx for idx, item in enumerate(priority)
                                 if item in {group["task"], group["group_id"]}), default=len(priority))
    groups.sort(key=lambda group: group["priority"])  # Stable: seed-major, then task name.
    total_target = sum(group["target_episodes"] for group in groups)
    if len(groups) != 162 or total_target != 6300 or len(episode_ids) != len(set(episode_ids)):
        raise ManifestError("full-evaluation group/count/identity invariant failed")
    return {"schema_version": 1, "benchmark": "RoboDojo", "embodiment": "arx_x5",
            "group_count": len(groups), "task_count": len(config_names), "family_count": 42,
            "eval_seeds": seeds, "total_target_episodes": total_target,
            "total_candidate_episodes": len(episode_ids),
            "action_budget_unit": "native_control_actions", "episode_id_format": "{task}_s{eval_seed}_l{layout_id}",
            "layout_id_rule": "zero-based position in task filenames sorted by numeric suffix, matching SeedManager",
            "source": {"robodojo_root": str(root), "assets_root": str(assets), "layout_root": str(layout_root),
                       "git_head": checkout_head(root),
                       "sha256": {str(path.relative_to(root)): sha256(path) for path in sources}},
            "episode_ids": episode_ids, "episode_seeds": episode_seeds, "groups": groups}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-root", required=True)
    parser.add_argument("--assets-root", required=True, help="Actual host Assets directory, not container-only /assets")
    parser.add_argument("--output", required=True)
    parser.add_argument("--priority", default="", help="Comma-separated task names or group IDs, in scheduling order")
    args = parser.parse_args(argv)
    try:
        priority = args.priority.split(",") if args.priority else []
        manifest = build_manifest(args.robodojo_root, args.assets_root, priority=priority)
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=output.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        try:
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    except (ManifestError, OSError, SyntaxError) as exc:
        parser.exit(2, f"manifest error: {exc}\n")
    print(json.dumps({"manifest": str(output), "groups": manifest["group_count"],
                      "target_episodes": manifest["total_target_episodes"],
                      "candidate_episodes": manifest["total_candidate_episodes"]}))


if __name__ == "__main__":
    main()
