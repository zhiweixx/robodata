#!/usr/bin/env python3
"""Plan an isolated official summary from a successful full-audit JSON.

Default is read-only. --create-view creates a NEW directory containing only
symlinks to the 162 verified native group directories. This helper never runs
the official summarizer, any evaluation, a model, or a cluster command.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex

from audit_full_results import OFFICIAL, expected_result_path, read_json


def literal(source, name):
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if len(nodes) != 1:
        raise ValueError(f"official summarizer must define one literal {name}")
    return ast.literal_eval(nodes[0].value)


def plan_summary(audit_path, view_root, summarizer_path, *, python="python3"):
    audit_path = Path(audit_path).resolve(strict=True)
    summarizer_path = Path(summarizer_path).resolve(strict=True)
    view_root = Path(view_root).expanduser().resolve()
    report = read_json(audit_path)
    scope, groups = report.get("official_scope", {}), report.get("groups", {})
    scoring, evidence = report.get("scoring", {}), report.get("evidence", {})
    if (report.get("campaign_complete") is not True
            or report.get("audit_mode") == "progress_only"
            or report.get("final_evidence_audit_required") is True
            or report.get("errors") != [] or "usage" not in report
            or scope.get("matches") is not True
            or scope.get("expected") != OFFICIAL or scope.get("observed") != OFFICIAL
            or groups.get("target") != 162 or groups.get("native_complete_verified") != 162
            or groups.get("states") != {"complete": 162}
            or scoring.get("target") != 6300 or scoring.get("scored") != 6300
            or scoring.get("scored_in_verified_complete_groups") != 6300
            or scoring.get("remaining_to_target") != 0
            or evidence.get("scored_episodes_with_three_matching_nonempty_videos") != 6300
            or evidence.get("scored_episodes_with_committed_report_and_native_budget") != 6300):
        raise ValueError("requires a successful full evidence audit of all 6300 episodes and 162 groups")

    source = summarizer_path.read_text(encoding="utf-8")
    dimensions = literal(source, "DIMENSIONS")
    if (literal(source, "EXPECTED_SEEDS") != [0, 1, 2]
            or literal(source, "STANDALONE_EPISODES") != 50
            or literal(source, "PAIRED_HALF_EPISODES") != 25
            or {key: len(value) for key, value in dimensions.items()} != {
                "Generalization": 12, "Precision": 8, "Long-Horizon": 8, "Memory": 6, "Open": 8}):
        raise ValueError("official summarizer scope changed; inspect before preparing a view")
    families = {task for tasks in dimensions.values() for task in tasks}
    tasks = families | {task + "_random" for task in dimensions["Generalization"]}
    if len(families) != 42 or len(tasks) != 54:
        raise ValueError("official task family/configuration identities are not unique")
    expected = {f"{task}_s{seed}": (task, seed) for task in tasks for seed in (0, 1, 2)}
    directories = report.get("official_summarizer_input", {}).get("verified_native_group_directories")
    if not isinstance(directories, dict) or set(directories) != set(expected):
        raise ValueError("audit must include exactly the official 162 verified native group directories")
    campaign_dir = Path(report["campaign_dir"])
    campaign = report["campaign"]
    if (not campaign_dir.is_absolute() or campaign_dir.name != campaign
            or not campaign.replace("_", "").isalnum()):
        raise ValueError("invalid audited campaign identity")
    simroot = campaign_dir.parents[2]
    native_root = simroot / "RoboDojo/eval_result/RoboDojo"
    if view_root.is_relative_to(native_root.resolve()):
        raise ValueError("isolated view must be outside the raw native result tree")
    if view_root.exists():
        raise ValueError("isolated view must be a new directory; existing trees cannot be reused")
    links = []
    for gid, (task, seed) in sorted(expected.items()):
        expected_dir = expected_result_path(simroot, campaign, {
            "group_id": gid, "task": task, "eval_seed": seed}).parent
        source_dir = Path(directories[gid])
        if not source_dir.is_absolute() or source_dir != expected_dir:
            raise ValueError(f"unexpected campaign/group source path: {gid}")
        if not source_dir.is_dir() or not (source_dir / "_result.json").is_file():
            raise ValueError(f"verified native result no longer exists: {gid}")
        destination = view_root / source_dir.relative_to(native_root)
        links.append({"group_id": gid, "source": str(source_dir), "link": str(destination)})
    command = ["env", f"ROBODOJO_EVAL_ROOT={view_root}", python, "-B", str(summarizer_path)]
    create_command = [python, "-B", str(Path(__file__).resolve()), "--audit", str(audit_path),
                      "--view-root", str(view_root), "--summarizer", str(summarizer_path),
                      "--python", python, "--create-view"]
    return {"schema_version": 1, "mode": "dry_run", "campaign": campaign,
            "audit_path": str(audit_path), "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "summarizer_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "view_root": str(view_root), "group_count": len(links), "links": links,
            "create_view_command": shlex.join(create_command),
            "official_summary_command": shlex.join(command),
            "official_summary_argv": command, "output_markdown": str(view_root / "_summary.md"),
            "expected_official_output": "126 filled task/seed cells, 1 complete policy, 162 selected folders, 0 result/video mismatches",
            "note": "Only exact audited campaign directories enter the new view. The helper does not run the official summary. Run the full audit immediately before this step; source files remain referenced by symlink."}


def create_view(plan):
    """Only mutate the newly created view; never the native source directories."""
    root = Path(plan["view_root"])
    root.mkdir(parents=True, exist_ok=False)
    for row in plan["links"]:
        destination = Path(row["link"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(row["source"], target_is_directory=True)
    plan["mode"] = "view_created"
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="Saved successful full audit, without --summary-only")
    parser.add_argument("--view-root", type=Path, required=True, help="New directory outside the native result tree")
    parser.add_argument("--summarizer", type=Path, required=True, help="Pinned RoboDojo scripts/internal/summarize_result.py")
    parser.add_argument("--python", default="python3", help="Python executable to print in the commands")
    parser.add_argument("--create-view", action="store_true", help="Create the new symlink view; still do not execute the summarizer")
    args = parser.parse_args(argv)
    try:
        plan = plan_summary(args.audit, args.view_root, args.summarizer, python=args.python)
        if args.create_view:
            create_view(plan)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        parser.exit(2, f"summary plan error: {exc}\n")
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
