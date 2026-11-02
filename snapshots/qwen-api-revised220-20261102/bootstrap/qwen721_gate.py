#!/usr/bin/env python3
"""Attest the explicitly authorized 721-state subset without rewriting Qwen.

The original Qwen 727-stage and 807-plan remain incomplete. This CPU-only gate
requires the full original evidence audit, exact source manifest, and inspected
deliberate-stop receipt. It never creates scores, retries states or submits jobs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from terra_paired_manifest import build_from_files


def verify_from_files(manifest_path, audit_path, *, terminal_receipt_path):
    """Validate every pairing proof, returning only the compact subset receipt."""
    pairing = build_from_files(manifest_path, audit_path,
                               terminal_receipt_path=terminal_receipt_path)
    return copy.deepcopy(pairing["source_subset_gate"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qwen-manifest", type=Path, required=True)
    p.add_argument("--qwen-full-audit", type=Path, required=True)
    p.add_argument("--qwen-terminal-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    result = verify_from_files(args.qwen_manifest, args.qwen_full_audit,
                               terminal_receipt_path=args.qwen_terminal_receipt)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()),
                      "subset_complete_verified": result["subset_complete_verified"],
                      "original_stage_complete_verified": result["original_stage_complete_verified"],
                      "paired_scored_episodes": result["paired_scored_episodes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
