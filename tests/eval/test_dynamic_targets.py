"""Auditor target accounting is independent of a particular subset size.

Selection/manifest validation has its own tests. Here the verified manifest
boundary is supplied explicitly to exercise progress accounting without GPU work.
"""
from pathlib import Path
from unittest.mock import patch
import json

from robodata.eval.robodojo.audit import audit_campaign, native_details
from robodata.eval.robodojo.runner import native_result_path, resume_status


def group(name, target, family=None):
    return {"task": name, "group_id": name + "_s0", "family": family or name,
            "dimension": "fixture", "eval_seed": 0,
            "target_episodes": target, "candidate_layout_ids": list(range(50))}


def fixture_manifest():
    # Forty ordinary families request 30 states; finite official pools contribute
    # six and one. A paired family allocates 15 to each configuration, not 30 each.
    groups = [group(f"ordinary_{i}", 30) for i in range(39)]
    groups += [group("paired_standard", 15, "paired"),
               group("paired_random", 15, "paired"),
               group("cover_blocks", 6), group("play_tic_tac_toe", 1)]
    return {"groups": groups}


def test_non767_progress_uses_group_targets_and_family_rollup(tmp_path):
    manifest = fixture_manifest()
    selected = manifest["groups"][0]
    native = native_result_path(tmp_path, selected)
    native.parent.mkdir(parents=True)
    native.write_text(json.dumps({"eval_time": 3, "details": {
        str(i): {"layout_id": i, "success": i != 1, "score": 1 if i != 1 else 0}
        for i in range(3)}}))
    with patch("robodata.eval.robodojo.audit.verify_campaign",
               return_value=({}, manifest, {}, {"tree_sha256": "fixture"})):
        report = audit_campaign(tmp_path)
    assert report["target"] == 1207
    assert report["scored"] == 3
    assert report["success"] == 2
    assert report["success_rate"] == 2 / 3
    assert report["by_family"]["paired"]["target"] == 30
    assert report["by_family"]["cover_blocks"]["target"] == 6
    assert report["by_family"]["play_tic_tac_toe"]["target"] == 1
    assert not report["complete"]
    assert report["errors"] == []
    assert "1207" in report["notes"][0]


def test_non767_full_audit_empty_campaign_is_not_complete(tmp_path):
    with patch("robodata.eval.robodojo.audit.verify_campaign",
               return_value=({}, fixture_manifest(), {}, {"tree_sha256": "fixture"})):
        report = audit_campaign(tmp_path, full=True)
    assert report["target"] == 1207
    assert report["scored"] == 0
    assert report["complete"] is False
    assert report["errors"] == []


def test_thirtieth_native_failure_is_completion_not_a_retry(tmp_path):
    selected = group("fixture", 30)
    native = native_result_path(tmp_path, selected)
    native.parent.mkdir(parents=True)
    details = {str(i): {"layout_id": i, "success": False, "score": 0}
               for i in range(30)}
    native.write_text(json.dumps({"eval_time": 30, "details": details}))
    for i in range(30):
        for camera in ("head", "left_wrist", "right_wrist"):
            (native.parent / f"episode_{i:07d}_cam_{camera}_fail.mp4").write_bytes(b"evidence")
    assert len(native_details({"eval_time": 30, "details": details}, selected)) == 30
    assert resume_status(tmp_path, selected) == "complete"
