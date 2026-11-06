"""Public archive conversion regressions, without loading model or running teacher code."""

import base64
import hashlib
import json
from importlib.resources import files

import pytest

from robodata.sft.archive import QUERY, object_sha, observation, outcome_query, public_reply


def test_fixed_physical_split():
    split = json.loads(files("robodata.sft").joinpath("split_manifest.json").read_bytes())
    states = split["states"]
    assert len(states) == len({s["semantic_state_sha256"] for s in states.values()}) == 220
    for partition, size, successes in (("train", 205, 46), ("heldout", 12, 4), ("unscored", 3, 0)):
        matches = [s for s in states.values() if s["split"] == partition]
        assert len(matches) == size
        assert sum(s["success"] is True for s in matches) == successes
    assert all(s["success"] is None for s in states.values() if s["split"] == "unscored")


def test_image_reuse_preserves_slots_and_rejects_substitution(tmp_path):
    image = b"public RGB image bytes"
    url = "data:image/png;base64," + base64.b64encode(image).decode()
    cameras = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
    public = {"task_instruction": "move the object", "previous_result": None, "public_observation": {
        "remaining_actions": 4, "images": [{"data_url": url, "camera": c} for c in cameras]}}
    turn = {"task_instruction": "move the object", "iteration": 1, "previous_execution": None,
            "episode_continues_without_reset": True}
    parts = [{"type": "input_text", "text": json.dumps(v)} for v in (turn, {"remaining_actions": 4})]
    descriptor = {"inline_image_redacted": True, "data_url_characters": len(url), "mime": "image/png",
                  "sha256_data_url": hashlib.sha256(url.encode()).hexdigest()}
    for camera in cameras:
        parts += [{"type": "input_text", "text": "Public camera: " + camera},
                  {"type": "input_image", "detail": "high", "image_url": descriptor}]
    (tmp_path / "images").mkdir()
    converted = observation(parts, public, 1, None, tmp_path)["content"]
    positions = [part["image"] for part in converted if part["type"] == "image"]
    assert len(positions) == 3
    assert len(set(positions)) == len(list((tmp_path / "images").iterdir())) == 1
    assert (tmp_path / positions[0]).read_bytes() == image
    descriptor["data_url_characters"] += 1
    with pytest.raises(ValueError, match="data URL"):
        observation(parts, public, 1, None, tmp_path)


def test_last_feedback_is_public_only_and_give_up_does_not_repeat_it():
    feedback = {"program_ok": True, "stdout": "unmodified tool result"}
    native = {"official": {"success": True}, "execution": feedback, "give_up": False}
    query = outcome_query(native)["content"].split("\n")
    assert query[0] == QUERY and json.loads(query[1]) == {"previous_execution": feedback}
    assert outcome_query({"give_up": True})["content"] == QUERY


def test_response_must_join_terminal_sse_receipt_and_commit(tmp_path):
    text = ' {"code":"print(1)","give_up":false,"summary":"move"} '
    public = {"role": "assistant", "type": "message", "content": [{"type": "output_text", "text": text}]}
    private = {"type": "reasoning", "status": "completed", "encrypted_content": "opaque"}
    response = {"id": "response1", "status": "completed", "output": [private, public]}
    receipt = {"budget_request_id": "request1", "raw_request_ordinal": "request1", "stream_complete": True,
               "payload_sha256": "hash", "http_status": 200, "terminal_event": "response.completed",
               "terminal_response": response, "transport_error": None, "stream_errors": []}
    raw = tmp_path / "response.sse"
    raw.write_text("data: " + json.dumps({"type": "response.completed", "response": response}) + "\n\n")
    replay = [{"type": "reasoning", "encrypted_content": "opaque"}, public]
    assert public_reply(receipt, raw, "request1", "hash", ("response1", object_sha(replay))) == text
    raw.write_text("")
    with pytest.raises(ValueError, match="Terminal SSE"):
        public_reply(receipt, raw, "request1", "hash", ("response1", object_sha(replay)))


def test_only_new_observation_appended_after_matching_complete_history():
    from robodata.sft.archive import verify_replayed_prefix

    user = {"role": "user", "content": [{"type": "input_text", "text": "Turn one"}]}
    public = {"role": "assistant", "type": "message", "content": [{"type": "output_text", "text": "proposal"}]}
    private = {"type": "reasoning", "encrypted_content": "opaque"}
    requests = {1: ("r1", "req_sha", "system", user["content"])}
    commits = {1: ("response", object_sha([private, public]))}
    verify_replayed_prefix([user, private, public], requests, commits, 2)
    with pytest.raises(ValueError, match="Replayed history"):
        verify_replayed_prefix([user, public], requests, commits, 2)
