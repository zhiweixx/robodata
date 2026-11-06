"""Reconstruct one public multimodal conversation per scored, published RoboDojo episode."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tarfile
import tempfile
from collections import Counter
from contextlib import ExitStack
from importlib.resources import files
from pathlib import Path

QUERY = "[EPISODE_OUTCOME_QUERY]"
MODE_RULE = (
    "Follow the action instructions below for every robot observation. When the final "
    "user message starts with [EPISODE_OUTCOME_QUERY], answer with exactly success "
    "or failure according to the episode's outcome.\n\n"
)
BAD_SUCCESS_ACTION = ("make_toast_s0_l3", "program_0021.py")
EXPECTED = {
    "train": {"episodes": 205, "successes": 46, "teacher_replies": 3920, "action": 603, "outcome": 205},
    "heldout": {"episodes": 12, "successes": 4, "teacher_replies": 271, "action": 83, "outcome": 12},
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def object_sha(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def rows(path):
    with path.open("rb") as handle:
        for line in handle:
            yield json.loads(line)


def keyed(seq, key):
    data = {}
    for row in seq:
        identifier = row[key]
        require(identifier not in data, f"Duplicate {key}: {identifier}")
        data[identifier] = row
    return data


def write_json(path, obj):
    path.write_bytes(canonical(obj) + b"\n")


def selected_members(index, episodes, attempts):
    """Whitelist release paths for the one native-scored attempt of each exact layout."""
    names = set()
    report_dirs = set()
    raw_dirs = set()
    for episode, meta in episodes.items():
        names.update((meta["native_result_path"], meta["episode_report_path"], attempts[episode]["usage_path"]))
        report_dirs.add(meta["episode_report_path"].rsplit("/", 1)[0])
        raw_dirs.add(attempts[episode]["raw_inference_path"])
    for member in index:
        parent, name = member.rsplit("/", 1)
        if parent in report_dirs and re.fullmatch(r"(public_input_\d{4}\.json|program_\d{4}\.py)", name):
            names.add(member)
        elif parent in raw_dirs and name.startswith("request_") and name.endswith((".sse", ".receipt.json")):
            names.add(member)
    require(names <= index.keys(), "Selected evidence missing from the release file index")
    return names


def materialize(dataset, stage, selected, index, export):
    """Only copy regular files with the exact published member and shard digests."""
    if (dataset / "campaign").is_dir() and (dataset / "native").is_dir():
        for name in selected:
            file = dataset / name
            require(file.is_file() and file.stat().st_size == index[name]["export_bytes"]
                    and digest(file) == index[name]["export_sha256"], f"Restored member differs: {name}")
        return dataset

    import zstandard

    found = set()
    used_shards = {index[n]["shard"] for n in selected}
    for shard in export["shards"]:
        name = shard["path"]
        if name not in used_shards:
            continue
        archive = dataset / name
        require(archive.stat().st_size == shard["bytes"] and digest(archive) == shard["sha256"],
                f"Shard differs: {name}")
        with ExitStack() as stack:
            handle = stack.enter_context(archive.open("rb"))
            stream = stack.enter_context(zstandard.ZstdDecompressor().stream_reader(handle))
            tar = stack.enter_context(tarfile.open(fileobj=stream, mode="r|"))
            for member in tar:
                if member.name not in selected:
                    continue
                expected = index[member.name]
                dest = stage / member.name
                require(member.name not in found and member.isfile() and expected["shard"] == name
                        and expected["export_bytes"] == member.size
                        and dest.resolve().is_relative_to(stage.resolve()), f"Invalid member: {member.name}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                sha = hashlib.sha256()
                with tar.extractfile(member) as src, dest.open("xb") as out:
                    while block := src.read(1 << 20):
                        out.write(block)
                        sha.update(block)
                require(sha.hexdigest() == expected["export_sha256"], f"Member hash differs: {member.name}")
                found.add(member.name)
    require(found == selected, "Release omits required files")
    return stage


def _public_image(source, descriptor, out):
    url = source["data_url"]
    header, data = url.split(",", 1)
    extensions = {"data:image/png;base64": "png", "data:image/jpeg;base64": "jpg"}
    require(header in extensions, "Unrecognized public image MIME type")
    mime = header[5:].split(";")[0]
    require(descriptor == {"inline_image_redacted": True, "data_url_characters": len(url),
                           "mime": mime, "sha256_data_url": hashlib.sha256(url.encode("ascii")).hexdigest()},
            "Image data URL does not match the actual request")
    binary = base64.b64decode(data, validate=True)
    require(bool(binary), "Empty public image")
    relative = f"images/{hashlib.sha256(binary).hexdigest()}.{extensions[header]}"
    target = out / relative
    if not target.exists():  # out is a fresh private staging directory
        target.write_bytes(binary)
    return {"type": "image", "image": relative}


def observation(parts, public, number, previous, out):
    observed = public["public_observation"]
    require(public["previous_result"] == previous, f"Changed native feedback at turn {number}")
    expected_first = {"task_instruction": public["task_instruction"], "iteration": number,
                      "previous_execution": previous, "episode_continues_without_reset": True}
    expected_second = {k: v for k, v in observed.items() if k != "images"}
    images = observed["images"]
    cams = [i["camera"] for i in images]
    require(cams == ["cam_head", "cam_left_wrist", "cam_right_wrist"], f"Unexpected cameras: {cams}")
    require(len(parts) == 2 + 2 * len(images), "Incomplete public observation")
    text_parts = []
    for part, expected in zip(parts[:2], [expected_first, expected_second]):
        require(part["type"] == "input_text" and json.loads(part["text"]) == expected,
                f"Public observation or previous feedback differs at turn {number}")
        text_parts.append({"type": "text", "text": part["text"]})
    for i, image in enumerate(images):
        label, visual = parts[2 + 2*i:4 + 2*i]
        require(label == {"type": "input_text", "text": "Public camera: " + image["camera"]}
                and visual["type"] == "input_image" and visual["detail"] == "high", "Camera order differs")
        text_parts.append({"type": "text", "text": label["text"]})
        text_parts.append(_public_image(image, visual["image_url"], out))
    return {"role": "user", "content": text_parts}


def verify_replayed_prefix(items, requests, commits, current):
    """Match the preceding observations and all committed provider output before dropping duplicate history."""
    groups = []
    for item in items:
        if item.get("role") == "user":
            groups.append((item["content"], []))
        else:
            require(bool(groups), "Replayed reply precedes its public observation")
            groups[-1][1].append(item)
    require(len(groups) == current - 1, f"Missing or extra history before turn {current}")
    for number, (user, response_items) in enumerate(groups, 1):
        require(user == requests[number][3] and object_sha(response_items) == commits[number][1],
                f"Replayed history differs at turn {number} in request {current}")


def read_usage(path, episode, iterations):
    requests, commits = {}, {}
    for row in rows(path):
        event = row["event"]
        if event not in ("program_input", "response_history_committed"):
            continue
        require(row["episode_id"] == episode, f"Foreign request in {episode}")
        i = row["iteration"]
        table = requests if event == "program_input" else commits
        require(i not in table, f"Ambiguous {event}, {episode} turn {i}")
        if event == "program_input":
            # Each request already contains every earlier exchange. Retain only its new observation.
            items = row["request"]["input"]
            require(items[0]["role"] == "system" and items[-1]["role"] == "user", "Invalid request roles")
            require(hashlib.sha256(items[0]["content"].encode()).hexdigest() == row["instructions_sha256"]
                    and row["raw_request_ordinal"] == row["budget_request_id"], "Request identity differs")
            verify_replayed_prefix(items[1:-1], requests, commits, i)
            table[i] = (row["budget_request_id"], row["request_sha256"], items[0]["content"], items[-1]["content"])
        else:
            table[i] = (row["response_id"], row["replay_output_sha256"])
    expected = set(range(1, len(iterations) + 1))
    require(set(requests) == set(commits) == expected, f"Report and accepted request turns differ: {episode}")
    return requests, commits


def public_reply(receipt, raw, request_id, request_sha, commit):
    require(receipt["http_status"] == 200 and receipt["stream_complete"] is True
            and not receipt["transport_error"] and not receipt["stream_errors"]
            and receipt["terminal_event"] == "response.completed"
            and receipt["budget_request_id"] == receipt["raw_request_ordinal"] == request_id
            and receipt["payload_sha256"] == request_sha,
            f"Receipt differs from request {request_id}")
    response = receipt["terminal_response"]
    require(response["id"] == commit[0] and response["status"] == "completed", "Uncommitted response")
    terminal = []
    with raw.open("rb") as f:
        for line in f:
            if line.startswith(b"data:") and b"response.completed" in line:
                packet = json.loads(line[5:].strip())
                if packet.get("type") == "response.completed":
                    terminal.append(packet["response"])
    require(terminal == [response], f"Terminal SSE differs from receipt {request_id}")
    replay = []
    messages = []
    for item in response["output"]:
        if item["type"] == "reasoning":
            replay.append({k: v for k, v in item.items() if k != "status"})
        else:
            require(item["type"] == "message" and item["role"] == "assistant", "Unexpected teacher output")
            replay.append(item)
            require(all(p["type"] == "output_text" for p in item["content"]), "Unexpected teacher content")
            messages.append("".join(p["text"] for p in item["content"]))
    require(len(messages) == 1 and object_sha(replay) == commit[1], "Unverified committed teacher output")
    return messages[0]


def outcome_query(last):
    text = QUERY
    if not last["give_up"]:
        text += "\n" + json.dumps({"previous_execution": last["execution"]}, sort_keys=True, allow_nan=False)
    return {"role": "user", "content": text}


def convert_episode(root, out, meta, attempt, split, index):
    episode = meta["episode_id"]
    report_path, result_path = meta["episode_report_path"], meta["native_result_path"]
    require(index[report_path]["export_sha256"] == meta["episode_report_sha256"] and
            index[result_path]["export_sha256"] == meta["native_result_sha256"], "Score provenance differs")
    report = read_json(root / report_path)
    details = read_json(root / result_path)["details"]
    require(set(details) == {"0"} and meta["native_index"] == 0, "Unexpected native index")
    detail = details["0"]
    success = meta["success"]
    require(type(success) is bool and report["native_score_committed"] is True and report["status"] == "completed"
            and report["episode_id"] == episode and report["layout_id"] == detail["layout_id"] == meta["layout_id"]
            and report["official"]["official_success"] is detail["success"] is success
            and detail["score"] == meta["native_score"] and report["termination_reason"] == meta["termination_reason"],
            f"Committed native score differs: {episode}")
    iterations = report["iterations"]
    require(iterations, f"No teacher turns: {episode}")
    requests, commits = read_usage(root / attempt["usage_path"], episode, iterations)
    report_dir = Path(report_path).parent
    raw_dir = Path(attempt["raw_inference_path"])
    messages, roles, turns = [], [], []
    previous = None
    stored_system = requests[1][2]
    messages.append({"role": "system", "content": MODE_RULE + stored_system})
    for number, entry in enumerate(iterations, 1):
        request_id, request_sha, system, parts = requests[number]
        require(entry["iteration"] == number and system == stored_system
                and request_id.startswith(f'{attempt["allocation_id"]}-{attempt["worker_id"]}-')
                and re.fullmatch(r"[\w-]+", request_id), f"Invalid turn {number}")
        public_path = report_dir / f"public_input_{number - 1:04d}.json"
        messages.append(observation(parts, read_json(root / public_path), number, previous, out))
        receipt_path = raw_dir / f"request_{request_id}.receipt.json"
        raw_path = raw_dir / f"request_{request_id}.sse"
        text = public_reply(read_json(root / receipt_path), root / raw_path, request_id, request_sha, commits[number])
        proposal = json.loads(text)
        require(set(proposal) == {"code", "summary", "give_up"} and proposal["give_up"] is entry["give_up"]
                and proposal["summary"] == entry["summary"] and entry["usage"]["response_id"] == commits[number][0],
                "Public response and report disagree")
        role = "ignore"
        source_sha = None
        if entry["give_up"]:
            require(number == len(iterations) and not proposal["code"].strip() and "execution" not in entry
                    and "source" not in entry and meta["termination_reason"] == "model_give_up", "Invalid give_up")
        else:
            source = entry["source"]
            require(source == f"program_{number - 1:04d}.py", "Unexpected executed source")
            require((root / report_dir / source).read_bytes() == proposal["code"].encode("utf-8"),
                    "Public teacher code differs from executed program")
            source_sha = index[str(report_dir / source)]["export_sha256"]
            previous = entry["execution"]
            require(type(previous["program_ok"]) is bool, "Invalid public execution status")
            if success:
                require(previous["program_ok"] == ((episode, source) != BAD_SUCCESS_ACTION),
                        f"Unexpected successful episode execution exception: {episode}/{source}")
                if previous["program_ok"]:
                    role = "action"
        messages.append({"role": "assistant", "content": text})
        roles.append(role)
        turns.append({"iteration": number, "request_id": request_id, "request_sha256": request_sha,
                      "response_id": commits[number][0], "receipt_sha256": index[str(receipt_path)]["export_sha256"],
                      "sse_sha256": index[str(raw_path)]["export_sha256"], "source": entry.get("source"),
                      "source_sha256": source_sha, "public_input_sha256": index[str(public_path)]["export_sha256"]})
    messages.extend((outcome_query(iterations[-1]), {"role": "assistant", "content": "success" if success else "failure"}))
    roles.append("outcome")
    # Native evaluator state stays outside the model-visible trajectory. Public code/feedback can
    # legitimately discuss task success; check actual private paths instead of censorship by keyword.
    path_markers = [report_path, result_path, meta["original_native_result_path"]]
    path_markers.extend(Path(video["path"]).name for video in meta["videos"])
    public_bytes = canonical(messages)
    require(all(marker.encode() not in public_bytes for marker in path_markers), "Private result path in trajectory")
    sample = {"episode_id": episode, "split": split, "success": success, "termination_reason": meta["termination_reason"],
              "messages": messages, "assistant_roles": roles,
              "provenance": {"report_sha256": meta["episode_report_sha256"], "native_sha256": meta["native_result_sha256"],
                             "allocation_id": attempt["allocation_id"], "worker_id": attempt["worker_id"],
                             "usage_sha256": index[attempt["usage_path"]]["export_sha256"], "turns": turns,
                             "semantic_state_sha256": meta["semantic_state_sha256"],
                             "canonical_json_sha256": meta["canonical_json_sha256"],
                             "source_layout_sha256": meta["source_layout_sha256"]}}
    path = out / "episodes" / f"{episode}.json"
    write_json(path, sample)
    return {"episode_id": episode, "split": split, "success": success, "sha256": digest(path),
            "teacher_replies": len(iterations), "action": roles.count("action"), "outcome": roles.count("outcome")}


def reconstruct(dataset_dir: Path, output_dir: Path):
    """Verify pinned release and atomically publish all scored episodes into a new directory."""
    dataset, out = Path(dataset_dir), Path(output_dir)
    out.parent.mkdir(exist_ok=True, parents=True)
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())), f"Expected empty output: {out}")
    split_bytes = files("robodata.sft").joinpath("split_manifest.json").read_bytes()
    split = json.loads(split_bytes)
    for name, sha in split["metadata_sha256"].items():
        require(digest(dataset / name) == sha, f"Pinned release metadata differs: {name}")
    episodes = keyed(rows(dataset / "episodes.jsonl"), "episode_id")
    attempts = keyed(rows(dataset / "attempts.jsonl"), "episode_id")
    states = keyed(rows(dataset / "planned_states.jsonl"), "episode_id")
    require(states.keys() == split["states"].keys() == attempts.keys() and len(states) == 220,
            "Split and release state inventory differ")
    require(episodes.keys() == {e for e, s in split["states"].items() if s["split"] != "unscored"},
            "Scored states differ from frozen split")
    for key in ["semantic_state_sha256", "canonical_json_sha256", "source_layout_sha256"]:
        require(len({row[key] for row in episodes.values()}) == len(episodes), f"Duplicate physical layout ({key})")
    for e, row in states.items():
        pinned = split["states"][e]
        require(pinned["semantic_state_sha256"] == row["semantic_state_sha256"]
                and pinned["success"] is row["success"], f"Physical layout or label differs: {e}")
        if e in episodes:
            require(episodes[e]["semantic_state_sha256"] == pinned["semantic_state_sha256"]
                    and episodes[e]["success"] is pinned["success"], f"Native index identity differs: {e}")
    index = keyed(rows(dataset / "file_manifest.jsonl"), "member")
    selected = selected_members(index, episodes, attempts)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".reconstruct-") as tmp:
        stage = Path(tmp)
        root = materialize(dataset, stage / "source", selected, index, read_json(dataset / "export_manifest.json"))
        build = stage / "result"
        (build / "episodes").mkdir(parents=True)
        (build / "images").mkdir()
        records = []
        counts = {k: Counter() for k in EXPECTED}
        for e, row in sorted(episodes.items()):
            which = split["states"][e]["split"]
            try:
                record = convert_episode(root, build, row, attempts[e], which, index)
            except Exception as exc:
                raise ValueError(f"Failed to reconstruct {e}: {exc}") from exc
            records.append(record)
            counts[which].update(episodes=1, successes=int(record["success"]), teacher_replies=record["teacher_replies"],
                                 action=record["action"], outcome=record["outcome"])
        require(counts == EXPECTED, f"Target or split counts differ: {counts}")
        manifest = {"dataset": split["dataset"], "revision": split["revision"], "counts": counts,
                    "split_sha256": hashlib.sha256(split_bytes).hexdigest(), "episodes": records,
                    "unscored": sorted(states.keys() - episodes.keys())}
        write_json(build / "archive_manifest.json", manifest)
        # Publish images, conversations and manifest as one unit, never a partially valid release.
        os.replace(build, out)
    return manifest
