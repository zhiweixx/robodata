"""Encode complete image/text episodes with the pinned Qwen processor, without truncation."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from robodata.sft.policy import BASE_REVISION as MODEL_REVISION, DATA_REVISION, END_ID, FAILURE_ID, SUCCESS_ID

CONTEXT_LIMIT = 262_144
PROCESSOR_HASHES = {
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "config.json": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
}
TEMPLATE_FLAGS = dict(enable_thinking=False, preserve_thinking=True, add_generation_prompt=False)
IGNORE, ACTION, OUTCOME = 0, 1, 2
ROLE_IDS = {"ignore": IGNORE, "action": ACTION, "outcome": OUTCOME}
ASSISTANT_HEADER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
EXCLUDED_LENGTHS = {
    "imitate_sorting_sequence_s0_l0": 270_194,
    "imitate_sorting_sequence_s0_l1": 361_556,
    "imitate_sorting_sequence_s0_l2": 293_154,
    "imitate_sorting_sequence_s0_l3": 269_075,
    "make_toast_s0_l2": 270_923,
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _write_json(path: Path, data) -> None:
    with NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n")
    Path(f.name).replace(path)


def load_processor(model_path):
    """Load local files only; refuse tokenizer, vision or template drift before encoding."""
    from transformers import AutoProcessor

    model_path = Path(model_path)
    for filename, digest in PROCESSOR_HASHES.items():
        if _sha((model_path / filename).read_bytes()) != digest:
            raise ValueError(f"{filename}: expected Qwen/Qwen3.8-27B@{MODEL_REVISION}")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    # Check the *actual* template selected by Transformers as well as the file on disk.
    if _sha(processor.chat_template.encode()) != PROCESSOR_HASHES["chat_template.jinja"]:
        raise ValueError("selected processor chat template differs from pinned chat_template.jinja")
    for text, token in (("success", SUCCESS_ID), ("failure", FAILURE_ID), ("<|im_end|>", END_ID)):
        if _ids(processor.tokenizer, text) != [token]:
            raise ValueError(f"{text!r} must map to pinned token {token}")
    return processor


def _ids(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)


def _image_paths(messages):
    paths = []
    for message in messages:
        blocks = message["content"]
        if isinstance(blocks, str):
            continue
        for block in blocks:
            if block["type"] == "image":
                if message["role"] != "user":
                    raise ValueError("image outside a user observation")
                paths.append(block["image"])
            elif block["type"] != "text":
                raise ValueError(f"unexpected input block {block['type']!r}")
    return paths


def _image_bytes(root, relative):
    path = Path(relative)
    if path.parent != Path("images") or len(path.stem) != 64:
        raise ValueError(f"invalid content-addressed image path: {relative}")
    raw = (root / path).read_bytes()
    if _sha(raw) != path.stem:
        raise ValueError(f"image checksum mismatch: {relative}")
    return raw


class _Grids:
    """Fixed processor resizing is shape-dependent. Verify real pixel preprocessing per size."""

    def __init__(self, root, processor):
        self.root = root
        self.processor = processor
        self.by_path = {}
        self.by_size = {}

    def for_path(self, path):
        from PIL import Image

        if path not in self.by_path:
            with Image.open(io.BytesIO(_image_bytes(self.root, path))) as image:
                if image.size not in self.by_size:
                    actual = self.processor.image_processor(images=[image.convert("RGB")], return_tensors="pt")
                    grid = tuple(int(n) for n in actual["image_grid_thw"][0])
                    if len(grid) != 3 or min(grid) <= 0:
                        raise ValueError(f"invalid processor image grid {grid}")
                    self.by_size[image.size] = grid
                self.by_path[path] = self.by_size[image.size]
        return self.by_path[path]


def _encode(episode, processor, grids):
    """Walk the framed messages in order; equal strings elsewhere never receive a label."""
    messages = episode["messages"]
    template = processor.apply_chat_template(messages, tokenize=False, **TEMPLATE_FLAGS)
    tokenizer = processor.tokenizer
    ids = np.asarray(_ids(tokenizer, template), dtype=np.int32)
    starts = np.flatnonzero(ids == tokenizer.convert_tokens_to_ids("<|im_start|>"))
    ends = np.flatnonzero(ids == tokenizer.convert_tokens_to_ids("<|im_end|>"))
    if len(starts) != len(messages) or len(ends) != len(messages):
        raise ValueError("message/frame count differs: empty system or injected control token")
    role_names = episode["assistant_roles"]
    n_assistant = sum(msg["role"] == "assistant" for msg in messages)
    if (not n_assistant or len(role_names) != n_assistant or role_names[-1] != "outcome"
            or "outcome" in role_names[:-1] or messages[-1]["role"] != "assistant"):
        raise ValueError("expected one role per assistant turn and exactly one final verdict")
    if not episode["success"] and "action" in role_names:
        raise ValueError("failed episode has supervised actions")
    labels = np.zeros_like(ids, dtype=np.uint8)
    boundaries = []
    prefix = _ids(tokenizer, ASSISTANT_HEADER)
    for message, start, end in zip(messages, starts, ends, strict=True):
        if start >= end:
            raise ValueError("invalid chat frame boundaries")
        if message["role"] != "assistant":
            continue
        role = ROLE_IDS[role_names[len(boundaries)]]
        body = message["content"].strip()
        first = int(start) + len(prefix)
        if ids[start:first].tolist() != prefix or first >= end:
            raise ValueError("empty assistant body or unexpected Qwen thinking/role prefix")
        actual = tokenizer.decode(ids[first:end].tolist(), skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        if actual != body:
            raise ValueError("decoded assistant body differs from public reply")
        if role == OUTCOME and (end - first != 1 or body != ("success" if episode["success"] else "failure")):
            raise ValueError("outcome must be the correct single token before its end marker")
        labels[first:end + 1] = role
        boundaries.append((int(start), first, int(end) + 1, role))

    paths = _image_paths(messages)
    image_grid = np.asarray([grids.for_path(path) for path in paths], dtype=np.int32).reshape(-1, 3)
    image_indices = np.flatnonzero(ids == processor.image_token_id)
    if len(image_indices) != len(paths) or np.any(labels[image_indices]):
        raise ValueError("image placeholder count or position differs from ordered observations")
    repeats = np.ones(len(ids), dtype=np.int32)
    patches = np.prod(image_grid, axis=1, dtype=np.int64)
    merge_length = processor.image_processor.merge_size ** 2
    if np.any(patches % merge_length):
        raise ValueError("vision grid does not divide into whole merged tokens")
    repeats[image_indices] = patches // merge_length
    offsets = np.concatenate(([0], repeats.cumsum(dtype=np.int64)))
    for i, (start, first, end, role) in enumerate(boundaries):
        boundaries[i] = (offsets[start], offsets[first], offsets[end], role)
    labels = np.repeat(labels, repeats)
    positions = np.flatnonzero(labels).astype(np.int32)
    return {
        "input_ids": np.repeat(ids, repeats),
        "target_positions": positions,
        "target_roles": labels[positions],
        "assistant_spans": np.asarray(boundaries, dtype=np.int32),
        "image_grid_thw": image_grid,
    }


def _verify_full_processor(episode, processor, grids):
    """Check the optimized path against native multimodal processing on the first actual turn."""
    from PIL import Image

    msgs = episode["messages"]
    first = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    # A short prefix plus the final verdict permits checking complete conversation framing.
    # Images from any final feedback are deliberately omitted from this geometry check.
    sample = dict(episode, messages=msgs[:first + 1] + [
        {"role": "user", "content": "[EPISODE_OUTCOME_QUERY]"}, msgs[-1]],
        assistant_roles=[episode["assistant_roles"][0], "outcome"])
    images = []
    for path in _image_paths(sample["messages"]):
        with Image.open(io.BytesIO(_image_bytes(grids.root, path))) as im:
            images.append(im.convert("RGB"))
    if not images:
        raise ValueError("first real observation has no camera images")
    template = processor.apply_chat_template(sample["messages"], tokenize=False, **TEMPLATE_FLAGS)
    native = processor(text=[template], images=images, return_tensors="pt", add_special_tokens=False,
                       padding=False, truncation=False)
    ours = _encode(sample, processor, grids)
    np.testing.assert_array_equal(native["input_ids"][0], ours["input_ids"])
    np.testing.assert_array_equal(native["image_grid_thw"], ours["image_grid_thw"])
    np.testing.assert_array_equal(native["mm_token_type_ids"][0], ours["input_ids"] == processor.image_token_id)


def _encode_file(root, source, processor, grids, expected_source_hash, *, verify_parity=False):
    raw = source.read_bytes()
    source_hash = _sha(raw)
    if source_hash != expected_source_hash:
        raise ValueError("episode bytes differ from verified archive reconstruction")
    episode = json.loads(raw)
    if source.stem != episode["episode_id"]:
        raise ValueError(f"episode/source identity differs: {source.name}")
    if verify_parity:
        _verify_full_processor(episode, processor, grids)
    arrays = _encode(episode, processor, grids)
    dest = root / "encoded" / f"{source.stem}.npz"
    with NamedTemporaryFile(dir=dest.parent, prefix=dest.stem, suffix=".tmp", delete=False) as f:
        np.savez_compressed(f, **arrays)
    Path(f.name).replace(dest)
    row = {key: episode[key] for key in ("episode_id", "split", "success", "termination_reason")}
    row.update(
        source_path=source.relative_to(root).as_posix(), source_sha256=source_hash,
        encoded_path=dest.relative_to(root).as_posix(), encoded_sha256=_sha(dest.read_bytes()),
        seq_length=len(arrays["input_ids"]), image_count=len(arrays["image_grid_thw"]),
        teacher_turns=len(episode["assistant_roles"]) - 1,
        action_turns=episode["assistant_roles"].count("action"), outcome_turns=1,
        action_tokens=int(np.sum(arrays["target_roles"] == ACTION)),
        outcome_tokens=int(np.sum(arrays["target_roles"] == OUTCOME)),
    )
    return row


def encode_all(prepared_dir, model_path):
    """Write the reproducible token manifest and report every overlength episode before training."""
    root = Path(prepared_dir)
    sources = sorted((root / "episodes").glob("*.json"))
    if not sources:
        raise ValueError("no reconstructed episode files")
    split_bytes = Path(__file__).with_name("split_manifest.json").read_bytes()
    release = json.loads(split_bytes)
    expected_states = {k: v for k, v in release["states"].items() if v["split"] != "unscored"}
    if release["revision"] != DATA_REVISION or {p.stem for p in sources} != set(expected_states):
        raise ValueError("reconstructed states differ from the pinned scored release")
    archive_bytes = (root / "archive_manifest.json").read_bytes()
    archive = json.loads(archive_bytes)
    if (archive["dataset"], archive["revision"], archive["split_sha256"]) != (
            release["dataset"], DATA_REVISION, _sha(split_bytes)):
        raise ValueError("verified archive identity differs from pinned scored release")
    archive_rows = {r["episode_id"]: r for r in archive["episodes"]}
    if len(archive_rows) != len(archive["episodes"]) or set(archive_rows) != set(expected_states):
        raise ValueError("archive manifest has missing, extra or duplicate scored states")
    for episode_id, record in archive_rows.items():
        frozen = expected_states[episode_id]
        if record["split"] != frozen["split"] or record["success"] is not frozen["success"]:
            raise ValueError(f"{episode_id}: archive manifest split or score differs from pinned state")
    processor = load_processor(model_path)
    grids = _Grids(root, processor)
    (root / "encoded").mkdir(exist_ok=True)
    rows = []
    for source in sources:
        try:
            row = _encode_file(root, source, processor, grids, archive_rows[source.stem]["sha256"],
                               verify_parity=source == sources[0])
            frozen = expected_states[row["episode_id"]]
            if row["split"] != frozen["split"] or row["success"] is not frozen["success"]:
                raise ValueError("split or label differs from fixed physical-state manifest")
            row["semantic_state_sha256"] = frozen["semantic_state_sha256"]
            rows.append(row)
        except ValueError as exc:
            raise ValueError(f"{source.name}: {exc}") from exc
    expected = {"train": (205, 46, 3920, 603), "heldout": (12, 4, 271, 83)}
    counts = {}
    for split, expected_counts in expected.items():
        group = [r for r in rows if r["split"] == split]
        counts[split] = {k: sum(r[k] for r in group) for k in
                         ("success", "teacher_turns", "action_turns", "outcome_turns", "action_tokens", "outcome_tokens")}
        counts[split]["episodes"] = len(group)
        if (len(group), counts[split]["success"], counts[split]["teacher_turns"],
                counts[split]["action_turns"]) != expected_counts:
            raise ValueError(f"unexpected {split} archive counts: {counts[split]}")
    if len(rows) != sum(count["episodes"] for count in counts.values()):
        raise ValueError("unknown split in reconstructed episode")
    lengths = {row["episode_id"]: row["seq_length"] for row in rows}
    over = {name: n for name, n in lengths.items() if n > CONTEXT_LIMIT}
    _write_json(root / "length_report.json", dict(context_limit=CONTEXT_LIMIT,
                over_limit=over, lengths=lengths, max_length=max(lengths.values())))
    manifest = dict(schema_version=1, model="Qwen/Qwen3.8-27B", model_revision=MODEL_REVISION,
                    dataset=release["dataset"], dataset_revision=DATA_REVISION, split_sha256=_sha(split_bytes),
                    archive_manifest_sha256=_sha(archive_bytes),
                    processor_hashes=PROCESSOR_HASHES, template_flags=TEMPLATE_FLAGS, context_limit=CONTEXT_LIMIT,
                    context_ready=not over, over_limit=over, counts=counts, rows=rows)
    manifest["fingerprint"] = _sha(_canonical(manifest))
    _write_json(root / "manifest.json", manifest)
    return select_eligible(root)


def eligible_manifest(prepared_dir):
    """Apply the five measured, ID-specific exclusions to the verified full manifest."""
    root = Path(prepared_dir)
    raw = (root / "manifest.json").read_bytes()
    full = json.loads(raw)
    fingerprint = full.pop("fingerprint")
    if _sha(_canonical(full)) != fingerprint or full["model_revision"] != MODEL_REVISION:
        raise ValueError("Prepared manifest fingerprint or model revision mismatch")
    if full["over_limit"] != EXCLUDED_LENGTHS:
        raise ValueError(f"Unapproved overlength set: {full['over_limit']}")
    if full["context_limit"] != CONTEXT_LIMIT or full["context_ready"] is not False:
        raise ValueError("Unexpected full-release context gate")
    rows = full["rows"]
    split = json.loads(Path(__file__).with_name("split_manifest.json").read_bytes())["states"]
    expected_ids = {sid for sid, state in split.items() if state["split"] != "unscored"}
    if len(rows) != len(expected_ids) or {r["episode_id"] for r in rows} != expected_ids:
        raise ValueError("Full manifest differs from the pinned scored states")
    lengths = {r["episode_id"]: r["seq_length"] for r in rows}
    report = json.loads((root / "length_report.json").read_bytes())
    measured = {sid: length for sid, length in lengths.items() if length > CONTEXT_LIMIT}
    if measured != EXCLUDED_LENGTHS or report != dict(context_limit=CONTEXT_LIMIT,
            over_limit=measured, lengths=lengths, max_length=max(lengths.values())):
        raise ValueError("Full manifest and measured length report disagree with authorized exclusions")
    excluded = []
    retained = []
    for row in rows:
        episode = row["episode_id"]
        pinned = split[episode]
        if (row["split"] != pinned["split"] or row["success"] is not pinned["success"]
                or row["semantic_state_sha256"] != pinned["semantic_state_sha256"]):
            raise ValueError(f"{episode}: physical state, split, or outcome changed")
        if episode in EXCLUDED_LENGTHS:
            if row["success"] or row["termination_reason"] != "environment_ended" or row["action_tokens"]:
                raise ValueError(f"{episode}: unexpected overlong result or supervision")
            excluded.append({k: row[k] for k in ("episode_id", "split", "seq_length", "semantic_state_sha256")})
        else:
            retained.append(row)
    counts = {}
    for subset in ("train", "heldout"):
        group = [r for r in retained if r["split"] == subset]
        counts[subset] = {k: sum(r[k] for r in group) for k in full["counts"][subset] if k != "episodes"}
        counts[subset]["episodes"] = len(group)
    result = dict(full, schema_version=2, rows=retained, counts=counts, context_ready=True, over_limit={},
                  excluded=excluded, full_manifest_sha256=_sha(raw), full_fingerprint=fingerprint)
    result["fingerprint"] = _sha(_canonical(result))
    return result


def select_eligible(prepared_dir):
    """Publish a sidecar using existing encodings, leaving all 217 records in place."""
    result = eligible_manifest(prepared_dir)
    _write_json(Path(prepared_dir) / "eligible_manifest.json", result)
    return result


def load_episode(prepared_dir, row, processor):
    """Materialize the original ordered RGB inputs and gold-token positions for one episode."""
    from PIL import Image
    import torch

    root = Path(prepared_dir)
    src = (root / row["source_path"]).read_bytes()
    encoded = (root / row["encoded_path"]).read_bytes()
    if _sha(src) != row["source_sha256"] or _sha(encoded) != row["encoded_sha256"]:
        raise ValueError(f"{row['episode_id']}: prepared episode checksum mismatch")
    episode = json.loads(src)
    paths = _image_paths(episode["messages"])
    with np.load(io.BytesIO(encoded), allow_pickle=False) as data:
        ids = torch.from_numpy(data["input_ids"].astype(np.int64)).unsqueeze(0)
        expected_grid = data["image_grid_thw"]
        positions = torch.from_numpy(data["target_positions"].astype(np.int64))
        roles = torch.from_numpy(data["target_roles"])
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  mm_token_type_ids=(ids == processor.image_token_id).long())
    # Process ordered images in bounded chunks. Preallocate the final tensor to avoid
    # retaining a second full trajectory worth of temporary normalized pixels at concatenation.
    total_patches = int(np.prod(expected_grid, axis=1, dtype=np.int64).sum())
    offset = 0
    for start in range(0, len(paths), 4):
        images = []
        for path in paths[start:start + 4]:
            with Image.open(io.BytesIO(_image_bytes(root, path))) as im:
                images.append(im.convert("RGB"))
        batch = processor.image_processor(images=images, return_tensors="pt")
        np.testing.assert_array_equal(batch["image_grid_thw"], expected_grid[start:start + len(images)])
        pixels = batch["pixel_values"]
        if start == 0:
            kwargs["pixel_values"] = pixels.new_empty((total_patches, *pixels.shape[1:]))
        kwargs["pixel_values"][offset:offset + len(pixels)] = pixels
        offset += len(pixels)
    if offset != total_patches:
        raise ValueError(f"{row['episode_id']}: incorrect flattened image patch count")
    if paths:
        kwargs["image_grid_thw"] = torch.from_numpy(expected_grid.astype(np.int64))
    return kwargs, positions, roles
