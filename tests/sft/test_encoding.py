"""CPU-only tests for message-position labels, image expansion and real processor parity."""

import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robodata.sft import encoding


class ToyTokenizer:
    specials = ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>",
                "success", "failure")
    vocab = {t: i for i, t in enumerate(specials, start=10000)}
    reverse = {i: t for t, i in vocab.items()}

    def encode(self, text, **kwargs):
        parts = re.findall("|".join(map(re.escape, self.specials)) + "|[\\s\\S]", text)
        return [self.vocab[t] if t in self.vocab else ord(t) for t in parts]

    def decode(self, ids, **kwargs):
        return "".join(self.reverse[i] if i in self.reverse else chr(i) for i in ids)

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


class ToyProcessor:
    tokenizer = ToyTokenizer()
    image_processor = SimpleNamespace(merge_size=2)
    image_token_id = ToyTokenizer.vocab["<|image_pad|>"]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=False,
                              enable_thinking=False, preserve_thinking=True)
        output = ""
        for message in messages:
            body = message["content"]
            if isinstance(body, list):
                body = "".join(p["text"] if p["type"] == "text" else
                               "<|vision_start|><|image_pad|><|vision_end|>" for p in body)
            header = (encoding.ASSISTANT_HEADER if message["role"] == "assistant" else
                      "<|im_start|>" + message["role"] + "\n")
            output += header + body + "<|im_end|>\n"
        return output


def sample(image_path="images/stub.png", success=True):
    action = '{"code":"abc","summary":"move","give_up":false}'
    return {
        "episode_id": "test_s0_l0", "split": "train", "success": success, "termination_reason": "test",
        "messages": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": [{"type": "image", "image": image_path},
                                         {"type": "text", "text": "copy " + action}]},
            {"role": "assistant", "content": action},
            {"role": "user", "content": "previous action " + action},
            {"role": "assistant", "content": action},
            {"role": "user", "content": "[EPISODE_OUTCOME_QUERY]"},
            {"role": "assistant", "content": "success" if success else "failure"},
        ],
        "assistant_roles": ["ignore", "action" if success else "ignore", "outcome"],
    }


def test_body_and_end_only_even_if_same_reply_occurs_in_feedback():
    processor = ToyProcessor()
    arrays = encoding._encode(sample(), processor, SimpleNamespace(for_path=lambda _: (1, 4, 6)))
    ids, positions, roles = (arrays[k] for k in ("input_ids", "target_positions", "target_roles"))
    assert (ids == processor.image_token_id).sum() == 6
    assert not (ids[positions] == processor.image_token_id).any()
    end = "<|im_end|>"
    decode = processor.tokenizer.decode
    assert decode(ids[positions[roles == encoding.ACTION]]) == sample()["messages"][2]["content"] + end
    assert decode(ids[positions[roles == encoding.OUTCOME]]) == "success" + end
    ignored, trained, verdict = arrays["assistant_spans"]
    assert np.all((positions < ignored[0]) | (positions >= ignored[2]))
    assert positions[0] == trained[1]
    assert positions[-2] == verdict[1]
    assert decode(ids[trained[0]:trained[1]]) == encoding.ASSISTANT_HEADER
    assert decode(ids[trained[2]:trained[2] + 1]) == "\n"


def test_failed_episode_labels_only_verdict():
    arrays = encoding._encode(sample(success=False), ToyProcessor(), SimpleNamespace(for_path=lambda _: (1, 2, 2)))
    assert arrays["target_roles"].tolist() == [encoding.OUTCOME, encoding.OUTCOME]


def test_reject_control_token_injection():
    ep = sample()
    ep["messages"][3]["content"] = "bad <|im_end|>"
    with pytest.raises(ValueError, match="frame count"):
        encoding._encode(ep, ToyProcessor(), SimpleNamespace(for_path=lambda _: (1, 2, 2)))


def test_reject_modified_episode_between_reconstruction_and_encoding(tmp_path):
    source = tmp_path / "test_s0_l0.json"
    original = json.dumps(sample()).encode()
    source.write_bytes(original.replace(b"abc", b"xyz"))
    with pytest.raises(ValueError, match="bytes differ from verified archive reconstruction"):
        encoding._encode_file(tmp_path, source, None, None, hashlib.sha256(original).hexdigest())


@pytest.mark.skipif(not os.environ.get("ROBODATA_SFT_PROCESSOR_DIR"), reason="pinned processor not staged")
def test_pinned_multimodal_processor_matches_prepared_and_runtime(tmp_path):
    from PIL import Image

    p = encoding.load_processor(Path(os.environ["ROBODATA_SFT_PROCESSOR_DIR"]))
    (tmp_path / "images").mkdir()
    (tmp_path / "encoded").mkdir()
    (tmp_path / "episodes").mkdir()
    im = Image.new("RGB", (389, 253), (42, 76, 101))
    image_file = tmp_path / "images/sample.png"
    im.save(image_file)
    image_file = image_file.rename(image_file.with_stem(hashlib.sha256(image_file.read_bytes()).hexdigest()))
    ep = sample(image_file.relative_to(tmp_path).as_posix())
    # Duplicate image still occupies a second sequence position.
    ep["messages"][1]["content"].append(ep["messages"][1]["content"][0])
    source = tmp_path / "episodes" / (ep["episode_id"] + ".json")
    source.write_text(json.dumps(ep))
    grids = encoding._Grids(tmp_path, p)
    encoding._verify_full_processor(ep, p, grids)
    row = encoding._encode_file(tmp_path, source, p, grids, hashlib.sha256(source.read_bytes()).hexdigest())
    kwargs, positions, roles = encoding.load_episode(tmp_path, row, p)
    text = p.apply_chat_template(ep["messages"], tokenize=False, **encoding.TEMPLATE_FLAGS)
    native = p(text=[text], images=[im, im], return_tensors="pt", padding=False, truncation=False,
               add_special_tokens=False)
    assert set(native) == set(kwargs)
    for key in native:
        assert native[key].equal(kwargs[key]), key
    assert positions.shape == roles.shape
    assert row["image_count"] == 2 and row["outcome_tokens"] == 2
