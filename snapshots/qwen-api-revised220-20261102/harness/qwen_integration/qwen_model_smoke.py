"""One isolated local three-image request; no episode IDs, relay, or robot action.

Run only after vLLM is ready. Full response (including any model reasoning) is
saved privately; stdout contains status, timing, counts, and final color labels.
Uses the pilot's same schema, medium thinking, final-only-history settings,
8192-token cap, and mandatory exact multimodal /tokenize preflight.
"""
import argparse
import ast
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import struct
import time
import zlib

from local_model_driver import (LoopbackHTTP, MODEL, SYSTEM_PREFIX,
                                build_chat_requests, redact)


def solid_png(rgb, width=640, height=480):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    raw = (b"\0" + bytes(rgb) * width) * height
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def smoke(output_path, base_url="http://127.0.0.1:8000/v1", *, _transport=None):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    report = {"purpose": "isolated_local_model_server_smoke", "measured_episode": False,
              "started_utc": datetime.now(timezone.utc).isoformat(), "ok": False,
              "model": MODEL, "image_count": 3, "image_shape": [480, 640, 3]}
    try:
        local = _transport or LoopbackHTTP(base_url, 400)
        content = [{"type": "text", "text": (
            "This is a model-server vision smoke test with no simulator. Inspect the three numbered images. "
            "Return the requested JSON schema. Set give_up=false. Set code to one Python assignment: "
            "result = {\"colors\": [\"COLOR1\", \"COLOR2\", \"COLOR3\"]}, replacing each COLOR with the "
            "dominant image color in order using lowercase English color names. Do not call any robot API. "
            "Use a brief summary.")}]
        for index, rgb in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255)), 1):
            content += [{"type": "text", "text": f"Image {index}"},
                        {"type": "image_url", "image_url": {"url": solid_png(rgb), "detail": "high"}}]
        messages = [{"role": "system", "content": SYSTEM_PREFIX + "Smoke test only. No robot tools are available."},
                    {"role": "user", "content": content}]
        generation, tokenize = build_chat_requests(messages)
        report["request"] = redact(generation)
        t0 = time.monotonic()
        counted = local("/tokenize", tokenize)
        report["tokenize_wall_s"] = time.monotonic() - t0
        count = counted.get("count")
        model_len = counted.get("max_model_len")
        if type(count) is not int or not 0 < count <= 128000 or type(model_len) is not int or count + 8192 > model_len:
            raise ValueError("invalid/oversize exact prompt count or insufficient server context")
        report["token_count"] = {"count": count, "max_model_len": model_len,
                                 "source": "vllm_multimodal_tokenize", "approximate": False}
        t0 = time.monotonic()
        response = local("/v1/chat/completions", generation)
        report["generation_wall_s"] = time.monotonic() - t0
        report["raw_response"] = redact(response)
        usage = response.get("usage", {})
        if (response.get("model") != MODEL or usage.get("prompt_tokens") != count
                or type(usage.get("completion_tokens")) is not int or not 0 <= usage["completion_tokens"] <= 8192):
            raise ValueError("model identity or usage/preflight mismatch")
        choices = response.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError("incomplete model output")
        message = choices[0]["message"]
        if message.get("tool_calls"):
            raise ValueError("unexpected native tool call")
        parsed = json.loads(message["content"])
        if set(parsed) != {"code", "summary", "give_up"} or parsed["give_up"] is not False or not isinstance(parsed["summary"], str):
            raise ValueError("wrong structured-output fields")
        tree = ast.parse(parsed["code"], mode="exec")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
            raise ValueError("smoke code must be one literal result assignment")
        assignment = tree.body[0]
        if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name) or assignment.targets[0].id != "result":
            raise ValueError("smoke output missing result assignment")
        value = ast.literal_eval(assignment.value)  # Never execute generated Python.
        report["reported_colors"] = value.get("colors")
        if report["reported_colors"] != ["red", "green", "blue"]:
            raise ValueError("three-image color identification failed")
        report["usage"] = usage
        report["ok"] = True
    except BaseException as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)[:1000]
        raise
    finally:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    return {key: report[key] for key in ("ok", "model", "measured_episode", "image_count",
                                         "reported_colors", "tokenize_wall_s", "generation_wall_s", "usage")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()
    print(json.dumps(smoke(args.output, args.base_url)), flush=True)
