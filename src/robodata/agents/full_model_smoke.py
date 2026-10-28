"""One three-image interface request through the full native-context driver.

No simulator, robot action, or scored benchmark episode. Generated code is only
parsed as a literal color result and is never executed. Run after local vLLM is
ready; ledger and report are created exclusively, so reruns need new paths.
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

from robodata.agents.full_model_driver import FullQwenDriver, MODEL, NATIVE_CONTEXT_LENGTH


def solid_png(rgb, width=640, height=480):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    raw = (b"\0" + bytes(rgb) * width) * height
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def smoke(output, base_url="http://127.0.0.1:8000/v1", *, _transport=None):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    report = {"purpose": "isolated_full_driver_model_server_smoke", "measured_episode": False,
              "model": MODEL, "expected_context_length": NATIVE_CONTEXT_LENGTH,
              "image_count": 3, "image_shape": [480, 640, 3], "ok": False,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        driver = FullQwenDriver(path.with_suffix(".usage.jsonl"), ["interface_smoke"],
            episode_seeds={"interface_smoke": 0}, base_url=base_url,
            api_documentation="Interface smoke only. No simulator or robot tools are available.",
            _transport=_transport)
        instruction = (
            "This is a model-server vision interface smoke test with no simulator. Inspect the three numbered "
            "images. Return the requested JSON schema. Set done=false and code to one Python assignment: "
            'result = {"colors": ["COLOR1", "COLOR2", "COLOR3"]}, replacing each COLOR with the '
            "dominant image color in order using lowercase English color names. Do not call any robot API. "
            "Use a brief summary.")
        images = [{"camera": f"Image {i}", "data_url": solid_png(rgb)} for i, rgb in
                  enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255)), 1)]
        proposal = driver.next_program("interface_smoke", instruction, {"images": images})
        if proposal.done:
            raise ValueError("smoke requires a result assignment, not done")
        tree = ast.parse(proposal.code, mode="exec")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
            raise ValueError("smoke output must be exactly one literal assignment")
        assignment = tree.body[0]
        if (len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name)
                or assignment.targets[0].id != "result"):
            raise ValueError("smoke output must assign result")
        value = ast.literal_eval(assignment.value)  # No generated code execution.
        report["reported_colors"] = value.get("colors") if isinstance(value, dict) else None
        if report["reported_colors"] != ["red", "green", "blue"]:
            raise ValueError("three-image identification failed")
        report["usage"] = proposal.usage
        report["budget"] = driver.finish_episode("interface_smoke", "interface_smoke_completed_not_scored")
        report["usage_jsonl"] = str(driver.path)
        report["ok"] = True
    except BaseException as exc:
        report["error_type"] = type(exc).__name__
        report["error_code"] = getattr(exc, "code", None)
        report["error"] = str(exc)
        raise
    finally:
        report["wall_s"] = time.monotonic() - started
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()
    print(json.dumps(smoke(args.output, args.base_url)), flush=True)
