"""Verify physical renderer and CUDA identity with a per-worker GPU mask."""
from pathlib import Path
import os
import re

def parse_active(log_text):
    found = []
    lines = log_text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r'^\|\s*(\d+)\s*\|.*\|\s*Yes:\s*\d+\s*\|', line)
        if match:
            uuid = re.search(r'\|\s*([0-9a-fA-F]{8})\.\.\s*\|', '\n'.join(lines[index:index+3]))
            if uuid: found.append({'vulkan_index':int(match[1]), 'uuid_prefix':uuid[1].lower()})
    if len(found) != 1: raise RuntimeError('expected exactly one active Vulkan renderer')
    return found[0]



def verify_simulator_gpu(log_path, expected_uuid, *, vulkan_index, cuda_index=0, env=None):
    import torch
    normalized = expected_uuid.lower().removeprefix("gpu-")
    if not re.fullmatch(r"[0-9a-f-]{36}", normalized):
        raise RuntimeError("invalid expected simulator GPU UUID")
    active = parse_active(Path(log_path).read_text(errors="replace"))
    if active["vulkan_index"] != vulkan_index or active["uuid_prefix"] != normalized[:8]:
        raise RuntimeError("Vulkan renderer differs from the reserved simulator GPU")
    if torch.cuda.current_device() != cuda_index:
        raise RuntimeError("simulator current CUDA device differs from requested device")
    props = torch.cuda.get_device_properties(cuda_index)
    actual = str(getattr(props, "uuid", "")).lower().removeprefix("gpu-")
    if not actual or actual != normalized:
        raise RuntimeError("CUDA UUID differs from the reserved simulator GPU")
    planners = {}
    if env is not None:
        for name, planner in env.robot_manager.planner.items():
            device = str(planner.device_cfg.device)
            planners[name] = device
            if device not in ("cuda", f"cuda:{cuda_index}"):
                raise RuntimeError(f"native planner {name} uses unexpected device {device}")
    return {"verified": True, "expected_uuid": expected_uuid, "active_vulkan": active,
            "cuda_uuid": actual, "cuda_device_index": cuda_index,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_name": props.name, "cuda_device_count": torch.cuda.device_count(),
            "planner_devices": planners}
