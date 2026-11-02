"""Check actual Vulkan and CUDA device identity before any measured inference."""
import os
from pathlib import Path
import re


def parse_active(log_text):
    lines = log_text.splitlines()
    found = []
    for i, line in enumerate(lines):
        match = re.match(r'^\|\s*(\d+)\s*\|.*\|\s*Yes:\s*\d+\s*\|', line)
        if match:
            block = '\n'.join(lines[i:i+3])
            uuid = re.search(r'\|\s*([0-9a-fA-F]{8})\.\.\s*\|', block)
            if uuid:
                found.append({'vulkan_index':int(match[1]),'uuid_prefix':uuid[1].lower()})
    if len(found) != 1:
        raise RuntimeError('expected exactly one active Vulkan renderer in startup log')
    return found[0]


def verify_simulator_gpu(log_path, expected_uuid):
    import torch
    normalized = expected_uuid.lower().removeprefix('gpu-')
    if not re.fullmatch(r'[0-9a-f-]{36}', normalized):
        raise RuntimeError('invalid expected simulator GPU UUID')
    active = parse_active(Path(log_path).read_text(errors='replace'))
    if active['vulkan_index'] != 0 or active['uuid_prefix'] != normalized[:8]:
        raise RuntimeError('Vulkan renderer does not use the reserved simulator GPU')
    if torch.cuda.device_count() != 4 or torch.cuda.current_device() != 0:
        raise RuntimeError('simulator must preserve four-device visibility while using CUDA device zero')
    props = torch.cuda.get_device_properties(0)
    actual_uuid = str(getattr(props, 'uuid', '')).lower().removeprefix('gpu-')
    if actual_uuid and actual_uuid != normalized:
        raise RuntimeError('CUDA device differs from the reserved simulator GPU')
    return {'verified':True,'expected_uuid':expected_uuid,'active_vulkan':active,
            'cuda_uuid':actual_uuid or None,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_name':props.name,'cuda_device_count':torch.cuda.device_count()}
