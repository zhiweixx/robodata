"""Start pinned, node-local perception services inside a Slurm allocation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

import full_supervisor as full
from perception_manifest import (ESTIMATOR_PATCH_FILE, estimator_patch_for_profile,
                                 validate_service_health)

CALIBRATION = {
    'native_closing_direction': [0, 1, 0],
    'source_id': 'X5A.urdf sha256:7aaa308a1a5b6a213d914db86ba09ddd6252f25cb498ac9a148bfa6c7166a423 joint7/8',
    'maximum_contact_separation_m': None,
}


def verify_artifacts(path):
    manifest = json.loads(Path(path).read_text())
    for record in manifest['files']:
        source = Path(record['path'])
        with source.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != record['sha256'] or source.stat().st_size != record['bytes']:
            raise RuntimeError(f'perception artifact drift: {source}')
    return manifest


def start_services(campaign):
    """Both arms keep the same resident services; only treatment can call them."""
    root, runtime = campaign.root, campaign.runtime
    artifacts = verify_artifacts(root / 'perception/runtime_manifest.json')
    source = getattr(campaign, 'source_profile', root)
    if 'perception_artifacts' in campaign.manifest:
        source = root / campaign.manifest['profiles']['perception']['relative_root']
        expected = campaign.manifest['perception_artifacts']
        if (artifacts != expected['runtime_manifest']
                or full.sha_file(root / 'perception/runtime_manifest.json') != expected['runtime_manifest_sha256']
                or full.sha_file(source / 'qwen_full/perception_service.py') != expected['service_runner_sha256']):
            raise RuntimeError('perception artifacts differ from frozen comparison')
    patch_file = source / ESTIMATOR_PATCH_FILE
    estimator_patch = estimator_patch_for_profile(
        {ESTIMATOR_PATCH_FILE: full.sha_file(patch_file)} if patch_file.exists() else {})
    if ('perception_artifacts' in campaign.manifest
            and estimator_patch != campaign.manifest['perception_artifacts'].get('estimator_patch')):
        raise RuntimeError('estimator patch differs from frozen comparison')
    for kind in ('sam3', 'graspnet'):
        python = root / 'ASPIRE' / ('.venv-' + kind) / 'bin/python'
        actual = json.loads(subprocess.check_output([str(python), '-c',
            'import importlib.metadata as m,json; print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))'], text=True))
        if actual != artifacts['environments'][kind]:
            raise RuntimeError(f'{kind} installed environment differs from manifest')
    env = dict(campaign.env, CUDA_VISIBLE_DEVICES=campaign.gpus[3]['uuid'],
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH=str(source / 'ASPIRE/src'), PYTHONDONTWRITEBYTECODE='1')
    urls, processes, health = {}, [], {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for kind, venv in [('sam3', '.venv-sam3'), ('graspnet', '.venv-graspnet')]:
        port = full.unused_loopback_port()
        url = f'http://127.0.0.1:{port}'
        command = [str(root / 'ASPIRE' / venv / 'bin/python'), '-B',
                   str(source / 'qwen_full/perception_service.py'), '--kind', kind,
                   '--host', '127.0.0.1', '--port', str(port),
                   '--sam3-manifest', str(root / 'perception/sam3_manifest.json'),
                   '--vendor-root', str(root / 'ASPIRE/third_party')]
        with (runtime / f'{kind}.log').open('w') as output:
            process = campaign.track(kind, subprocess.Popen(command, env=env, cwd=root,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True))
        processes.append((kind, process))
        campaign.record('perception_service_started', kind=kind, pid=process.pid, url=url)
        deadline = time.monotonic() + 1200
        while True:
            if process.poll() is not None or campaign.stop.is_set():
                raise RuntimeError(f'{kind} exited before readiness')
            try:
                with opener.open(url + '/healthz', timeout=5) as response:
                    health[kind] = json.load(response)
                if health[kind].get('loaded') is not True or health[kind].get('ok') is not True:
                    raise ValueError('perception service is not loaded')
                break
            except (OSError, ValueError):
                if time.monotonic() > deadline:
                    raise RuntimeError(f'{kind} startup deadline exceeded')
                time.sleep(1)
        urls[kind + '_url'] = url
        validate_service_health(health[kind], artifacts, kind, estimator_patch=estimator_patch)
        campaign.record('perception_service_ready', kind=kind, health=health[kind])
    config = dict(urls, grasp_calibration=CALIBRATION, provenance={
        'runtime_manifest': artifacts, 'runtime_manifest_sha256': full.sha_file(root / 'perception/runtime_manifest.json'),
        'service_health': health, 'gpu_uuid': campaign.gpus[3]['uuid'],
        'service_runner_sha256': full.sha_file(source / 'qwen_full/perception_service.py'),
        'offline_skill_discovery': False})
    full.write_json(runtime / 'providers.json', config)
    campaign.report['perception_services'] = config['provenance']
    full.write_json(runtime / 'allocation.json', campaign.report)
    return config, processes
