"""Frozen execution contract for a fresh, repaired 767-episode campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from build_subset_manifest import canonical_sha256, validate_subset_manifest
from paired_manifest import (PINNED_MODEL, PINNED_INFERENCE, PINNED_SAMPLING,
    PINNED_THINKING, source_files, sha_file)
from perception_manifest import estimator_patch_for_profile, GRASP_CALIBRATION

PROTOCOL = 'robodojo_repaired_subset767_v1'
DATASET_SHA256 = 'efb941830b989b8a455462f21ff959a00369c3984430138172b98f52bf865373'
HARDWARE = {'inference_gpus': [1, 2], 'simulator_gpus': [0], 'perception_gpu': 3}
PROVIDERS = {'sam3': True, 'contact_graspnet': True, 'offline_skill_discovery': False}
WRAPPERS = ('scripts/container_exec.sh', 'scripts/python_exec.sh')
REQUIRED_BOOTSTRAP = ('qwen_full/repaired_subset_eval.sbatch',
    'qwen_full/repaired_subset_supervisor.py', 'qwen_full/repaired_subset_inside.sh',
    'qwen_full/repaired_subset_entry.py', 'qwen_full/repaired_subset_contract.py',
    'qwen_full/subset_supervisor.py', 'qwen_full/subset_entry.py',
    'qwen_full/full_supervisor.py', 'qwen_full/perception_services.py',
    'qwen_full/perception_service.py')
# Repair-bearing and model-policy bytes from the reviewed 9c71ebb revision.
# New deployment plumbing must not silently alter these implementations.
PINNED_SOURCE_HASHES = {
    'qwen_integration/robodojo_adapter.py': 'ad7980b5950570049c857eddd1eb901543aa3e83d931bb52d981e49c982df472',
    'qwen_integration/native_stdout.py': 'ca3868c19e6ede7de0ed8a7af122c54c2f4b8f7fc8fdae2b5381e2d85743ea99',
    'qwen_integration/receipt_executor.py': '204db972345c9922606d7ac2e782f01d33e64f45c6c82d97ebc0df84d8344a24',
    'ASPIRE/src/cap_harness/api.py': 'c6bc0e75862a4c51465c4a6dbec7e54bcaa0560da235fa1e0d009876b505b077',
    'qwen_full/cgn_singleton_fix.py': '5f2bd4158e1feee72bb700fcf210da47d3a9da134ae44c9000bf6a3895d70acb',
    'qwen_full/full_episode.py': '8a5b5519ac87f9ce14b40439f5154f822400ca5ab4c5154f9dd09d3fb7b03f17',
    'qwen_full/full_model_driver.py': '982f38be14c0ca569a383260a63a353e4ecfd579b4f316898a91a8edfea1d5b3',
    'qwen_full/full_relay.py': '9958970e29bb68eb5416bd59b7d490168b0ea010da422b9dd5a15fe888c81f25',
}


def validate_contract(contract):
    campaign = contract.get('campaign', '')
    if not re.fullmatch(r'qwen38_subset20_repaired_[A-Za-z0-9_]+', campaign):
        raise ValueError('invalid repaired campaign identity')
    expected = {'schema_version': 1, 'protocol': PROTOCOL, 'fixes': [1, 2, 3, 4, 5, 6],
        'dataset_manifest_sha256': DATASET_SHA256, 'target_episodes': 767,
        'model': PINNED_MODEL, 'inference': PINNED_INFERENCE, 'sampling': PINNED_SAMPLING,
        'thinking': PINNED_THINKING, 'providers': PROVIDERS, 'hardware': HARDWARE,
        'grasp_calibration': GRASP_CALIBRATION,
        'limits': {'model_turn_cap': None, 'output_token_cap': None,
                   'episode_walltime_cap': None, 'history_compaction': False,
                   'native_action_budgets': True}, 'reuse_historical_scores': False}
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f'repaired execution contract changed: {key}')
    profile = contract['source_profile']
    if (profile.get('relative_root') != 'harness_profiles/' + campaign
            or not re.fullmatch(r'[a-f0-9]{40}', profile.get('git_commit', ''))):
        raise ValueError('invalid frozen source profile identity')
    files = profile['files_sha256']
    if profile.get('files_manifest_sha256') != canonical_sha256(files):
        raise ValueError('source inventory checksum mismatch')
    for name, digest in files.items():
        if (Path(name).is_absolute() or '..' in Path(name).parts
                or not re.fullmatch(r'[a-f0-9]{64}', digest)):
            raise ValueError('invalid frozen source filename/hash')
    if not set(REQUIRED_BOOTSTRAP) <= files.keys():
        raise ValueError('missing frozen deployment bootstrap')
    if any(files.get(name) != digest for name, digest in PINNED_SOURCE_HASHES.items()):
        raise ValueError('reviewed repair or unchanged model-policy source differs')
    wrappers = contract.get('simulator_wrapper_files', {})
    if (set(wrappers) != set(WRAPPERS)
            or any(not re.fullmatch(r'[a-f0-9]{64}', v) for v in wrappers.values())):
        raise ValueError('missing simulator wrapper inventory')
    artifacts = contract['perception_artifacts']
    if (artifacts.get('runtime_manifest_canonical_sha256') != canonical_sha256(artifacts['runtime_manifest'])
            or not re.fullmatch(r'[a-f0-9]{64}', artifacts.get('runtime_manifest_sha256', ''))
            or artifacts.get('service_runner_sha256') != files['qwen_full/perception_service.py']
            or artifacts.get('estimator_patch') != estimator_patch_for_profile(files)):
        raise ValueError('perception artifacts/patch differ from frozen source')
    return contract


def load_contract(root, campaign):
    directory = Path(root) / 'full_results' / campaign
    contract = validate_contract(json.loads((directory / 'execution_manifest.json').read_text()))
    if contract['campaign'] != campaign:
        raise ValueError('execution contract belongs to another campaign')
    dataset = directory / 'manifest.json'
    if sha_file(dataset) != contract['dataset_manifest_sha256']:
        raise ValueError('dataset differs from original 767-episode candidate queue')
    validate_subset_manifest(json.loads(dataset.read_text()))
    return contract


def verify_profile(root, contract):
    validate_contract(contract)
    root = Path(root).resolve(strict=True)
    profile = contract['source_profile']
    directory = root / profile['relative_root']
    if directory.resolve(strict=True) != directory or directory.is_symlink():
        raise ValueError('source profile must be a real directory at the frozen path')
    files = source_files(directory)
    if files != profile['files_sha256']:
        raise ValueError('source profile differs from frozen inventory')
    for name, digest in contract['simulator_wrapper_files'].items():
        if sha_file(root.parent / name) != digest:
            raise ValueError(f'simulator wrapper changed: {name}')
    return {'profile_root': str(directory), 'git_commit': profile['git_commit'],
            'files_manifest_sha256': profile['files_manifest_sha256'], 'files_count': len(files)}


def verify_executed_sources(root, contract, sources):
    validate_contract(contract)
    profile = contract['source_profile']
    directory = Path(root).resolve(strict=True) / profile['relative_root']
    evidence = {}
    for name, filename in sources.items():
        expected = directory / name
        if (name not in profile['files_sha256'] or Path(filename).resolve(strict=True) != expected
                or sha_file(expected) != profile['files_sha256'][name]):
            raise ValueError(f'executed source outside frozen bytes/path: {name}')
        evidence[name] = {'path': str(expected), 'sha256': sha_file(expected)}
    return {'profile_root': str(directory), 'files_manifest_sha256': profile['files_manifest_sha256'],
            'executed_sources': evidence}


def build_contract(root, campaign, git_commit):
    root = Path(root).resolve(strict=True)
    profile_root = root / 'harness_profiles' / campaign
    files = source_files(profile_root)
    runtime_path = root / 'perception/runtime_manifest.json'
    runtime = json.loads(runtime_path.read_text())
    contract = {'schema_version': 1, 'protocol': PROTOCOL, 'campaign': campaign,
        'fixes': [1, 2, 3, 4, 5, 6], 'target_episodes': 767,
        'dataset_manifest_sha256': DATASET_SHA256, 'model': PINNED_MODEL,
        'inference': PINNED_INFERENCE, 'sampling': PINNED_SAMPLING, 'thinking': PINNED_THINKING,
        'providers': PROVIDERS, 'hardware': HARDWARE, 'grasp_calibration': GRASP_CALIBRATION,
        'limits': {'model_turn_cap': None, 'output_token_cap': None,
                   'episode_walltime_cap': None, 'history_compaction': False,
                   'native_action_budgets': True}, 'reuse_historical_scores': False,
        'source_profile': {'relative_root': str(profile_root.relative_to(root)), 'git_commit': git_commit,
            'files_sha256': files, 'files_manifest_sha256': canonical_sha256(files)},
        'simulator_wrapper_files': {name: sha_file(root.parent / name) for name in WRAPPERS},
        'perception_artifacts': {'runtime_manifest': runtime,
            'runtime_manifest_sha256': sha_file(runtime_path),
            'runtime_manifest_canonical_sha256': canonical_sha256(runtime),
            'service_runner_sha256': files['qwen_full/perception_service.py'],
            'estimator_patch': estimator_patch_for_profile(files)},
        'state_policy': 'Original unique candidate queues; native invalid-initialization replacement only. Fresh rerun, no historical score reuse.'}
    return validate_contract(contract)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--git-commit', required=True)
    args = parser.parse_args()
    contract = build_contract(args.root, args.campaign, args.git_commit)
    path = args.root / 'full_results' / args.campaign / 'execution_manifest.json'
    with path.open('x') as handle:
        json.dump(contract, handle, indent=2, sort_keys=True)
        handle.write('\n')
    load_contract(args.root, args.campaign)
    print(json.dumps({'execution_contract': str(path), 'sha256': sha_file(path),
                      'profile': verify_profile(args.root, contract)}))


if __name__ == '__main__':
    main()
