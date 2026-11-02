"""Six unused-state pairs: current harness vs SAM3 + Contact-GraspNet only."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from build_subset_manifest import canonical_sha256, validate_subset_manifest
from paired_manifest import (PINNED_MODEL, PINNED_INFERENCE, PINNED_SAMPLING,
    PINNED_THINKING, REQUIRED_PROFILE_FILES, sha_file, source_files,
    verify_benchmark_sources)

LEGACY_PROTOCOL = 'robodojo_perception_unused_paired6_v1'
PROTOCOL = 'robodojo_perception_unused_paired6_v2'
ARMS = ('baseline', 'perception')
BOOTSTRAP_PROFILE = 'perception'
BOOTSTRAP_FILES = ('qwen_full/perception_supervisor.py', 'qwen_full/perception_inside.sh',
    'qwen_full/perception_entry.py', 'qwen_full/perception_manifest.py',
    'qwen_full/perception_eval.sbatch', 'qwen_full/full_supervisor.py',
    'qwen_full/paired_supervisor.py', 'qwen_full/perception_services.py')
BOOTSTRAP_ROLES = {'supervisor': 'qwen_full/perception_supervisor.py',
    'inside': 'qwen_full/perception_inside.sh', 'entry': 'qwen_full/perception_entry.py',
    'manifest': 'qwen_full/perception_manifest.py', 'full_supervisor': 'qwen_full/full_supervisor.py',
    'paired_supervisor': 'qwen_full/paired_supervisor.py',
    'perception_services': 'qwen_full/perception_services.py'}
TASKS = ('stack_blocks', 'hang_mugs', 'insert_tubes')
STATES_PER_TASK = 2
PROTECTED_PROFILE_FILES = ('qwen_full/full_model_driver.py', 'qwen_full/full_relay.py',
    'qwen_full/full_entry.py', 'qwen_full/full_episode.py', 'qwen_full/full_adapter.py',
    'qwen_integration/robodojo_adapter.py', 'qwen_integration/native_retime.py',
    'qwen_integration/model_episode.py', 'qwen_integration/receipt_executor.py',
    'ASPIRE/src/cap_harness/api.py', 'ASPIRE/src/cap_harness/contracts.py',
    'ASPIRE/src/cap_harness/runtime.py')
PROTECTED_PROFILE_FILES_V2 = PROTECTED_PROFILE_FILES + ('qwen_integration/native_stdout.py',)
HARDWARE = {'inference_gpus': [1, 2], 'simulator_gpu': 0, 'perception_gpu': 3,
            'execution': 'serial arms and states', 'services_resident_for_both_arms': True}
GRASP_CALIBRATION = {
    'native_closing_direction': [0, 1, 0],
    'source_id': 'X5A.urdf sha256:7aaa308a1a5b6a213d914db86ba09ddd6252f25cb498ac9a148bfa6c7166a423 joint7/8',
    'maximum_contact_separation_m': None,
}
ESTIMATOR_PATCH_FILE = 'qwen_full/cgn_singleton_fix.py'


def estimator_patch_for_profile(files_sha256):
    """Pin the optional in-process correction without changing vendor artifacts."""
    digest = files_sha256.get(ESTIMATOR_PATCH_FILE)
    if digest is None:
        return None
    from cgn_singleton_fix import expected_patch_evidence
    return expected_patch_evidence(digest)


def validate_service_health(health, runtime_manifest, kind, *, estimator_patch=None):
    """Match a loaded GPU model's actual health evidence to pinned checkpoint bytes."""
    if kind not in ('sam3', 'graspnet'):
        raise ValueError('unknown perception service kind')
    checkpoint_suffix = ('/sam3.pt' if kind == 'sam3'
                         else '/contact_graspnet/checkpoints/model.pt')
    # HF snapshot filenames may resolve to content-addressed blob paths.
    # Keep the logical artifact name separate from the exact loaded path.
    matches = [row for row in runtime_manifest.get('files', [])
               if ('/' + str(row.get('file', row.get('path', ''))).lstrip('/')).endswith(checkpoint_suffix)]
    if len(matches) != 1:
        raise ValueError(f'{kind} runtime manifest must identify exactly one checkpoint')
    checkpoint = matches[0]
    expected = {'kind': kind, 'ok': True, 'loaded': True, 'model_loaded': True,
                'offline_weights': True, 'device': 'cuda:0', 'parameter_devices': ['cuda:0'],
                'random_seed': 0, 'checkpoint_sha256': checkpoint['sha256']}
    if (any(health.get(key) != value for key, value in expected.items())
            or type(health.get('parameter_count')) is not int or health['parameter_count'] <= 0):
        raise ValueError(f'{kind} health did not verify pinned weights loaded on the assigned GPU')
    actual_file = health.get('files', {}).get('sam3.pt', {}) if kind == 'sam3' else health.get('checkpoint', {})
    if any(actual_file.get(key) != checkpoint.get(key) for key in ('path', 'bytes', 'sha256')):
        raise ValueError(f'{kind} loaded checkpoint path/size/hash differs from runtime manifest')
    expected_patch = estimator_patch if kind == 'graspnet' else None
    if health.get('estimator_patch') != expected_patch:
        raise ValueError(f'{kind} estimator patch differs from frozen comparison')
    return health


def arm_campaign(manifest, arm, unit_id):
    if arm not in ARMS or unit_id not in {u['unit_id'] for u in manifest['units']}:
        raise ValueError('unknown perception comparison arm/state')
    return f"{manifest['campaign']}_{arm}_{unit_id}"


def arm_order(index):
    return list(ARMS if index % 2 == 0 else reversed(ARMS))


def profile_record(directory, *, comparison, arm, commit):
    files = source_files(directory)
    return {'relative_root': f'perception_profiles/{comparison}/{arm}', 'git_commit': commit,
            'files_sha256': files, 'files_manifest_sha256': canonical_sha256(files)}


def bootstrap_record(profiles):
    """Both arms use one shared frozen launcher; old baseline profiles need not contain it."""
    profile = profiles[BOOTSTRAP_PROFILE]
    files = profile['files_sha256']
    if not set(BOOTSTRAP_FILES) <= files.keys():
        raise ValueError('perception profile lacks the frozen shared bootstrap')
    return {'profile': BOOTSTRAP_PROFILE, 'relative_root': profile['relative_root'],
            'files_sha256': {name: files[name] for name in BOOTSTRAP_FILES},
            'files_manifest_sha256': profile['files_manifest_sha256']}


def require_frozen_bootstrap(manifest):
    if manifest.get('protocol') != PROTOCOL:
        raise ValueError('legacy comparison has no verified frozen bootstrap; audit only, do not resume')
    expected = bootstrap_record(manifest['profiles'])
    if manifest.get('bootstrap') != expected:
        raise ValueError('shared bootstrap differs from pinned perception profile')
    return expected


def bootstrap_source_hashes(manifest):
    """Exact qwen_full inventory actually used by the shared supervisor, including imports."""
    require_frozen_bootstrap(manifest)
    return {Path(name).name: digest
            for name, digest in manifest['profiles'][BOOTSTRAP_PROFILE]['files_sha256'].items()
            if Path(name).parent == Path('qwen_full') and Path(name).suffix in {'.py', '.sh', '.sbatch'}}


def verify_bootstrap(root, manifest, executed_sources):
    record = require_frozen_bootstrap(manifest)
    directory = (Path(root) / record['relative_root']).resolve(strict=True)
    actual = {name: sha_file(directory / name) for name in record['files_sha256']}
    if actual != record['files_sha256']:
        raise ValueError('shared bootstrap source differs from pinned bytes')
    executed = {}
    for role, filename in executed_sources.items():
        expected = directory / BOOTSTRAP_ROLES[role]
        if Path(filename).resolve(strict=True) != expected:
            raise ValueError(f'{role} executed outside the frozen shared bootstrap')
        executed[role] = {'path': str(expected), 'sha256': sha_file(expected)}
    return {'relative_root': record['relative_root'], 'root': str(directory),
            'files_sha256': actual, 'files_manifest_sha256': record['files_manifest_sha256'],
            'executed_sources': executed}


def validate_bootstrap_evidence(evidence, manifest, *, roles, source_sha256=None):
    """Validate saved execution evidence without treating legacy metadata as a freeze."""
    record = require_frozen_bootstrap(manifest)
    if (not isinstance(evidence, dict)
            or any(evidence.get(k) != record[k] for k in
                   ('relative_root', 'files_sha256', 'files_manifest_sha256'))):
        raise ValueError('missing or changed frozen bootstrap evidence')
    if not isinstance(evidence.get('root'), str):
        raise ValueError('missing frozen bootstrap execution root')
    root = Path(evidence['root'])
    relative = Path(record['relative_root'])
    if not root.is_absolute() or '..' in root.parts or root.parts[-len(relative.parts):] != relative.parts:
        raise ValueError('bootstrap execution root differs from frozen profile')
    executed = evidence.get('executed_sources', {})
    if not isinstance(executed, dict) or set(executed) != set(roles):
        raise ValueError('missing or unexpected bootstrap execution roles')
    for role in roles:
        name = BOOTSTRAP_ROLES[role]
        if executed[role] != {'path': str(root / name), 'sha256': record['files_sha256'][name]}:
            raise ValueError(f'{role} path/hash differs from frozen bootstrap')
    if source_sha256 is not None and source_sha256 != bootstrap_source_hashes(manifest):
        raise ValueError('allocation source_sha256 differs from frozen bootstrap inventory')
    return {'verified': True, 'profile': BOOTSTRAP_PROFILE,
            'relative_root': record['relative_root'], 'files_manifest_sha256': record['files_manifest_sha256']}


def select_units(sources, exposures):
    used_hashes = {r['semantic_state_sha256'] for r in exposures}
    units, selected_hashes = [], set()
    for source in sources:
        selected = []
        for layout_id in sorted(source['candidate_layout_ids']):
            digest = source['layout_content_sha256_by_id'][str(layout_id)]
            if digest in used_hashes or digest in selected_hashes:
                continue
            selected.append(layout_id)
            selected_hashes.add(digest)
            if len(selected) == STATES_PER_TASK:
                break
        if len(selected) != STATES_PER_TASK:
            raise ValueError(f"{source['task']}: fewer than two unexposed physical states")
        by_id = {item['layout_id']: item for item in source['layouts']}
        for layout_id in selected:
            group = copy.deepcopy(source)
            group.update(target_episodes=1, candidate_count=1, candidate_layout_ids=[layout_id],
                         episode_ids=[f"{source['group_id']}_l{layout_id}"], layouts=[by_id[layout_id]],
                         runtime_overrides={'eval_nums': 1, 'eval_layout_ids': [layout_id]},
                         perception_source_group_sha256=canonical_sha256(source))
            number = len(units)
            units.append({'unit_id': f'p{number:03d}', 'group': group,
                          'arm_order': arm_order(number),
                          'initial_state_sha256': source['layout_content_sha256_by_id'][str(layout_id)]})
    return units


def validate_manifest(manifest):
    campaign = manifest.get('campaign', '')
    protocol = manifest.get('protocol')
    if (protocol not in (LEGACY_PROTOCOL, PROTOCOL)
            or manifest.get('schema_version') != (1 if protocol == LEGACY_PROTOCOL else 2)):
        raise ValueError('unsupported perception comparison protocol/schema')
    expected = {'arms': list(ARMS), 'tasks': list(TASKS),
                'states_per_task': 2, 'pair_count': 6, 'total_target_episodes': 12,
                'model': PINNED_MODEL, 'inference': PINNED_INFERENCE,
                'sampling': PINNED_SAMPLING, 'thinking': PINNED_THINKING,
                'hardware': HARDWARE, 'offline_skill_discovery': False,
                'grasp_calibration': GRASP_CALIBRATION,
                'is_official_full_evaluation': False}
    if (not campaign.startswith('qwen38_perception_') or not campaign.replace('_', '').isalnum()
            or any(manifest.get(k) != v for k, v in expected.items())):
        raise ValueError('perception comparison scope/model/inference changed')
    if [g['task'] for g in manifest['source_groups']] != list(TASKS):
        raise ValueError('source task order differs from outcome-independent plan')
    for source in manifest['source_groups']:
        if (source['eval_seed'] != 0 or source['dimension'] == 'open' or source['variant'] != 'standard'
                or source['native_options'].get('step_limit') not in (None, source['step_limit'])):
            raise ValueError('source seed/category/native action budget changed')
    inventory = manifest['exposure_inventory']
    if (inventory.get('complete_for_scanned_roots') is not True or inventory.get('unresolved')
            or not inventory.get('scanned_roots') or not manifest.get('inventory_coverage_attestation')
            or len(manifest.get('exposure_inventory_sha256', '')) != 64):
        raise ValueError('incomplete historical exposure inventory')
    for row in inventory['exposures']:
        if len(row.get('semantic_state_sha256', '')) != 64 or not row.get('evidence'):
            raise ValueError('exposure lacks content hash/evidence')
    if manifest['units'] != select_units(manifest['source_groups'], inventory['exposures']):
        raise ValueError('state selection, order or native budget changed')
    for arm in ARMS:
        profile = manifest['profiles'][arm]
        files = profile['files_sha256']
        if (profile['relative_root'] != f'perception_profiles/{campaign}/{arm}'
                or len(profile['git_commit']) != 40
                or any(c not in '0123456789abcdef' for c in profile['git_commit'])
                or canonical_sha256(files) != profile['files_manifest_sha256']
                or not set(REQUIRED_PROFILE_FILES) <= files.keys()):
            raise ValueError('profile identity/hash mismatch')
        for relative, digest in files.items():
            if Path(relative).is_absolute() or '..' in Path(relative).parts or len(digest) != 64:
                raise ValueError('invalid source profile path/hash')
    protected = PROTECTED_PROFILE_FILES if protocol == LEGACY_PROTOCOL else PROTECTED_PROFILE_FILES_V2
    for name in protected:
        first = manifest['profiles']['baseline']['files_sha256'].get(name)
        second = manifest['profiles']['perception']['files_sha256'].get(name)
        if first is None or first != second:
            raise ValueError(f'protected model/motion code differs between arms: {name}')
    if protocol == PROTOCOL:
        require_frozen_bootstrap(manifest)
    elif 'bootstrap' in manifest:
        raise ValueError('legacy manifest cannot claim a retrospectively frozen bootstrap')
    artifacts = manifest.get('perception_artifacts', {})
    if (not isinstance(artifacts.get('runtime_manifest'), dict)
            or not artifacts['runtime_manifest']
            or len(artifacts.get('runtime_manifest_sha256', '')) != 64
            or artifacts.get('runtime_manifest_canonical_sha256') != canonical_sha256(artifacts['runtime_manifest'])
            or artifacts.get('service_runner_sha256') != manifest['profiles']['perception']['files_sha256'].get('qwen_full/perception_service.py')
            or len(artifacts.get('service_runner_sha256', '')) != 64):
        raise ValueError('perception weights/environment/service runner provenance differs from freeze')
    if artifacts.get('estimator_patch') != estimator_patch_for_profile(
            manifest['profiles']['perception']['files_sha256']):
        raise ValueError('estimator patch provenance differs from frozen source profile')
    if len(manifest.get('source_subset_sha256', '')) != 64:
        raise ValueError('missing native source manifest provenance')
    return manifest


def verify_profiles(root, manifest):
    reports = {}
    for arm in ARMS:
        record = manifest['profiles'][arm]
        directory = Path(root) / record['relative_root']
        files = source_files(directory)
        if files != record['files_sha256']:
            raise ValueError(f'{arm} source profile differs from pinned bytes')
        reports[arm] = {'root': str(directory), 'files': len(files),
                        'files_manifest_sha256': canonical_sha256(files)}
    return reports


def verify_layouts(manifest):
    from paired_manifest import verify_layouts as verify
    return verify({'groups': [u['group'] for u in manifest['units']]})


def build_manifest(subset_path, exposure_path, *, comparison, baseline_profile, perception_profile,
                   baseline_commit, perception_commit, coverage_attestation, runtime_manifest_path,
                   full_manifest_path=None):
    subset_path, exposure_path = Path(subset_path), Path(exposure_path)
    subset = validate_subset_manifest(json.loads(subset_path.read_text()))
    inventory = json.loads(exposure_path.read_text())
    runtime_path = Path(runtime_manifest_path)
    runtime = json.loads(runtime_path.read_text())
    full_path = Path(full_manifest_path) if full_manifest_path is not None else subset_path.with_name('full_manifest.json')
    full = json.loads(full_path.read_text())
    if (sha_file(full_path) != subset['source_full_manifest_sha256']
            or inventory.get('source_full_manifest_canonical_sha256') != canonical_sha256(full)):
        raise ValueError('exposure inventory/native manifests belong to different source assets')
    if inventory.get('complete_for_scanned_roots') is not True or inventory.get('unresolved'):
        raise ValueError('resolve historical exposure inventory before selecting states')
    sources = {g['task']: g for g in subset['groups']}
    selected = [sources[task] for task in TASKS]
    result = {'schema_version': 2, 'campaign': comparison, 'protocol': PROTOCOL,
              'arms': list(ARMS), 'tasks': list(TASKS), 'states_per_task': STATES_PER_TASK,
              'pair_count': 6, 'total_target_episodes': 12,
              'source_groups': selected, 'units': select_units(selected, inventory['exposures']),
              'source_subset_sha256': sha_file(subset_path), 'source': subset['source'],
              'exposure_inventory_sha256': sha_file(exposure_path), 'exposure_inventory': inventory,
              'inventory_coverage_attestation': coverage_attestation,
              'perception_artifacts': {'runtime_manifest': runtime,
                  'runtime_manifest_sha256': sha_file(runtime_path),
                  'runtime_manifest_canonical_sha256': canonical_sha256(runtime),
                  'service_runner_sha256': sha_file(Path(perception_profile) / 'qwen_full/perception_service.py')},
              'model': PINNED_MODEL, 'inference': PINNED_INFERENCE,
              'sampling': PINNED_SAMPLING, 'thinking': PINNED_THINKING, 'hardware': HARDWARE,
              'grasp_calibration': GRASP_CALIBRATION,
              'offline_skill_discovery': False, 'is_official_full_evaluation': False,
              'history': 'Full public per-episode history, no compaction or cross-episode learned/reused skills; native model and action budgets unchanged.',
              'selection_note': 'Three preselected non-Open grasp-related task families; first two never-started canonical physical layout hashes per family, independent of outcomes. Six pairs are a small diagnostic, not benchmark-wide performance.',
              'invalid_layout_policy': 'Exact states only, no outcome-dependent replacement. Native abandoned initializations leave incomplete pairs requiring diagnosis.',
              'profiles': {arm: profile_record(path, comparison=comparison, arm=arm, commit=commit)
                   for arm, path, commit in [('baseline', baseline_profile, baseline_commit),
                                             ('perception', perception_profile, perception_commit)]}}
    result['bootstrap'] = bootstrap_record(result['profiles'])
    result['perception_artifacts']['estimator_patch'] = estimator_patch_for_profile(
        result['profiles']['perception']['files_sha256'])
    return validate_manifest(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset-manifest', type=Path, required=True)
    parser.add_argument('--exposure-inventory', type=Path, required=True)
    parser.add_argument('--full-manifest', type=Path)
    parser.add_argument('--runtime-manifest', type=Path, required=True)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--baseline-profile', type=Path, required=True)
    parser.add_argument('--perception-profile', type=Path, required=True)
    parser.add_argument('--baseline-commit', required=True)
    parser.add_argument('--perception-commit', required=True)
    parser.add_argument('--coverage-attestation', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.subset_manifest, args.exposure_inventory,
        comparison=args.comparison, baseline_profile=args.baseline_profile,
        perception_profile=args.perception_profile, baseline_commit=args.baseline_commit,
        perception_commit=args.perception_commit, coverage_attestation=args.coverage_attestation,
        full_manifest_path=args.full_manifest, runtime_manifest_path=args.runtime_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'manifest': str(args.output), 'sha256': sha_file(args.output),
                      'pairs': 6, 'episodes': 12,
                      'states': [u['group']['episode_ids'][0] for u in result['units']]}))


if __name__ == '__main__':
    main()
