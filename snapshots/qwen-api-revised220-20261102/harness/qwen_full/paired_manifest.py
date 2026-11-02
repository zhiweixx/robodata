"""Frozen, diagnostic 12-state baseline/fixed harness comparison (24 new runs)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from build_subset_manifest import canonical_sha256, semantic_layout, validate_subset_manifest

PROTOCOL = 'robodojo_harness_plumbing_paired12_v1'
ARMS = ('baseline', 'fixed')
SELECTION = {
    'align_blocks': [0, 13],
    'arrange_largest_number': [8],
    'arrange_largest_number_random': [8],
    'build_tower': [1, 17],
    'classify_objects': [0, 14],
    'classify_objects_by_language': [9, 18],
    'cover_blocks': [1, 12],
}
PINNED_MODEL = {'repo': 'Qwen/Qwen3.8-27B', 'revision': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'}
PINNED_INFERENCE = {'dtype': 'bfloat16', 'tensor_parallel_size': 2,
                    'mtp_tokens': 3, 'native_context_tokens': 262144,
                    'max_num_seqs': 2, 'model_seed': 0}
PINNED_SAMPLING = {'temperature': 1.0, 'top_p': .95, 'top_k': 20, 'min_p': 0.0,
                  'presence_penalty': 0.0, 'repetition_penalty': 1.0}
PINNED_THINKING = {'enable_thinking': True, 'preserve_thinking': False, 'reasoning_effort': 'medium'}
REQUIRED_PROFILE_FILES = ('qwen_full/full_entry.py', 'qwen_full/full_episode.py',
                         'qwen_full/full_model_driver.py', 'qwen_full/full_adapter.py',
                         'qwen_full/full_relay.py',
                         'qwen_integration/aspire_glue.py',
                         'ASPIRE/src/cap_harness/contracts.py')


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files(directory):
    directory = Path(directory)
    rows = {}
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory)
        if any(part in {'.git', '__pycache__'} for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f'profile symlink is not an immutable source file: {relative}')
        if path.is_file():
            rows[str(relative)] = sha_file(path)
    for required in REQUIRED_PROFILE_FILES:
        if required not in rows:
            raise ValueError(f'profile missing {required}')
    return rows


def profile_record(directory, *, comparison, arm, commit):
    files = source_files(directory)
    return {'relative_root': f'paired_profiles/{comparison}/{arm}', 'git_commit': commit,
            'files_sha256': files, 'files_manifest_sha256': canonical_sha256(files)}


def selected_group(source):
    group = copy.deepcopy(source)
    ids = SELECTION[group['task']]
    group['target_episodes'] = len(ids)
    group['candidate_count'] = len(ids)
    group['candidate_layout_ids'] = ids.copy()
    group['episode_ids'] = [f"{group['group_id']}_l{i}" for i in ids]
    by_id = {row['layout_id']: row for row in source['layouts']}
    group['layouts'] = [by_id[i] for i in ids]
    group['runtime_overrides'] = {'eval_nums': len(ids), 'eval_layout_ids': ids.copy()}
    group['paired_source_group_sha256'] = canonical_sha256(source)
    return group


def validate_manifest(manifest):
    campaign = manifest.get('campaign', '')
    if (not campaign.startswith('qwen38_paired_') or not campaign.replace('_', '').isalnum()
            or manifest.get('protocol') != PROTOCOL or manifest.get('arms') != list(ARMS)
            or manifest.get('model') != PINNED_MODEL or manifest.get('inference') != PINNED_INFERENCE
            or manifest.get('selection') != SELECTION or manifest.get('pairs') != 12
            or manifest.get('group_count') != 14 or manifest.get('total_target_episodes') != 24
            or manifest.get('is_official_full_evaluation') is not False):
        raise ValueError('paired comparison scope/model/inference changed')
    sources = manifest['source_groups']
    if [g['task'] for g in sources] != list(SELECTION):
        raise ValueError('paired source configuration order changed')
    if manifest['groups'] != [selected_group(g) for g in sources]:
        raise ValueError('paired state selection or native action budgets changed')
    for group in sources:
        if group['eval_seed'] != 0 or not set(SELECTION[group['task']]) <= set(group['candidate_layout_ids']):
            raise ValueError('selected state outside source subset')
        if group['native_options'].get('step_limit') is not None and group['native_options']['step_limit'] != group['step_limit']:
            raise ValueError('source action budget differs from native options')
    for arm in ARMS:
        profile = manifest['profiles'][arm]
        files = profile['files_sha256']
        if (profile['relative_root'] != f'paired_profiles/{campaign}/{arm}'
                or len(profile['git_commit']) != 40
                or any(c not in '0123456789abcdef' for c in profile['git_commit'])
                or canonical_sha256(files) != profile['files_manifest_sha256']):
            raise ValueError('profile identity/hash mismatch')
        for relative, digest in files.items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts or len(digest) != 64:
                raise ValueError('invalid source profile file')
        if not set(REQUIRED_PROFILE_FILES) <= files.keys():
            raise ValueError('profile incomplete')
    for filename in ('qwen_full/full_model_driver.py', 'qwen_full/full_relay.py'):
        if (manifest['profiles']['baseline']['files_sha256'].get(filename)
                != manifest['profiles']['fixed']['files_sha256'].get(filename)):
            raise ValueError('model driver/relay must be byte-identical between arms')
    if len(manifest.get('source_subset_sha256', '')) != 64:
        raise ValueError('missing parent subset provenance')
    return manifest


def verify_profiles(root, manifest):
    reports = {}
    for arm in ARMS:
        profile = manifest['profiles'][arm]
        directory = Path(root) / profile['relative_root']
        actual = source_files(directory)
        if actual != profile['files_sha256']:
            raise ValueError(f'{arm} source profile differs from pinned files')
        reports[arm] = {'root': str(directory), 'files': len(actual),
                        'files_manifest_sha256': canonical_sha256(actual)}
    return reports


def verify_layouts(manifest):
    checked = []
    for group in manifest['groups']:
        for item in group['layouts']:
            idx = str(item['layout_id'])
            raw = Path(item['source_path']).read_bytes()
            parsed = json.loads(raw)
            if (hashlib.sha256(raw).hexdigest() != group['layout_source_sha256_by_id'][idx]
                    or canonical_sha256(parsed) != group['layout_canonical_json_sha256_by_id'][idx]
                    or canonical_sha256(semantic_layout(parsed)) != group['layout_content_sha256_by_id'][idx]):
                raise ValueError(f"paired initial state drifted: {group['group_id']}_l{idx}")
            checked.append(f"{group['group_id']}_l{idx}")
    return checked


def verify_benchmark_sources(manifest):
    root = Path(manifest['source']['robodojo_root'])
    expected = manifest['source']['sha256']
    for relative, digest in expected.items():
        if sha_file(root / relative) != digest:
            raise ValueError(f'native benchmark source changed: {relative}')
    return len(expected)


def arm_campaign(manifest, arm):
    if arm not in ARMS:
        raise ValueError('unknown comparison arm')
    return manifest['campaign'] + '_' + arm


def gpu_assignment(group_index):
    return {'baseline': 0, 'fixed': 3} if group_index % 2 == 0 else {'baseline': 3, 'fixed': 0}


def build_manifest(subset_path, *, comparison, baseline_profile, fixed_profile,
                   baseline_commit, fixed_commit):
    subset_path = Path(subset_path)
    subset = validate_subset_manifest(json.loads(subset_path.read_text()))
    sources = {g['task']: g for g in subset['groups']}
    selected = [sources[name] for name in SELECTION]
    result = {'schema_version': 1, 'campaign': comparison, 'protocol': PROTOCOL,
              'arms': list(ARMS), 'selection': SELECTION, 'source_groups': selected,
              'groups': [selected_group(g) for g in selected],
              'source_subset_sha256': sha_file(subset_path), 'source': subset['source'],
              'model': PINNED_MODEL, 'inference': PINNED_INFERENCE,
              'pairs': 12, 'group_count': 14, 'total_target_episodes': 24,
              'is_official_full_evaluation': False,
              'selection_note': 'Failure-enriched diagnostic cohort, not an unbiased benchmark success estimate; both arms rerun fresh.',
              'invalid_layout_policy': 'Exact paired states only; no replacement. An abandoned state leaves a missing pair requiring diagnosis.',
              'history': 'Full public history, no compaction; unchanged model/action budgets and done/code semantics.',
              'profiles': {arm: profile_record(path, comparison=comparison, arm=arm, commit=commit)
                  for arm, path, commit in [('baseline', baseline_profile, baseline_commit),
                                            ('fixed', fixed_profile, fixed_commit)]}}
    return validate_manifest(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subset-manifest', type=Path, default=Path(__file__).with_name('subset_manifest.json'))
    p.add_argument('--comparison', required=True)
    p.add_argument('--baseline-profile', required=True, type=Path)
    p.add_argument('--fixed-profile', required=True, type=Path)
    p.add_argument('--baseline-commit', required=True)
    p.add_argument('--fixed-commit', required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = build_manifest(args.subset_manifest, comparison=args.comparison,
        baseline_profile=args.baseline_profile, fixed_profile=args.fixed_profile,
        baseline_commit=args.baseline_commit, fixed_commit=args.fixed_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'manifest': str(args.output), 'sha256': sha_file(args.output), 'pairs': 12, 'episodes': 24}))


if __name__ == '__main__':
    main()
