"""Machine-independent native state selection and whole-group assignments."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from .subset_validation import (validate_subset_manifest, canonical_sha256,
                                semantic_layout, POOL_KEYS, ManifestError)

ORIGINAL_MANIFEST_SHA256 = 'efb941830b989b8a455462f21ff959a00369c3984430138172b98f52bf865373'
PORTABLE_SCHEMA = 'robodata.robodojo.subset767.v1'
PORTABLE_MANIFEST_SHA256 = '727769480aa06816094ff6f0ed1cadbd4e0e664d1533b9c0f5718b51d631d5e0'


def _safe_relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or str(path) in ('', '.'):
        raise ManifestError('manifest paths must be nonempty benchmark/assets-relative paths')
    return path


def _native_group(group):
    value = copy.deepcopy(group)
    value.pop('source_group_sha256')
    value['target_episodes'] = value.pop('native_target_episodes')
    value.pop('runtime_overrides')
    pool = value.pop('native_candidate_pool')
    for key in POOL_KEYS: value[key] = pool[key]
    for key in ('layout_content_sha256_by_id', 'layout_canonical_json_sha256_by_id',
                'layout_source_sha256_by_id', 'candidate_selection'):
        value.pop(key)
    return value


def migrate_manifest(original_path):
    """Only locations change; task, candidate order, hashes and budgets do not."""
    raw = Path(original_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_MANIFEST_SHA256:
        raise ManifestError('migration input is not the reviewed 767 subset')
    original = validate_subset_manifest(json.loads(raw))
    result = copy.deepcopy(original)
    assets = Path(original['source']['assets_root'])
    source_groups = {}
    for group in result['groups']:
        source_groups[group['group_id']] = group['source_group_sha256']
        for pool in (group, group['native_candidate_pool']):
            for row in pool['layouts']:
                row['source_path'] = str(Path(row['source_path']).relative_to(assets))
        group['source_group_sha256'] = canonical_sha256(_native_group(group))
    result['source'] = {'git_head': original['source']['git_head'],
                        'sha256': original['source']['sha256'],
                        'layout_root': 'Eval_Layout/RoboDojo/arx_x5'}
    result['source_full_manifest_path'] = None
    result['portable_schema'] = PORTABLE_SCHEMA
    result['migration'] = {'original_manifest_sha256': ORIGINAL_MANIFEST_SHA256,
                           'original_source_group_sha256': source_groups,
                           'changes': 'Only physical source paths are made relative; original state contents/order and native budgets are unchanged.'}
    return validate_manifest(result)


def validate_manifest(manifest):
    if canonical_sha256(manifest)!=PORTABLE_MANIFEST_SHA256:
        raise ManifestError('reviewed semantic manifest bytes changed')
    validate_subset_manifest(manifest)
    if (manifest.get('portable_schema') != PORTABLE_SCHEMA
            or manifest.get('migration', {}).get('original_manifest_sha256') != ORIGINAL_MANIFEST_SHA256):
        raise ManifestError('not the reviewed portable 767 migration')
    for group in manifest['groups']:
        for row in group['native_candidate_pool']['layouts']: _safe_relative(row['source_path'])
    for name in manifest['source']['sha256']: _safe_relative(name)
    return manifest


def load_manifest(path):
    return validate_manifest(json.loads(Path(path).read_text()))


def bind_manifest(manifest, benchmark_root, assets_root=None):
    """Return a resolved copy for source verification; never rewrite selection."""
    manifest = validate_manifest(copy.deepcopy(manifest))
    benchmark_root = Path(benchmark_root).resolve(strict=True)
    assets_root = Path(assets_root or benchmark_root / '.cache/robodojo_assets_repo/Assets').resolve(strict=True)
    for group in manifest['groups']:
        for pool in (group, group['native_candidate_pool']):
            for row in pool['layouts']:
                row['source_path'] = str(assets_root / _safe_relative(row['source_path']))
    manifest['source']['robodojo_root'] = str(benchmark_root)
    manifest['source']['assets_root'] = str(assets_root)
    return manifest


def verify_sources(manifest, benchmark_root, assets_root, group_id=None):
    bound = bind_manifest(manifest, benchmark_root, assets_root)
    benchmark_root = Path(benchmark_root).resolve(strict=True)
    for name, expected in manifest['source']['sha256'].items():
        if hashlib.sha256((benchmark_root / name).read_bytes()).hexdigest() != expected:
            raise ManifestError(f'benchmark source drift: {name}')
    groups = [g for g in bound['groups'] if group_id is None or g['group_id'] == group_id]
    if not groups: raise ManifestError('unknown group')
    for group in groups:
        for row in group['native_candidate_pool']['layouts']:
            i = str(row['layout_id']); raw = Path(row['source_path']).read_bytes(); data = json.loads(raw)
            actual = (hashlib.sha256(raw).hexdigest(), canonical_sha256(data), canonical_sha256(semantic_layout(data)))
            expected = tuple(group[k][i] for k in ('layout_source_sha256_by_id','layout_canonical_json_sha256_by_id','layout_content_sha256_by_id'))
            if actual != expected: raise ManifestError(f'layout source drift: {group["group_id"]}/{i}')
    return {'benchmark_files_verified': len(manifest['source']['sha256']),
            'groups_verified': [g['group_id'] for g in groups]}


def plan_shards(manifest, num_shards=51):
    """Assign complete native groups once; never split replacement pools."""
    validate_manifest(manifest)
    if type(num_shards) is not int or not 1 <= num_shards <= len(manifest['groups']):
        raise ValueError('shard count must be between 1 and 51')
    assignments = [{'shard_index': i, 'group_ids': []} for i in range(num_shards)]
    for i, group in enumerate(manifest['groups']): assignments[i % num_shards]['group_ids'].append(group['group_id'])
    return assignments


def get_group(manifest, group_id):
    rows = [g for g in validate_manifest(manifest)['groups'] if g['group_id'] == group_id]
    if len(rows) != 1: raise ManifestError('unknown group identity')
    return rows[0]
