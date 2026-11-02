"""Read-only inventory of previously started RoboDojo layouts; no model/simulator.

Search only explicit experiment/result roots, never a model/assets home directory.
Counts interrupted, unscored, smoke and diagnostic attempts as exposure. Planned
candidate manifests alone are not execution evidence. Unknown identities are
reported and must be resolved before an unused-state manifest is frozen.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from build_subset_manifest import canonical_sha256, semantic_layout, HASH_METHOD

SKIP_DIRS = {'.git', '__pycache__', '.secrets', 'secrets', 'hf-cache', 'models',
             'Assets', 'assets', 'ASPIRE', 'paired_profiles', 'perception_profiles',
             'profiles', '.venv', '.venv-runtime', 'site-packages', 'node_modules'}
JSONL = {'usage.jsonl', 'attempts.jsonl'}
NAMES = {'_result.json', 'episode_report.json', 'episode_progress.json',
         'execution_state.json', 'smoke_report.json', 'retime_report.json',
         'report.json', 'smoke.json'}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_index(manifest):
    result = {}
    for group in manifest['groups']:
        for row in group['layouts']:
            key = (group['task'], int(group['eval_seed']), int(row['layout_id']))
            path = row['source_path']
            if key in result and result[key] != path:
                raise ValueError('conflicting native layout source identity')
            result[key] = path
    return result


def inventory(full_manifest, roots):
    index = source_index(full_manifest)
    tasks = {key[0] for key in index}
    task_pattern = '|'.join(re.escape(t) for t in sorted(tasks, key=len, reverse=True))
    episode_re = re.compile(r'(?<![a-zA-Z0-9_])(' + task_pattern + r')_s([0-9]+)_l([0-9]+)(?![0-9])')
    records, unresolved, files, visited = {}, [], {}, set()
    scanned_roots = []

    def add(task, seed, layout, source):
        if task not in tasks or type(seed) is not int or type(layout) is not int:
            unresolved.append({'source': str(source), 'reason': 'incomplete executed layout identity',
                               'task': task if task in tasks else None})
            return
        key = (task, seed, layout)
        records.setdefault(key, set()).add(str(source))

    def path_context(path):
        context = {}
        matches = list(episode_re.finditer(str(path)))
        if matches:
            match = matches[-1]
            context.update(task=match[1], eval_seed=int(match[2]), layout_id=int(match[3]))
        parts = path.parts
        if 'eval_result' in parts:
            for pos, part in enumerate(parts):
                if part in tasks:
                    context.setdefault('task', part)
                    for child in parts[pos + 1:]:
                        match = re.match(r'^([0-9]+)_', child)
                        if match:
                            context.setdefault('eval_seed', int(match[1]))
                            break
                    break
        return context

    def walk_value(value, context, source):
        if isinstance(value, list):
            for child in value:
                walk_value(child, context, source)
            return
        if not isinstance(value, dict):
            return
        context = dict(context)
        task = value.get('task', value.get('task_name'))
        if isinstance(task, str) and task in tasks:
            context['task'] = task
        seed = value.get('eval_seed', value.get('seed'))
        if type(seed) is int:
            context['eval_seed'] = seed
        eid = value.get('episode_id')
        if isinstance(eid, str):
            match = episode_re.fullmatch(eid)
            if match:
                add(match[1], int(match[2]), int(match[3]), source)
                context.update(task=match[1], eval_seed=int(match[2]), layout_id=int(match[3]))
        if type(value.get('layout_id')) is int:
            add(context.get('task'), context.get('eval_seed'), value['layout_id'], source)
        for field in ('completed_layout_ids', 'abandoned_layout_ids', 'started_layout_ids'):
            if field in value:
                for layout in value[field]:
                    add(context.get('task'), context.get('eval_seed'), layout, source)
        for key, child in value.items():
            # Queue/config values are only planned identities, not attempts.
            if key not in {'manifest', 'config', 'configuration', 'candidate_layout_ids',
                           'episode_ids', 'episode_seeds', 'layouts', 'source_groups'}:
                if isinstance(child, (dict, list)):
                    walk_value(child, context, source)

    for root_value in roots:
        root = Path(root_value).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f'inventory root is not a directory: {root}')
        scanned_roots.append(str(root))
        for directory, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                             and not d.startswith('.venv')
                             and not (Path(directory) / d).is_symlink())
            path = Path(directory)
            match = episode_re.fullmatch(path.name)
            if match:
                add(match[1], int(match[2]), int(match[3]), path)
            for name in sorted(names):
                if not (name in NAMES or name in JSONL or name.startswith('_resume_') and name.endswith('.json')
                        or name.endswith('_report.json')):
                    continue
                file = path / name
                if file.is_symlink() or str(file) in visited:
                    continue
                visited.add(str(file))
                context = path_context(file)
                try:
                    before = file.stat()
                    digest = sha_file(file)
                    with file.open() as handle:
                        if name in JSONL:
                            for line_number, line in enumerate(handle, 1):
                                if not line.strip():
                                    continue
                                item = json.loads(line)
                                if item.get('event') != 'suite_initialized':
                                    walk_value(item, context, f'{file}:{line_number}')
                        else:
                            walk_value(json.load(handle), context, file)
                    after = file.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError('evidence changed during inventory; rerun after allocation stops')
                    files[str(file)] = {'sha256': digest, 'bytes': after.st_size}
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    unresolved.append({'source': str(file), 'reason': type(exc).__name__ + ': ' + str(exc)[:180]})
    exposed = []
    for (task, seed, layout), evidence in sorted(records.items()):
        key = (task, seed, layout)
        row = {'task': task, 'eval_seed': seed, 'layout_id': layout, 'evidence': sorted(evidence)}
        if key not in index:
            unresolved.append({'source': sorted(evidence)[0], 'reason': 'executed identity absent from native full manifest',
                               'task': task, 'eval_seed': seed, 'layout_id': layout})
        else:
            try:
                source = Path(index[key])
                raw = source.read_bytes()
                row.update(source_path=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
                           semantic_state_sha256=canonical_sha256(semantic_layout(json.loads(raw))))
            except (OSError, ValueError, TypeError) as exc:
                unresolved.append({'source': index[key], 'reason': type(exc).__name__ + ': ' + str(exc)[:180]})
        exposed.append(row)
    return {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'purpose': 'Exclude every identified started/scored/unscored/smoke/diagnostic physical layout before outcome-independent selection',
            'semantic_hash_method': HASH_METHOD, 'source_full_manifest_canonical_sha256': canonical_sha256(full_manifest),
            'scanned_roots': scanned_roots, 'evidence_files': files, 'exposures': exposed,
            'unresolved': unresolved, 'complete_for_scanned_roots': not unresolved,
            'coverage_note': 'Explicit roots only; caller must attest coverage of every historical experiment root. Asset hashing is identity verification, not a rollout.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-manifest', type=Path, required=True)
    parser.add_argument('--root', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inventory(json.loads(args.full_manifest.read_text()), args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha_file(args.output),
                      'roots': result['scanned_roots'], 'files': len(result['evidence_files']),
                      'exposures': len(result['exposures']), 'unresolved': result['unresolved'],
                      'complete_for_scanned_roots': result['complete_for_scanned_roots']}))


if __name__ == '__main__':
    main()
