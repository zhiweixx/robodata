"""Run a manifest-limited number of native valid layouts with the unchanged policy.

Only the scored-layout count and manifest-selected initial-state queue change.
All unique native candidates remain available for invalid-layout replacement;
the native scorer and resume lifecycle are reused without relabeling layout IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from build_subset_manifest import validate_subset_manifest


def load_group_contract(manifest_path, task, seed):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    validate_subset_manifest(manifest)
    matches = [group for group in manifest['groups']
               if group['task'] == task and group['eval_seed'] == seed]
    if len(matches) != 1:
        raise ValueError('task/seed is not an authorized subset group')
    return manifest, matches[0]


def checked_subset_count(native_count, *, mode, override, group):
    if mode != 'evaluation':
        raise ValueError('subset entry runs scored evaluation only')
    if type(native_count) is not int or native_count != group['native_target_episodes']:
        raise ValueError('native count differs from pinned subset manifest')
    if override is None or str(override) != str(group['target_episodes']):
        raise ValueError('EVAL_NUM must equal the authorized subset target')
    target = group['target_episodes']
    if type(target) is not int or not 0 < target <= native_count:
        raise ValueError('invalid subset target')
    return target


def checked_layout_queue(override, group):
    expected = ','.join(str(value) for value in group['candidate_layout_ids'])
    if not expected or override != expected:
        raise ValueError('EVAL_LAYOUT_IDS must equal the manifest unique-state queue')
    return group['candidate_layout_ids']


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--seed', required=True, type=int)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if os.environ.get('ASPIRE_FULL_MODE', 'evaluation') != 'evaluation':
        raise ValueError('subset mode must be evaluation')
    manifest_path = Path(os.environ['ASPIRE_SUBSET_MANIFEST']).resolve(strict=True)
    manifest, group = load_group_contract(manifest_path, args.task_name, args.seed)
    checked_layout_queue(os.environ.get('EVAL_LAYOUT_IDS'), group)
    checked_subset_count(group['native_target_episodes'], mode='evaluation',
                         override=os.environ.get('EVAL_NUM'), group=group)
    output = Path(os.environ['ASPIRE_OUTPUT_DIR'])
    campaign = output.parent.parent.name
    if not campaign.startswith('qwen38_subset20_'):
        raise ValueError('subset cannot write an old full-campaign namespace')
    if output.name != group['group_id']:
        raise ValueError('subset output group identity mismatch')
    from full_episode import write_json
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'subset_contract.json', {
        'protocol': manifest['protocol'], 'manifest_path': str(manifest_path),
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'group_id': group['group_id'], 'native_target_episodes': group['native_target_episodes'],
        'target_episodes': group['target_episodes'], 'step_limit': group['step_limit'],
        'candidate_layout_ids': group['candidate_layout_ids'],
        'invalid_layouts': 'native skip to next manifest-unique candidate; no duplicate replacement',
        'policy_and_action_budgets': 'unchanged from full evaluation'})
    # This patch exists only in this new process. Original full_entry.py and old
    # campaign code are unchanged. Native main applies EVAL_NUM to its outer loop;
    # the hook also sets the same count on the copied EvalEnv configuration.
    import full_entry
    full_entry.effective_eval_count = lambda native_count, *, mode, override: checked_subset_count(
        native_count, mode=mode, override=override, group=group)
    original_episode = full_entry.run_full_episode
    def checked_episode(env, **kwargs):
        if (int(env.step_lim) != group['step_limit'] or int(env.eval_num) != group['target_episodes']
                or env.task_name != group['task'] or int(env.eval_seed) != group['eval_seed']):
            raise full_entry.FullEvaluationInterrupted('native subset identity/count/action-budget mismatch')
        return original_episode(env, **kwargs)
    full_entry.run_full_episode = checked_episode
    full_entry.main()


if __name__ == '__main__':
    main()
