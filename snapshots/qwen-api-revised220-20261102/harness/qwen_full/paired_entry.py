"""Native scored lifecycle with exact paired states and a frozen source profile."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from paired_manifest import arm_campaign, sha_file, validate_manifest


def checked_count(native_count, *, mode, override, group):
    if (mode != 'evaluation' or type(native_count) is not int
            or native_count != group['native_target_episodes']
            or str(override) != str(group['target_episodes'])):
        raise ValueError('paired target must preserve the native count provenance')
    return group['target_episodes']


def load_contract(path, task, seed, arm, profile_root):
    manifest = validate_manifest(json.loads(Path(path).read_text()))
    groups = [g for g in manifest['groups'] if g['task'] == task and g['eval_seed'] == seed]
    if len(groups) != 1:
        raise ValueError('task/seed outside paired manifest')
    expected = Path('/work/aspire_validate') / manifest['profiles'][arm]['relative_root']
    if Path(profile_root).resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError('source profile path differs from arm identity')
    return manifest, groups[0]


def activate_profile(profile):
    """Script-dir precedence must not import mutable common full_* modules."""
    profile = Path(profile).resolve(strict=True)
    selected = [str(profile / 'qwen_full'), str(profile / 'qwen_integration'),
                str(profile / 'ASPIRE/src')]
    common = str(Path(__file__).resolve().parent)
    sys.path[:] = selected + [p for p in sys.path if p not in selected and p != common]
    return selected


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--task_name', required=True)
    p.add_argument('--seed', type=int, required=True)
    args, _ = p.parse_known_args()
    path = Path(os.environ['ASPIRE_PAIRED_MANIFEST'])
    arm = os.environ['ASPIRE_PAIRED_ARM']
    profile = Path(os.environ['ASPIRE_PAIRED_PROFILE'])
    manifest, group = load_contract(path, args.task_name, args.seed, arm, profile)
    if os.environ.get('EVAL_LAYOUT_IDS') != ','.join(map(str, group['candidate_layout_ids'])):
        raise ValueError('paired layout queue changed')
    checked_count(group['native_target_episodes'], mode=os.environ.get('ASPIRE_FULL_MODE'),
                  override=os.environ.get('EVAL_NUM'), group=group)
    output = Path(os.environ['ASPIRE_OUTPUT_DIR'])
    if (output.name != group['group_id']
            or output.parent.parent.name != arm_campaign(manifest, arm)):
        raise ValueError('paired output identity mismatch')
    activate_profile(profile)
    import full_entry
    import full_episode
    import aspire_glue
    import cap_harness.contracts
    selected = {'full_entry': full_entry.__file__, 'full_episode': full_episode.__file__,
                'aspire_glue': aspire_glue.__file__, 'contracts': cap_harness.contracts.__file__}
    for filename in selected.values():
        if not Path(filename).resolve().is_relative_to(profile.resolve()):
            raise ValueError('simulator imported code outside its frozen source profile')
    output.mkdir(parents=True, exist_ok=True)
    full_episode.write_json(output / 'paired_contract.json', {
        'protocol': manifest['protocol'], 'comparison': manifest['campaign'], 'arm': arm,
        'manifest_sha256': sha_file(path), 'profile': manifest['profiles'][arm],
        'group_id': group['group_id'], 'candidate_layout_ids': group['candidate_layout_ids'],
        'step_limit': group['step_limit'], 'target': group['target_episodes'],
        'imported_sources': selected})
    full_entry.effective_eval_count = lambda native_count, *, mode, override: checked_count(
        native_count, mode=mode, override=override, group=group)
    original = full_entry.run_full_episode
    def episode(env, **kwargs):
        if (int(env.step_lim) != group['step_limit'] or int(env.eval_num) != group['target_episodes']
                or env.task_name != group['task'] or int(env.eval_seed) != 0
                or int(env.env_seeds[0]) not in group['candidate_layout_ids']):
            raise full_entry.FullEvaluationInterrupted('paired task/state/action-budget mismatch')
        return original(env, **kwargs)
    full_entry.run_full_episode = episode
    full_entry.main()


if __name__ == '__main__':
    main()
