"""One exact paired physical state per native Isaac process.

The shared work-conserving queue assigns independent episodes to nine slots.
Every state has its own native index-zero score/checkpoint and public history.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from harness_revision import validate_revision

PROTOCOL = 'terra_paired721_task_reasoning_v2'
PROTOCOL220 = 'qwen_paired220_task_reasoning_v2'
TOPOLOGY = [0, 0, 1, 1, 2, 2, 3, 3, 3]
REQUIRED_BOOTSTRAP = {'terra_entry.py', 'terra_inside.sh',
    'terra_supervisor.py', 'qwen_driver.py', 'scaling_metrics.py'}

QUEUE_POLICY = {'granularity': 'episode', 'state_output_directory': 'items',
    'initial_paid_admission': 'static first paired episode verified without success threshold, then nine slots',
    'native_eval_nums_per_item': 1, 'simulator_lifecycle': 'fresh process per exact paired state',
    'claim_policy': 'one atomic claim per item per allocation; exclusive campaign owner',
    'order': 'deterministic round-robin across source configurations; preserve each state order',
    'invalid_state_policy': 'unscored missing pair; never substitute',
    'scored_state_policy': 'verify and skip; never replay'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


MODEL = {'served_alias': 'qwen38-27b', 'provider': 'professor_api',
    'base_url': 'http://qwen-api.example.internal:8000/v1', 'api': 'chat_completions',
    'reasoning_effort': 'medium', 'native_context_tokens': 262144,
    'native_max_output_tokens': None, 'checkpoint_verified': False, 'checkpoint_revision': None,
    'dtype': None, 'tensor_parallel_size': None, 'mtp_tokens': None,
    'server_context_verified': False, 'server_configuration': 'unverified professor service; requested native context remains 262144'}

LIMITS = {'action_budget': 'native task config', 'output_tokens': 'no independent output-token cap; native remaining context',
    'context_tokens': 262144, 'context_source': 'pinned Qwen serving context',
    'program_count': None, 'program_wall_seconds': None}
BOOTSTRAP_FILES = ('prepare_terra.py', 'terra_paired_manifest.py', 'terra_entry.py', 'terra_supervisor.py',
    'terra_inside.sh', 'terra_eval.sbatch', 'qwen_driver.py', 'budget_ledger.py', 'budget_broker.py',
    'audit_qwen_results.py', 'scaling_metrics.py', 'qwen721_gate.py', 'harness_revision.py',
    'terra220_pairing.py', 'terra_platform.py', 'ngx_child_reaper.c', 'ngx_child_reaper.so',
    'qwen_network.py', 'provider_transition.py')


USER_MANAGED_FINANCIAL_POLICY = {'mode': 'user_managed', 'spending_control_enabled': False,
    'cap_usd': None, 'cost_monitoring_enabled': False, 'reasoning_effort': 'medium'}
LEGACY_FINANCIAL_POLICY = {'cap_usd': '2000', 'reasoning_effort': 'medium',
    'admission_pause_is_unscored': True, 'durable_ledger': 'budget/ledger.jsonl'}


def user_managed_billing(manifest):
    policy=manifest.get('financial_policy')
    return (policy == USER_MANAGED_FINANCIAL_POLICY
            and policy.get('spending_control_enabled') is False
            and policy.get('cost_monitoring_enabled') is False)


def planned_episodes(manifest):
    return 220 if manifest.get('protocol') == PROTOCOL220 else 721


def bootstrap_relative_root(manifest):
    value = manifest.get('bootstrap_relative_root', 'terra_robodojo')
    pattern = r'qwen_api_revised220_[A-Za-z0-9_]+' if manifest.get('protocol') == PROTOCOL220 else r'terra_robodojo'
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ValueError('bootstrap directory must be a separate frozen revision identity')
    return value


def validate_manifest(manifest):
    validate_revision(manifest)
    if manifest.get('history') != 'All public Qwen Chat Completions messages and images; no rolling truncation. No OpenAI Responses item replay.':
        raise ValueError('revised full Qwen public history policy required')
    name = manifest.get('campaign', '')
    if (manifest.get('protocol') != PROTOCOL220 or not re.fullmatch(r'qwen_paired_[A-Za-z0-9_]+', name)
            or manifest.get('is_official_full_evaluation') is not False
            or manifest.get('queue_policy') != QUEUE_POLICY
            or manifest.get('simulator_gpu_slots') != TOPOLOGY or manifest.get('model') != MODEL
            or not user_managed_billing(manifest)):
        raise ValueError('invalid isolated Terra identity/model/financial policy')
    bootstrap_relative_root(manifest)
    target = planned_episodes(manifest)
    groups = manifest.get('groups', [])
    if (len(groups) != 48 or len({g.get('group_id') for g in groups}) != 48
            or manifest.get('planned_scored_episodes') != target
            or sum(g.get('target_episodes', 0) for g in groups) != target
            or len({g.get('family') for g in groups}) != 38):
        raise ValueError(f'exactly {target} paired states / 48 configurations / 38 families required')
    if canonical_sha256(groups) != manifest.get('groups_binding_sha256'):
        raise ValueError('exact paired group binding changed')
    all_ids = []
    for group in groups:
        task, family, variant = group.get('task'), group.get('family'), group.get('variant')
        if (not isinstance(task, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', task)
                or task != family + ('_random' if variant == 'random' else '')
                or variant not in ('standard', 'random') or group.get('eval_seed') != 0
                or group.get('group_id') != task + '_s0'):
            raise ValueError('paired task identity changed')
        ids, target = group.get('candidate_layout_ids', []), group.get('target_episodes')
        if (type(target) is not int or target <= 0 or len(ids) != target
                or any(type(i) is not int or i < 0 for i in ids) or ids != sorted(set(ids))
                or group.get('candidate_count') != target or group.get('primary_layout_ids') != ids
                or group.get('reserve_layout_ids') != []
                or group.get('runtime_overrides') != {'eval_nums': target, 'eval_layout_ids': ids}
                or group.get('episode_ids') != [f'{task}_s0_l{i}' for i in ids]
                or [r.get('layout_id') for r in group.get('layouts', [])] != ids
                or type(group.get('step_limit')) is not int or group['step_limit'] <= 0
                or group.get('native_options', {}).get('eval_nums') != group.get('native_target_episodes')):
            raise ValueError('paired IDs / original native budgets changed; no fallback states permitted')
        for field in ('layout_source_sha256_by_id', 'layout_canonical_json_sha256_by_id', 'layout_content_sha256_by_id'):
            hashes = group.get(field, {})
            if set(hashes) != set(map(str, ids)) or any(not re.fullmatch('[0-9a-f]{64}', x) for x in hashes.values()):
                raise ValueError('exact paired layout hash mapping invalid')
        if len(set(group['layout_content_sha256_by_id'].values())) != target:
            raise ValueError('duplicate physical state in paired group')
        all_ids.extend(group['episode_ids'])
    if manifest.get('episode_ids') != all_ids or len(set(all_ids)) != planned_episodes(manifest):
        raise ValueError('paired episode identities differ')
    profile = manifest['source_profile']; files = profile['files_sha256']
    if (profile['relative_root'] != 'harness_profiles/' + name
            or canonical_sha256(files) != profile['files_manifest_sha256']
            or set(manifest.get('bootstrap_sha256', {})) != set(BOOTSTRAP_FILES)):
        raise ValueError('frozen source inventory incomplete')
    for relative, digest in {**files, **manifest['bootstrap_sha256']}.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid source inventory entry')
    return manifest


def select_group(manifest, group_id):
    matches = [g for g in manifest['groups'] if g['group_id'] == group_id]
    if len(matches) != 1:
        raise ValueError('unknown or duplicate task/config group')
    return matches[0]


def episode_items(manifest):
    """Derive independent native groups without changing any robot/model budget."""
    batches = []
    for group in manifest['groups']:
        batch = []
        for layout_id, episode_id in zip(group['candidate_layout_ids'], group['episode_ids'], strict=True):
            item = copy.deepcopy(group)
            item.update(group_id=episode_id, episode_id=episode_id,
                source_group_id=group['group_id'], target_episodes=1, candidate_count=1,
                candidate_layout_ids=[layout_id], primary_layout_ids=[layout_id],
                reserve_layout_ids=[], episode_ids=[episode_id],
                runtime_overrides={'eval_nums': 1, 'eval_layout_ids': [layout_id]})
            item['layouts'] = [r for r in item['layouts'] if r['layout_id'] == layout_id]
            item['states'] = [r for r in item.get('states', []) if r['layout_id'] == layout_id]
            for field in ('layout_source_sha256_by_id', 'layout_canonical_json_sha256_by_id',
                          'layout_content_sha256_by_id'):
                item[field] = {str(layout_id): group[field][str(layout_id)]}
            batch.append(item)
        batches.append(batch)
    result = [batch[index] for index in range(max(map(len,batches),default=0))
              for batch in batches if index < len(batch)]
    if len(result) != len({item['episode_id'] for item in result}):
        raise ValueError('duplicate episode queue identity')
    return result


def select_episode_item(manifest, episode_id):
    matches = [item for item in episode_items(manifest) if item['episode_id'] == episode_id]
    if len(matches) != 1:
        raise ValueError('unknown or duplicate paired episode queue item')
    return matches[0]


def verify_sources(root, manifest, code):
    root, code = Path(root).resolve(strict=True), Path(code).resolve(strict=True)
    if code != (root / bootstrap_relative_root(manifest)).resolve(strict=True):
        raise ValueError('runtime bootstrap path differs from the isolated frozen manifest')
    profile = (root / manifest['source_profile']['relative_root']).resolve(strict=True)
    if not profile.is_relative_to(root / 'harness_profiles'):
        raise ValueError('source profile escaped isolated harness directory')
    actual = {}
    for path in sorted(profile.rglob('*')):
        rel = path.relative_to(profile)
        if any(part in ('.git', '__pycache__') for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError('frozen source profile contains symlink')
        if path.is_file():
            actual[str(rel)] = sha_file(path)
    if actual != manifest['source_profile']['files_sha256']:
        raise ValueError('frozen copied source changed')
    for name, digest in manifest['bootstrap_sha256'].items():
        path = code / name
        if path.is_symlink() or sha_file(path) != digest:
            raise ValueError('bootstrap source changed: ' + name)
    return profile


def activate_profile(profile):
    for relative in ('ASPIRE/src', 'qwen_integration', 'qwen_full'):
        sys.path.insert(0, str(Path(profile) / relative))


def checked_count(native_count, *, mode, override, group):
    if (mode != 'evaluation' or type(native_count) is not int
            or native_count != group['native_target_episodes']
            or str(override) != str(group['target_episodes'])):
        raise ValueError('group count must preserve native count provenance')
    return group['target_episodes']


class GroupCompletion:
    """Observe native exhaustion without replacing its selection/scoring logic."""
    def __init__(self, group, output, write_json, interrupt_type):
        self.group, self.output = group, Path(output)
        self.write_json, self.interrupt_type = write_json, interrupt_type
        self.env = None
        self.native_queue_exhausted = False
        self.selected_this_process = []
        self.close_seen = False

    def attach(self, env):
        if self.env is not None:
            raise self.interrupt_type('native environment recreated inside one group process')
        self.env = env
        original = env.seed_manager.get_seeds
        def get_seeds(*args, **kwargs):
            values = original(*args, **kwargs)
            self.native_queue_exhausted = values is None
            if values is not None:
                scored = {int(d['layout_id']) for d in env.eval_result.get('details', {}).values()}
                for value in values:
                    if (type(value) is not int or value not in self.group['candidate_layout_ids']
                            or value in scored or value in self.selected_this_process):
                        raise self.interrupt_type('native queue returned duplicate, scored, or unauthorized state')
                    self.selected_this_process.append(value)
            return values
        env.seed_manager.get_seeds = get_seeds
        return env

    def publish(self, *, before_native_close=False, error=None):
        self.close_seen = self.close_seen or before_native_close
        env = self.env
        scored = int(env.success_nums + env.fail_nums) if env is not None else 0
        exhausted = (self.close_seen and error is None and self.native_queue_exhausted
                     and scored < self.group['target_episodes'])
        result = {'group_id': self.group['group_id'], 'target_episodes': self.group['target_episodes'],
            'scored': scored, 'native_candidate_exhausted': exhausted,
            'observed_native_queue_exhausted': self.native_queue_exhausted,
            'selected_layout_ids_this_process': self.selected_this_process,
            'native_close_seen': self.close_seen, 'error': error,
            'native_result_path': str(Path(env.save_dir).resolve() / '_result.json') if env is not None else None,
            'updated_utc': utc_now(), 'no_automatic_replay': True}
        self.write_json(self.output / 'terra_group_completion.json', result)
        return result


def install_native_observers(native, observer, interrupt_type):
    original_create, original_close = native.create_eval_env, native.simulation_app.close
    def create(*args, **kwargs):
        return observer.attach(original_create(*args, **kwargs))
    def close(*args, **kwargs):
        observer.publish(before_native_close=True)
        return original_close(*args, **kwargs)
    def no_hidden_restart(env, *args, **kwargs):
        env.persist_resume_manifest()
        raise interrupt_type('native fatal interruption; checkpoint retained for explicit supervisor resume')
    native.create_eval_env, native.simulation_app.close = create, close
    native._restart_or_exit = no_hidden_restart
    native._exit_for_shell_restart = no_hidden_restart


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args, _ = parser.parse_known_args()
    manifest_path = Path(os.environ['ASPIRE_TERRA_MANIFEST'])
    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    group = select_episode_item(manifest, os.environ['ASPIRE_TERRA_GROUP'])
    worker = os.environ['ASPIRE_TERRA_WORKER']
    if worker not in [f'w{i:02d}' for i in range(9)]:
        raise ValueError('unknown worker slot')
    root = Path('/work/aspire_validate')
    output = root / 'full_results' / manifest['campaign'] / 'items' / group['group_id']
    if (args.task_name != group['task'] or args.seed != group['eval_seed']
            or os.environ.get('EVAL_LAYOUT_IDS') != ','.join(map(str, group['candidate_layout_ids']))
            or Path(os.environ['ASPIRE_OUTPUT_DIR']).resolve() != output.resolve()
            or os.environ.get('ROBODOJO_RUN_ID') != manifest['campaign'] + '_' + group['group_id']
            or os.environ.get('ASPIRE_MODEL_NAME') != manifest['model']['served_alias']):
        raise ValueError('group launch differs from frozen identity')
    checked_count(group['native_target_episodes'], mode=os.environ.get('ASPIRE_FULL_MODE'),
        override=os.environ.get('EVAL_NUM'), group=group)
    profile = verify_sources(root, manifest, Path(__file__).parent)
    activate_profile(profile)
    import full_entry, full_episode, perception_entry, perception_bridge, aspire_glue
    import full_adapter, full_model_driver, model_driver, task_prompt_contract
    import cap_harness.contracts
    modules = (full_entry, full_episode, perception_entry, perception_bridge, aspire_glue, cap_harness.contracts,
        full_adapter, full_model_driver, model_driver, task_prompt_contract)
    selected = {m.__name__: str(Path(m.__file__).resolve()) for m in modules}
    if any(not Path(path).is_relative_to(profile) for path in selected.values()):
        raise ValueError('harness imported outside frozen profile')
    config_path = Path(os.environ['ASPIRE_PERCEPTION_CONFIG'])
    config = perception_entry.read_provider_config(config_path)
    perception_entry.verify_provider_config(config, manifest)
    output.mkdir(parents=True, exist_ok=True)
    full_episode.write_json(output / 'terra_contract.json', {'protocol': manifest['protocol'],
        'campaign': manifest['campaign'], 'group': group, 'worker': worker,
        'episode_id': group['episode_id'], 'source_group_id': group['source_group_id'],
        'manifest_sha256': sha_file(manifest_path), 'imported_sources': selected,
        'source_profile': manifest['source_profile'], 'harness_revision': manifest['harness_revision'], 'remote_model': manifest['model'],
        'fresh_campaign': True, 'is_official_full_evaluation': False, 'offline_skill_discovery': False,
        'native_invalid_candidates': 'exact Qwen-scored states only; abandoned initialization leaves an unscored missing pair; no replacement',
        'lifecycle': 'fresh native process per exact paired physical state'})
    original_write = full_episode.write_json
    def honest_report(path, value):
        if isinstance(value, dict) and 'episode_id' in value and 'limits' in value:
            value['remote_model'] = manifest['model']
            value['limits']['output_tokens'] = LIMITS['output_tokens']
            value['limits']['context_tokens'] = LIMITS['context_tokens']
            value['limits']['context_source'] = LIMITS['context_source']
            value['is_official_full_evaluation'] = False
            value['fresh_campaign'] = True
        return original_write(path, value)
    full_episode.write_json = honest_report
    full_entry.effective_eval_count = lambda native_count, *, mode, override: checked_count(
        native_count, mode=mode, override=override, group=group)
    perception_entry.install_runtime_hooks(full_entry, full_episode, arm='perception', group=group,
        config=config, config_sha256=sha_file(config_path),
        calibration_factory=perception_bridge.X5GraspCalibration.from_mapping,
        services_resident_for_both_arms=False,
        report_scope=f'Exact-state paired {planned_episodes(manifest)}-episode RoboDojo evaluation with frozen ASPIRE API/executor, '
        'X5/CuRobo, SAM3 and Contact-GraspNet, self-hosted Qwen3.8-27B; no offline skill learning.')
    original_episode = full_entry.run_full_episode
    def timed_episode(env, **kwargs):
        # Verify the pinned narrow child collector after native reset and before API admission.
        import carb.settings
        from terra_platform import verify_kit_auxiliary_policy
        auxiliary = verify_kit_auxiliary_policy(carb.settings.get_settings(), os.environ)
        allocation = os.environ.get('ASPIRE_TERRA_ALLOCATION', '')
        if not allocation.isdigit():
            raise RuntimeError('Isaac optional helper receipt lacks explicit allocation identity')
        auxiliary.update(campaign=manifest['campaign'], episode_id=group['episode_id'],
            allocation=allocation, manifest_sha256=sha_file(manifest_path))
        full_episode.write_json(output / 'kit_auxiliary_policy.json', auxiliary)
        started = utc_now()
        full_episode.write_json(output / 'terra_episode_timing.json', {
            'episode_id': group['episode_id'], 'source_group_id': group['source_group_id'],
            'episode_started_utc': started, 'episode_callback_finished_utc': None})
        value = None
        try:
            value = original_episode(env, **kwargs)
            return value
        finally:
            timing = {'episode_started_utc': started, 'episode_callback_finished_utc': utc_now(),
                'worker_id': worker, 'group_id': group['group_id'], 'episode_id': group['episode_id'],
                'source_group_id': group['source_group_id'], 'stage': 'qwen',
                'native_campaign': manifest['campaign']}
            full_episode.write_json(output / 'terra_episode_timing.json', timing)
            if value is not None:
                value['terra_timing'] = timing
                value['scaling_timing'] = timing
                full_episode.write_json(Path(value['artifact_dir']) / 'episode_report.json', value)
            else:
                episode_id = f'{env.task_name}_s{int(env.eval_seed)}_l{int(env.env_seeds[0])}'
                latest = output / 'episodes' / episode_id / 'latest_attempt.json'
                if latest.exists():
                    path = Path(json.loads(latest.read_text())['report'])
                    prior = json.loads(path.read_text())
                    prior['terra_timing'] = timing
                    prior['scaling_timing'] = timing
                    full_episode.write_json(path, prior)
    full_entry.run_full_episode = timed_episode
    from src.eval_client import main as native
    observer = GroupCompletion(group, output, full_episode.write_json, full_entry.FullEvaluationInterrupted)
    install_native_observers(native, observer, full_entry.FullEvaluationInterrupted)
    error = None
    try:
        full_entry.main()
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        # A normal native exhaustion may exit nonzero through full_entry's
        # completeness check. Its pre-close receipt must remain authoritative.
        if not observer.close_seen:
            observer.publish(error=error)
        raise
    finally:
        if not observer.close_seen:
            observer.publish(error=error)


if __name__ == '__main__':
    main()
