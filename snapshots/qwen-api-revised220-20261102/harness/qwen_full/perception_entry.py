"""Native one-state lifecycle with explicit, audited online perception providers."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

from perception_manifest import arm_campaign, sha_file, validate_manifest, verify_bootstrap
import perception_manifest


def checked_count(native_count, *, mode, override, group):
    if (mode != 'evaluation' or type(native_count) is not int
            or native_count != group['native_target_episodes']
            or str(override) != '1' or group['target_episodes'] != 1):
        raise ValueError('perception target must preserve native count provenance')
    return 1


def load_contract(path, task, seed, arm, unit_id, profile_root):
    manifest = validate_manifest(json.loads(Path(path).read_text()))
    units = [u for u in manifest['units'] if u['unit_id'] == unit_id]
    if len(units) != 1 or arm not in manifest['arms']:
        raise ValueError('unknown perception arm/state')
    unit = units[0]
    group = unit['group']
    if group['task'] != task or group['eval_seed'] != seed:
        raise ValueError('task/seed outside perception unit')
    expected = Path('/work/aspire_validate') / manifest['profiles'][arm]['relative_root']
    if Path(profile_root).resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError('source profile differs from arm identity')
    return manifest, unit


def activate_profile(profile):
    profile = Path(profile).resolve(strict=True)
    selected = [str(profile / 'qwen_full'), str(profile / 'qwen_integration'),
                str(profile / 'ASPIRE/src')]
    common = str(Path(__file__).resolve().parent)
    sys.path[:] = selected + [p for p in sys.path if p not in selected and p != common]


def read_provider_config(path):
    config = json.loads(Path(path).read_text())
    required = {'sam3_url', 'graspnet_url', 'grasp_calibration', 'provenance'}
    if not isinstance(config, dict) or not required <= config.keys():
        raise ValueError('incomplete perception service configuration')
    for name in ('sam3_url', 'graspnet_url'):
        parsed = urlparse(config[name])
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ('', '/') or parsed.port is None):
            raise ValueError('perception services must use explicit node-local HTTP endpoints')
    if not isinstance(config['grasp_calibration'], dict) or not isinstance(config['provenance'], dict):
        raise ValueError('perception calibration and source provenance must be objects')
    return config


def class_identity(value):
    if value is None:
        return None
    cls = type(value)
    return f'{cls.__module__}.{cls.__qualname__}'


def verify_provider_config(config, manifest):
    artifacts = manifest['perception_artifacts']
    if (config['grasp_calibration'] != manifest['grasp_calibration']
            or any(config['provenance'].get(key) != artifacts[key] for key in (
                'runtime_manifest', 'runtime_manifest_sha256', 'service_runner_sha256'))
            or config['provenance'].get('offline_skill_discovery') is not False):
        raise ValueError('perception configuration differs from frozen artifacts/calibration')


def install_runtime_hooks(full_entry, full_episode, *, arm, group, config,
                          config_sha256, calibration_factory=None,
                          services_resident_for_both_arms=True, report_scope=None):
    """Only provider construction changes; original episode and native commit stay intact.

    A fresh runtime record is written for each episode's current attempt before
    its first model call, and is also
    retained if initialization or a later inference request is interrupted.
    """
    if arm not in ('baseline', 'perception'):
        raise ValueError('unknown perception arm')
    enabled = arm == 'perception'
    original_build = full_episode.build_api
    original_episode = full_entry.run_full_episode
    state = {'runtime': None, 'attempt_root': None}

    def save_runtime():
        if state['attempt_root'] is not None and state['runtime'] is not None:
            full_episode.write_json(state['attempt_root'] / 'perception_runtime.json', state['runtime'])

    def build_api(adapter, **kwargs):
        progress_path = getattr(adapter, '_progress_path', None)
        if progress_path is None:
            raise full_entry.FullEvaluationInterrupted('perception attempt evidence path is unavailable')
        state['attempt_root'] = Path(progress_path).parent
        runtime = {'arm': arm, 'providers': {'sam3': None, 'contact_graspnet': None},
                   'offline_skill_discovery': False, 'provider_config_sha256': config_sha256,
                   'service_provenance': copy.deepcopy(config['provenance']),
                   'services_resident_for_both_arms': services_resident_for_both_arms,
                   'initialization_complete': False}
        state['runtime'] = runtime
        save_runtime()
        try:
            if enabled:
                if calibration_factory is None:
                    raise ValueError('treatment requires explicit X5 calibration factory')
                calibration = calibration_factory(config['grasp_calibration'])
                api = original_build(adapter, **kwargs, sam3_url=config['sam3_url'],
                                     graspnet_url=config['graspnet_url'], grasp_calibration=calibration)
            else:
                # No URL or calibration is passed into the unchanged baseline.
                api = original_build(adapter, **kwargs)
            segmentation = getattr(api, '_segmentation_provider', None)
            grasp_providers = getattr(api, '_grasp_providers', {})
            grasp = grasp_providers.get('contact-graspnet')
            actual = {'sam3': segmentation is not None, 'contact_graspnet': grasp is not None}
            runtime.update(providers=actual, api_class=class_identity(api),
                           provider_classes={'sam3': class_identity(segmentation),
                                             'contact_graspnet': class_identity(grasp)},
                           provider_metadata=copy.deepcopy(getattr(api, '_robodojo_perception', {})))
            if actual != {'sam3': enabled, 'contact_graspnet': enabled}:
                raise ValueError('actual API providers differ from the assigned arm')
            if enabled:
                runtime['provider_health'] = {'sam3': bool(segmentation.health()),
                                              'contact_graspnet': bool(grasp.health())}
                runtime['grasp_calibration'] = calibration.as_dict()
                if not all(runtime['provider_health'].values()):
                    raise ValueError('online perception provider health check failed')
            else:
                runtime['provider_health'] = {}
                runtime['grasp_calibration'] = None
            runtime['initialization_complete'] = True
            save_runtime()
            return api
        except Exception as exc:
            runtime['initialization_error_type'] = type(exc).__name__
            save_runtime()
            raise full_entry.FullEvaluationInterrupted(
                f'perception provider initialization failed ({type(exc).__name__})') from exc

    def episode(env, **kwargs):
        if (int(env.step_lim) != group['step_limit'] or int(env.eval_num) != group['target_episodes']
                or env.task_name != group['task'] or int(env.eval_seed) != group['eval_seed']
                or int(env.env_seeds[0]) not in group['candidate_layout_ids']):
            raise full_entry.FullEvaluationInterrupted('perception task/state/action-budget mismatch')
        state.update(runtime=None, attempt_root=None)
        returned = None
        try:
            returned = original_episode(env, **kwargs)
            if state['runtime'] is None or not state['runtime']['initialization_complete']:
                raise full_entry.FullEvaluationInterrupted('missing actual perception runtime evidence')
            returned['perception_runtime'] = copy.deepcopy(state['runtime'])
            if report_scope is not None:
                returned['scope'] = report_scope
            return returned
        finally:
            save_runtime()
            if state['attempt_root'] is not None and state['runtime'] is not None:
                path = state['attempt_root'] / 'episode_report.json'
                if returned is not None:
                    full_episode.write_json(path, returned)
                elif path.exists():
                    report = json.loads(path.read_text())
                    report['perception_runtime'] = copy.deepcopy(state['runtime'])
                    if report_scope is not None:
                        report['scope'] = report_scope
                    full_episode.write_json(path, report)

    full_episode.build_api = build_api
    full_entry.run_full_episode = episode


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args, _ = parser.parse_known_args()
    path = Path(os.environ['ASPIRE_PERCEPTION_MANIFEST'])
    arm = os.environ['ASPIRE_PERCEPTION_ARM']
    unit_id = os.environ['ASPIRE_PERCEPTION_UNIT']
    profile = Path(os.environ['ASPIRE_PERCEPTION_PROFILE'])
    config_path = Path(os.environ['ASPIRE_PERCEPTION_CONFIG'])
    config = read_provider_config(config_path)
    manifest, unit = load_contract(path, args.task_name, args.seed, arm, unit_id, profile)
    bootstrap_evidence = verify_bootstrap(Path('/work/aspire_validate'), manifest, {
        'entry': __file__, 'inside': os.environ['ASPIRE_BOOTSTRAP_INSIDE'],
        'manifest': perception_manifest.__file__})
    verify_provider_config(config, manifest)
    group = unit['group']
    if os.environ.get('EVAL_LAYOUT_IDS') != ','.join(map(str, group['candidate_layout_ids'])):
        raise ValueError('perception layout queue changed')
    checked_count(group['native_target_episodes'], mode=os.environ.get('ASPIRE_FULL_MODE'),
                  override=os.environ.get('EVAL_NUM'), group=group)
    output = Path(os.environ['ASPIRE_OUTPUT_DIR'])
    if (output.name != group['group_id']
            or output.parent.parent.name != arm_campaign(manifest, arm, unit_id)):
        raise ValueError('perception output identity mismatch')
    activate_profile(profile)
    import full_entry
    import full_episode
    import aspire_glue
    import cap_harness.contracts
    selected = {'full_entry': full_entry.__file__, 'full_episode': full_episode.__file__,
                'aspire_glue': aspire_glue.__file__, 'contracts': cap_harness.contracts.__file__}
    calibration_factory = None
    if arm == 'perception':
        import perception_bridge
        selected['perception_bridge'] = perception_bridge.__file__
        calibration_factory = perception_bridge.X5GraspCalibration.from_mapping
    for filename in selected.values():
        if not Path(filename).resolve().is_relative_to(profile.resolve()):
            raise ValueError('simulator imported code outside frozen source profile')
    output.mkdir(parents=True, exist_ok=True)
    full_episode.write_json(output / 'perception_contract.json', {
        'protocol': manifest['protocol'], 'comparison': manifest['campaign'], 'arm': arm,
        'unit_id': unit_id, 'manifest_sha256': sha_file(path), 'profile': manifest['profiles'][arm],
        'group_id': group['group_id'], 'candidate_layout_ids': group['candidate_layout_ids'],
        'step_limit': group['step_limit'], 'target': 1,
        'providers': {'sam3': arm == 'perception', 'contact_graspnet': arm == 'perception'},
        'provider_config_sha256': sha_file(config_path), 'imported_sources': selected,
        'bootstrap_verification': bootstrap_evidence})
    full_entry.effective_eval_count = lambda native_count, *, mode, override: checked_count(
        native_count, mode=mode, override=override, group=group)
    install_runtime_hooks(full_entry, full_episode, arm=arm, group=group, config=config,
                          config_sha256=sha_file(config_path), calibration_factory=calibration_factory)
    full_entry.main()


if __name__ == '__main__':
    main()
