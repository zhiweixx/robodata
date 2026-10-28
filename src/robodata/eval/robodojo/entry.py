"""Frozen container entry: native lifecycle with verified online perception."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit
from urllib import request

from robodata.data.provenance import verify_source, verify_imports
from robodata.eval.robodojo.manifest import load_manifest, get_group, verify_sources


def checked_count(native_count, *, mode, override, group):
    if (mode != 'evaluation' or type(native_count) is not int
            or native_count != group['native_target_episodes']
            or str(override) != str(group['target_episodes'])):
        raise ValueError('native count/authorized subset target mismatch')
    return group['target_episodes']


def install_hooks(full_entry, full_episode, group, config, config_hash):
    original_build, original_episode = full_episode.build_api, full_entry.run_full_episode
    state = {'root': None, 'runtime': None}
    def persist():
        if state['root'] is not None and state['runtime'] is not None:
            full_episode.write_json(state['root'] / 'perception_runtime.json', state['runtime'])
    def build(adapter, **kwargs):
        from robodata.perception.perception_bridge import X5GraspCalibration
        path = getattr(adapter, '_progress_path', None)
        if path is None: raise full_entry.FullEvaluationInterrupted('missing runtime evidence path')
        state['root'] = Path(path).parent
        state['runtime'] = {'providers': {}, 'initialization_complete': False,
                            'provider_config_sha256': config_hash,
                            'service_provenance': copy.deepcopy(config['provenance']),
                            'offline_skill_discovery': False}
        persist()
        try:
            api = original_build(adapter, **kwargs, sam3_url=config['sam3_url'],
                graspnet_url=config['graspnet_url'],
                grasp_calibration=X5GraspCalibration.from_mapping(config['grasp_calibration']))
            sam = getattr(api, '_segmentation_provider', None)
            cgn = getattr(api, '_grasp_providers', {}).get('contact-graspnet')
            actual = {'sam3': sam is not None, 'contact_graspnet': cgn is not None}
            health = {'sam3': bool(sam.health()) if sam else False,
                      'contact_graspnet': bool(cgn.health()) if cgn else False}
            if actual != {'sam3': True, 'contact_graspnet': True} or not all(health.values()):
                raise RuntimeError('required actual perception providers not ready')
            state['runtime'].update(providers=actual, provider_health=health,
                provider_classes={k: f'{type(v).__module__}.{type(v).__qualname__}'
                                  for k,v in [('sam3',sam),('contact_graspnet',cgn)]},
                initialization_complete=True)
            persist()
            return api
        except Exception as exc:
            state['runtime']['initialization_error'] = type(exc).__name__; persist()
            raise full_entry.FullEvaluationInterrupted('perception initialization failed') from exc
    def episode(env, **kwargs):
        if (int(env.step_lim) != group['step_limit'] or int(env.eval_num) != group['target_episodes']
                or env.task_name != group['task'] or int(env.eval_seed) != group['eval_seed']
                or int(env.env_seeds[0]) not in group['candidate_layout_ids']):
            raise full_entry.FullEvaluationInterrupted('native state/budget differs from manifest')
        state.update(root=None, runtime=None); returned = None
        try:
            returned = original_episode(env, **kwargs)
            if not state['runtime'] or not state['runtime']['initialization_complete']:
                raise full_entry.FullEvaluationInterrupted('missing provider runtime evidence')
            returned['perception_runtime'] = copy.deepcopy(state['runtime'])
            returned['scope'] = 'ASPIRE API/program executor + custom ARX X5/CuRobo bridge; SAM3 and Contact-GraspNet online providers; no VLA or offline skill discovery'
            returned['harness_profile'] = 'sam3-contact-graspnet-six-fixes'
            return returned
        finally:
            persist()
            if returned is not None and state['root'] is not None:
                full_episode.write_json(state['root'] / 'episode_report.json', returned)
    full_episode.build_api = build
    full_entry.run_full_episode = episode
    full_entry.effective_eval_count = lambda native_count, *, mode, override: checked_count(
        native_count, mode=mode, override=override, group=group)


def main():
    source = Path('/opt/robodata')
    campaign = Path('/campaign')
    sim_environment=json.loads(Path('/opt/robodata-env/sim/installation.json').read_text())
    actual_packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    if actual_packages!=sim_environment.get('packages'):raise ValueError('simulator environment changed after installation')
    inventory = json.loads((campaign/'source_inventory.json').read_text())
    verify_source(source, inventory)
    manifest = load_manifest(campaign/'manifest.json')
    group = get_group(manifest, os.environ['ROBODATA_GROUP_ID'])
    if os.environ.get('EVAL_LAYOUT_IDS') != ','.join(map(str,group['candidate_layout_ids'])):
        raise ValueError('layout candidate queue changed')
    checked_count(group['native_target_episodes'], mode=os.environ.get('ASPIRE_FULL_MODE'),
                  override=os.environ.get('EVAL_NUM'), group=group)
    verify_sources(manifest, '/work/RoboDojo', '/assets', group['group_id'])
    from robodata.eval.robodojo import full_entry, full_episode
    from robodata.execution import aspire_glue, receipt_executor, native_stdout
    from robodata.environments.robodojo import robodojo_adapter, full_adapter
    from robodata.agents import full_relay
    from robodata.perception import perception_bridge, perception_service, cgn_singleton_fix
    import cap_harness.api
    import cap_harness.contracts
    import cap_harness.runtime
    modules = [sys.modules[__name__], full_entry, full_episode, aspire_glue,
        receipt_executor, native_stdout, robodojo_adapter, full_adapter, full_relay,
        perception_bridge, cap_harness.api, cap_harness.contracts, cap_harness.runtime]
    modules += [module for name,module in list(sys.modules.items()) if name.startswith(('robodata.', 'cap_harness.')) and getattr(module,'__file__',None)]
    imported = {}
    for module in modules:
        path = Path(module.__file__).resolve(strict=True)
        if not path.is_relative_to(source): raise ValueError(f'module outside frozen source: {module.__name__}')
        imported[str(path.relative_to(source))] = str(path)
    verify_imports(source, inventory, imported)
    config_path = Path(os.environ['ASPIRE_PERCEPTION_CONFIG'])
    config = json.loads(config_path.read_text())
    for key in ('sam3_url','graspnet_url'):
        url = urlsplit(config[key])
        if (url.scheme != 'http' or url.hostname not in {'127.0.0.1','localhost','::1'}
                or url.port is None or url.username or url.password or url.query or url.fragment):
            raise ValueError('perception URL must be authenticated-by-node-isolation loopback')
        opener=request.build_opener(request.ProxyHandler({}))
        with opener.open(config[key]+'/healthz',timeout=10) as response: health=json.load(response)
        kind='sam3' if key=='sam3_url' else 'graspnet'
        if health != config['provenance']['service_health'][kind]:
            raise ValueError('live provider differs from frozen worker readiness evidence')
        from robodata.perception import perception_service
        expected_weight=perception_service.SAM3_WEIGHT_SHA256 if kind=='sam3' else perception_service.CGN_WEIGHT_SHA256
        if health.get('checkpoint_sha256')!=expected_weight or health.get('parameter_devices')!=['cuda:0']:
            raise ValueError('provider loaded unexpected weights/device')
        if kind=='graspnet':
            from robodata.perception.cgn_singleton_fix import expected_patch_evidence
            patch_hash=hashlib.sha256((source/'src/robodata/perception/cgn_singleton_fix.py').read_bytes()).hexdigest()
            if health.get('estimator_patch')!=expected_patch_evidence(patch_hash):
                raise ValueError('live CGN repair evidence differs')
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    output = Path(os.environ['ASPIRE_OUTPUT_DIR'])
    if output != campaign/'groups'/group['group_id']: raise ValueError('incorrect group output')
    output.mkdir(parents=True,exist_ok=True)
    full_episode.write_json(output/'execution_contract.json', {
        'simulator_environment':sim_environment,'group_id':group['group_id'],'manifest_sha256':hashlib.sha256((campaign/'manifest.json').read_bytes()).hexdigest(),
        'source_tree_sha256':inventory['tree_sha256'],'imported_sources':imported,
        'step_limit':group['step_limit'],'target_episodes':group['target_episodes'],
        'candidate_layout_ids':group['candidate_layout_ids'],'provider_config_sha256':config_hash})
    install_hooks(full_entry,full_episode,group,config,config_hash)
    full_entry.main()

if __name__ == '__main__': main()
