#!/usr/bin/env python3
"""CPU-only freeze of a fresh Terra campaign from verified Qwen721 or old Terra220 evidence.

This creates a new campaign/profile, never edits the source Qwen campaign and
never submits a job or calls an API. All source input artifacts are retained.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from harness_revision import build_revised_profile, copy_revised_profile, validate_revision

from terra_paired_manifest import build_from_files, canonical_sha256, read_json
import terra220_pairing
from terra_entry import BOOTSTRAP_FILES, MODEL, PROTOCOL, PROTOCOL220, TOPOLOGY, QUEUE_POLICY, USER_MANAGED_FINANCIAL_POLICY, sha_file, validate_manifest


def verify_preflight(value, bootstrap):
    if (value.get('ready_for_deployment') is not True
            or value.get('user_managed_billing') is not True
            or value.get('spending_control_enabled') is not False
            or value.get('cost_monitoring_enabled') is not False):
        raise ValueError('fresh CPU preparation must explicitly retain user-managed billing without spending controls')
    if value.get('bootstrap_sha256') != bootstrap:
        raise ValueError('CPU preflight belongs to different runtime source bytes')


def groups_from_pairing(pairing):
    result = []
    for paired in pairing['groups']:
        g = copy.deepcopy(paired)
        if pairing.get('artifact_type') == 'terra_revised_exact_state_pairing':
            for state in g['states']:
                for name in ('terra_success', 'terra_score', 'qwen_success',
                             'terra_native_result', 'terra_episode_report', 'terra_videos'):
                    state.pop(name, None)
        g['native_target_episodes'] = g['native_options']['eval_nums']
        g['layouts'] = [copy.deepcopy(s['source_layout']) for s in g['states']]
        for key, source in (('layout_source_sha256_by_id', 'source_sha256'),
                            ('layout_canonical_json_sha256_by_id', 'canonical_json_sha256'),
                            ('layout_content_sha256_by_id', 'semantic_state_sha256')):
            g[key] = {str(s['layout_id']): s[source] for s in g['states']}
        result.append(g)
    return result


def build_manifest(pairing, qwen, *, campaign, bootstrap, pairing_file_sha256, preflight_sha256, revised_profile, harness_revision, bootstrap_relative_root='qwen_api_revised220_20261102',
                   native_verification_sha256=None):
    if not re.fullmatch(r'qwen_paired_[A-Za-z0-9_]+', campaign):
        raise ValueError('new isolated campaign must begin qwen_paired_')
    is220 = pairing.get('artifact_type') == 'terra_revised_exact_state_pairing'
    groups = groups_from_pairing(pairing)
    profile = copy.deepcopy(revised_profile)
    profile['relative_root'] = 'harness_profiles/' + campaign
    value = {'schema_version': 1, 'campaign': campaign, 'protocol': PROTOCOL220 if is220 else PROTOCOL,
        'bootstrap_relative_root': bootstrap_relative_root,
        'model': copy.deepcopy(MODEL), 'planned_scored_episodes': 220 if is220 else 721,
        'queue_policy': copy.deepcopy(QUEUE_POLICY),
        'source_subset_gate': copy.deepcopy(pairing['source_subset_gate']),
        'group_count': pairing['group_count'], 'family_count': pairing['family_count'], 'groups': groups,
        'groups_binding_sha256': canonical_sha256(groups), 'episode_ids': list(pairing['episode_ids']),
        'episode_seeds': {eid: 0 for eid in pairing['episode_ids']},
        'is_official_full_evaluation': False, 'offline_skill_discovery': False,
        'source_profile': profile, 'harness_revision': copy.deepcopy(harness_revision), 'bootstrap_sha256': dict(bootstrap),
        'source': copy.deepcopy(qwen['source']), 'perception_artifacts': copy.deepcopy(qwen['perception_artifacts']),
        'grasp_calibration': copy.deepcopy(qwen['grasp_calibration']),
        'providers': copy.deepcopy(qwen.get('providers')),
        'simulator_gpu_slots': list(TOPOLOGY),
        'hardware': {'nodes': 1, 'gpus': 4, 'gpu_type': 'L40S', 'cpus': 32, 'memory_gib': 240,
                     'simulator_workers': 9, 'shared_perception_gpu_slot': 3},
        'history': 'All public Qwen Chat Completions messages and images; no rolling truncation. No OpenAI Responses item replay.',
        'selection_policy': copy.deepcopy(pairing['policy']),
        'pairing_file': 'paired_states.json', 'pairing_file_sha256': pairing_file_sha256,
        'pairing_sha256': pairing['pairing_sha256'],
        'source_qwen': copy.deepcopy(pairing['source_qwen'] if is220 else pairing['source']),
        'deferred_group_targets': copy.deepcopy(pairing['deferred_group_targets']),
        'user_held_group_shortfalls': copy.deepcopy(pairing['user_held_group_shortfalls']),
        'deployment_preflight_file': 'source_evidence/deployment_preflight.json',
        'deployment_preflight_sha256': preflight_sha256,
        'financial_policy': copy.deepcopy(USER_MANAGED_FINANCIAL_POLICY),
        'limits': {'action_budget': 'native task config', 'model_turns': None,
                   'client_output_tokens': None, 'episode_wall_seconds': None,
                   'native_context_tokens': 262144, 'native_max_output_tokens': None},
        'sampling_difference': 'Pinned Qwen sampling and thinking settings are preserved; provider is self-hosted vLLM rather than the previous unverified remote alias.',
        'created_utc': datetime.now(timezone.utc).isoformat()}
    if is220:
        value.update(source_terra=copy.deepcopy(pairing['source']),
            source_terra_unscored_episodes_outside_scope=501, original_plan_episodes=807,
            source_native_verification_file='source_evidence/source_native_verification.json',
            source_native_verification_sha256=native_verification_sha256,
            baseline_comparison={'campaign':terra220_pairing.SOURCE_CAMPAIGN,
                'model':'gpt-6.1-terra','reasoning_effort':'medium','native_scored_episodes':220,
                'same_exact_physical_states':True,'old_scores_reused_as_new_results':False})
    return validate_manifest(value)


def source_files(directory):
    files = {}
    for path in sorted(Path(directory).rglob('*')):
        rel = path.relative_to(directory)
        if any(p in ('.git', '__pycache__') for p in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError('source profile must not contain symlinks')
        if path.is_file():
            files[str(rel)] = sha_file(path)
    return files


def verify_pairing_artifacts(campaign_dir, manifest):
    """Rebuild exact states from immutable terminal evidence, never outcomes."""
    c = Path(campaign_dir)
    pairing_path = c / 'paired_states.json'
    if sha_file(pairing_path) != manifest['pairing_file_sha256']:
        raise ValueError('paired-state artifact bytes changed')
    pairing, _ = read_json(pairing_path)
    is220 = manifest.get('protocol') == PROTOCOL220
    if is220:
        rebuilt = terra220_pairing.build_from_files(c/'source_terra/manifest.json',
            c/'source_terra/full_audit.json', c/'source_terra/paired_states.json',
            terminal_receipt_path=c/'source_terra/terminal_receipt.json')
        source_manifest, _ = read_json(c/'source_terra/manifest.json')
        path_fields = ('manifest_path','audit_path','pairing_path','terminal_receipt_path')
    else:
        rebuilt = build_from_files(c/'source_qwen/manifest.json', c/'source_qwen/full_audit.json',
            terminal_receipt_path=c/'source_qwen/terminal_receipt.json')
        source_manifest, _ = read_json(c/'source_qwen/manifest.json')
        path_fields = ('manifest_path','audit_path','terminal_receipt_path')
    for key in path_fields:
        rebuilt['source'][key] = pairing['source'][key]
    if (rebuilt != pairing or groups_from_pairing(pairing) != manifest['groups']
            or manifest.get('source_subset_gate') != pairing['source_subset_gate']
            or manifest.get('pairing_sha256') != pairing['pairing_sha256']):
        raise ValueError('paired states or native budgets disagree with final source evidence')
    validate_revision(manifest, pairing['source_profile'])
    if is220:
        if (manifest.get('source_terra') != pairing['source']
                or manifest.get('source_qwen') != pairing['source_qwen']
                or manifest.get('selection_policy') != pairing['policy']
                or manifest.get('source_terra_unscored_episodes_outside_scope') != 501
                or manifest.get('original_plan_episodes') != 807):
            raise ValueError('Terra220 baseline provenance or authorized scope changed')
        for key in ('source','perception_artifacts','grasp_calibration','providers',
                    'deferred_group_targets','user_held_group_shortfalls'):
            if manifest.get(key) != source_manifest.get(key):
                raise ValueError('source/provider/native configuration changed: ' + key)
        native_path = c / manifest['source_native_verification_file']
        native, digest = read_json(native_path)
        if (digest != manifest.get('source_native_verification_sha256')
                or native.get('verified') is not True
                or native.get('episode_ids') != pairing['episode_ids']
                or native.get('native_results') != 220 or native.get('committed_reports') != 220
                or native.get('nonempty_camera_videos') != 660
                or native.get('source_audit_sha256') != pairing['source']['audit_sha256']):
            raise ValueError('original220 native artifact verification is incomplete')
    preflight_path = c / manifest['deployment_preflight_file']
    if sha_file(preflight_path) != manifest['deployment_preflight_sha256']:
        raise ValueError('deployment preflight bytes changed')
    verify_preflight(read_json(preflight_path)[0], manifest['bootstrap_sha256'])
    return pairing


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--campaign', required=True)
    p.add_argument('--bootstrap-dir', default='qwen_api_revised220_20261102', help='Isolated basename of this deployed bootstrap directory.')
    p.add_argument('--harness-source', type=Path, required=True, help='Revised shared harness source; historical campaign profiles are never changed.')
    p.add_argument('--source-scope', choices=('terra220',), default='terra220')
    p.add_argument('--qwen-manifest', type=Path)
    p.add_argument('--qwen-full-audit', type=Path)
    p.add_argument('--qwen-terminal-receipt', type=Path)
    p.add_argument('--terra-manifest', type=Path)
    p.add_argument('--terra-full-audit', type=Path)
    p.add_argument('--terra-paired-states', type=Path)
    p.add_argument('--terra-terminal-receipt', type=Path)
    p.add_argument('--deployment-preflight', type=Path, required=True)
    args = p.parse_args(argv)
    root = args.root.resolve(strict=True); code = Path(__file__).resolve().parent
    if (not re.fullmatch(r'qwen_api_revised220_[A-Za-z0-9_]+', args.bootstrap_dir)
            or code != (root / args.bootstrap_dir).resolve()
            or code.is_symlink()):
        raise ValueError('run the exact isolated deployed preparer under ROOT/--bootstrap-dir')
    if not re.fullmatch(r'qwen_paired_[A-Za-z0-9_]+', args.campaign):
        raise ValueError('invalid new campaign name')
    out = root/'full_results'/args.campaign; profile = root/'harness_profiles'/args.campaign
    if out.exists() or profile.exists():
        raise ValueError('refusing to overwrite a campaign/profile; resume belongs to the runtime')
    is220 = args.source_scope == 'terra220'
    qwen_inputs = (args.qwen_manifest,args.qwen_full_audit,args.qwen_terminal_receipt)
    terra_inputs = (args.terra_manifest,args.terra_full_audit,args.terra_paired_states,args.terra_terminal_receipt)
    if is220:
        if not all(terra_inputs) or any(qwen_inputs):
            raise ValueError('terra220 requires all four --terra-* inputs and no --qwen-* inputs')
        pairing = terra220_pairing.build_from_files(*terra_inputs[:3], terminal_receipt_path=terra_inputs[3])
        source_manifest, _ = read_json(args.terra_manifest)
        original_verification = terra220_pairing.verify_native_artifacts(pairing)
        source_folder = 'source_terra'
        sources = dict(zip(('manifest.json','full_audit.json','paired_states.json','terminal_receipt.json'),terra_inputs))
    else:
        if not all(qwen_inputs) or any(terra_inputs):
            raise ValueError('qwen721 requires all three --qwen-* inputs and no --terra-* inputs')
        pairing = build_from_files(*qwen_inputs[:2], terminal_receipt_path=qwen_inputs[2])
        source_manifest, _ = read_json(args.qwen_manifest)
        original_verification = None
        source_folder = 'source_qwen'
        sources = dict(zip(('manifest.json','full_audit.json','terminal_receipt.json'),qwen_inputs))
    bootstrap = {}
    for name in BOOTSTRAP_FILES:
        path = code/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('missing isolated runtime bootstrap: ' + name)
        bootstrap[name] = sha_file(path)
    preflight, preflight_hash = read_json(args.deployment_preflight)
    verify_preflight(preflight, bootstrap)
    source = args.harness_source.resolve(strict=True)
    revised_profile, revision = build_revised_profile(source, pairing['source_profile'], args.campaign)
    serialized = json.dumps(pairing, indent=2, sort_keys=True) + '\n'
    native_serialized = None if original_verification is None else json.dumps(original_verification, indent=2, sort_keys=True) + '\n'
    import hashlib
    manifest = build_manifest(pairing, source_manifest, campaign=args.campaign, bootstrap=bootstrap,
        pairing_file_sha256=hashlib.sha256(serialized.encode()).hexdigest(), preflight_sha256=preflight_hash,
        revised_profile=revised_profile, harness_revision=revision, bootstrap_relative_root=args.bootstrap_dir,
        native_verification_sha256=None if native_serialized is None else hashlib.sha256(native_serialized.encode()).hexdigest())
    # Preserve partial preparation on failure; no deletion, source mutation, or implicit restart.
    out.mkdir(parents=True, exist_ok=False)
    (out/source_folder).mkdir(); (out/'source_evidence').mkdir()
    for name,path in sources.items():shutil.copyfile(path,out/source_folder/name)
    shutil.copyfile(args.deployment_preflight,out/'source_evidence/deployment_preflight.json')
    if native_serialized is not None:
        (out/'source_evidence/source_native_verification.json').write_text(native_serialized)
    (out/'paired_states.json').write_text(serialized)
    copy_revised_profile(source, profile, manifest['source_profile'])
    if source_files(profile) != manifest['source_profile']['files_sha256']:
        raise ValueError('copied harness bytes failed verification')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    verify_pairing_artifacts(out, manifest)
    (out/'deployment.json').write_text(json.dumps({'manifest_sha256':sha_file(out/'manifest.json'),
        'bootstrap_sha256':bootstrap,'bootstrap_relative_root':args.bootstrap_dir,
        'source_profile_files':len(manifest['source_profile']['files_sha256']), 'cpu_preparation_verified':True,
        'gpu_or_paid_api_verified':False}, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'campaign':args.campaign,'episodes':manifest['planned_scored_episodes'],
        'manifest':str(out/'manifest.json'),'submitted':False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
