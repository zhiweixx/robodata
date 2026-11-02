"""Read-only audit of six unused-state perception pairs and all raw model usage."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import audit_full_results as full
from audit_paired_results import episode_metrics, ledger_episode_metrics
from perception_manifest import (ARMS, HARDWARE, PINNED_MODEL, PINNED_SAMPLING,
    PINNED_THINKING, arm_campaign, sha_file, validate_manifest, verify_profiles,
    verify_layouts, verify_benchmark_sources, validate_service_health,
    LEGACY_PROTOCOL, validate_bootstrap_evidence, bootstrap_source_hashes)


def allocation_model_evidence(path, manifest):
    report = full.read_json(path)
    if manifest['protocol'] == LEGACY_PROTOCOL:
        bootstrap = {'verified': False, 'reason': 'Legacy v1 recorded mutable common-source hashes without a manifest freeze.'}
    else:
        if report.get('source_sha256') != bootstrap_source_hashes(manifest):
            raise ValueError('allocation source_sha256 differs from frozen bootstrap inventory')
        if report.get('batch_script_sha256') != manifest['bootstrap']['files_sha256']['qwen_full/perception_eval.sbatch']:
            raise ValueError('submitted batch script differs from frozen bootstrap')
        bootstrap = validate_bootstrap_evidence(report.get('bootstrap_verification'), manifest,
            roles=('supervisor', 'manifest', 'full_supervisor', 'paired_supervisor', 'perception_services'),
            source_sha256=report['source_sha256'])
    pinned, command = report.get('model_manifest', {}), report.get('server_command', [])
    def option(name):
        return command[command.index(name) + 1] if name in command else None
    if (report.get('actual_inference_variant') != 'bf16_mtp3'
            or report.get('context_length') != 262144
            or {k: pinned.get(k) for k in ('repo', 'revision')} != PINNED_MODEL
            or option('--dtype') != 'bfloat16' or option('--tensor-parallel-size') != '2'
            or option('--max-model-len') != '262144' or option('--max-num-seqs') != '2'
            or json.loads(option('--speculative-config') or '{}') != {'method': 'mtp', 'num_speculative_tokens': 3}
            or option('--model') != pinned.get('snapshot_path')
            or report.get('inference_variant_evidence', {}).get('variant') != 'bf16_mtp3'
            or report.get('execution_hardware') != HARDWARE
            or any(report.get('profile_verification', {}).get(arm, {}).get('files_manifest_sha256')
                   != manifest['profiles'][arm]['files_manifest_sha256'] for arm in ARMS)):
        raise ValueError('allocation model/source/hardware differs from frozen comparison')
    provenance = service_provenance(report.get('perception_services', {}), manifest)
    gpus = report.get('gpus', [])
    if len(gpus) != 4 or provenance['gpu_uuid'] != gpus[3].get('uuid'):
        raise ValueError('perception services did not use the allocated perception GPU')
    return {'allocation': report['allocation'], 'model_revision': pinned['revision'],
            'allocation_report_sha256': sha_file(path), 'started_utc': report.get('started_utc'),
            'finished_utc': report.get('finished_utc'), 'execution_hardware': report['execution_hardware'],
            'bootstrap_freeze': bootstrap}


def service_provenance(provenance, manifest):
    artifacts = manifest['perception_artifacts']
    if any(provenance.get(key) != artifacts[key] for key in (
            'runtime_manifest', 'runtime_manifest_sha256', 'service_runner_sha256')):
        raise ValueError('runtime perception artifacts differ from frozen weights/environment/service runner')
    if provenance.get('offline_skill_discovery') is not False:
        raise ValueError('perception service provenance does not confirm no offline skill discovery')
    if not isinstance(provenance.get('gpu_uuid'), str) or not provenance['gpu_uuid'].startswith('GPU-'):
        raise ValueError('missing physical perception GPU identity')
    health = provenance.get('service_health', {})
    for kind in ('sam3', 'graspnet'):
        validate_service_health(health.get(kind, {}), artifacts['runtime_manifest'], kind,
                                estimator_patch=artifacts.get('estimator_patch'))
    return provenance


def provider_evidence(report, arm, manifest=None):
    actual = report.get('perception_runtime', {})
    expected = {'sam3': arm == 'perception', 'contact_graspnet': arm == 'perception'}
    if actual.get('providers') != expected:
        raise ValueError('actual API perception provider configuration differs from arm')
    if actual.get('offline_skill_discovery') is not False:
        raise ValueError('runtime did not verify no offline skill discovery')
    if manifest is not None:
        service_provenance(actual.get('service_provenance', {}), manifest)
        if actual.get('initialization_complete') is not True:
            raise ValueError('perception API initialization did not complete')
        if arm == 'perception' and actual.get('provider_health') != {'sam3': True, 'contact_graspnet': True}:
            raise ValueError('actual perception providers did not pass health checks')
        if actual.get('grasp_calibration') != (manifest['grasp_calibration'] if arm == 'perception' else None):
            raise ValueError('actual grasp calibration differs from frozen robot geometry')
    return actual


def audit_comparison(campaign_dir, *, simroot=None, progress_only=False):
    campaign_dir = Path(campaign_dir).resolve(strict=True)
    root = campaign_dir.parent.parent
    simroot = Path(simroot) if simroot is not None else root.parent
    manifest_path = campaign_dir / 'manifest.json'
    manifest = validate_manifest(full.read_json(manifest_path))
    if campaign_dir.name != manifest['campaign']:
        raise ValueError('comparison path differs from pinned identity')
    errors, groups, episodes = [], [], {}
    result = {'campaign': manifest['campaign'], 'snapshot_utc': datetime.now(timezone.utc).isoformat(),
              'audit_mode': 'progress_only' if progress_only else 'full_evidence',
              'target_pairs': 6, 'target_episodes': 12, 'selection_note': manifest['selection_note'],
              'official_leaderboard_eligible': False, 'complete': False, 'errors': errors,
              'groups': groups, 'pairs': [], 'arms': {}, 'offline_skill_discovery': False}
    result['bootstrap_freeze_required'] = manifest['protocol'] != LEGACY_PROTOCOL
    result['bootstrap_freeze_verified'] = False
    if not progress_only:
        try:
            result['source_profiles_verified'] = verify_profiles(root, manifest)
            result['initial_state_sources_verified'] = verify_layouts(manifest)
            result['benchmark_sources_verified'] = verify_benchmark_sources(manifest)
        except (OSError, ValueError, KeyError) as exc:
            full.issue(errors, manifest_path, exc)
    for arm in ARMS:
        arm_rows = []
        for unit in manifest['units']:
            uid, group = unit['unit_id'], unit['group']
            arm_dir = campaign_dir.parent / arm_campaign(manifest, arm, uid)
            row = full.audit_group(arm_dir, simroot, group, errors, progress_only=progress_only)
            row.update(arm=arm, unit_id=uid)
            groups.append(row)
            arm_rows.append(row)
            native_path = Path(row['native_result_path'])
            try:
                native = full.native_details(full.read_json(native_path), group) if native_path.exists() else {}
            except (OSError, ValueError, KeyError) as exc:
                full.issue(errors, native_path, exc)
                native = {}
            contract_path = arm_dir / 'groups' / group['group_id'] / 'perception_contract.json'
            if not progress_only and row['scored']:
                try:
                    contract = full.read_json(contract_path)
                    expected = {'protocol': manifest['protocol'], 'comparison': manifest['campaign'],
                        'arm': arm, 'unit_id': uid, 'manifest_sha256': sha_file(manifest_path),
                        'profile': manifest['profiles'][arm], 'group_id': group['group_id'],
                        'candidate_layout_ids': group['candidate_layout_ids'],
                        'step_limit': group['step_limit'], 'target': 1,
                        'providers': {'sam3': arm == 'perception', 'contact_graspnet': arm == 'perception'}}
                    if any(contract.get(k) != v for k, v in expected.items()):
                        raise ValueError('executed contract differs from comparison manifest')
                    if not contract.get('imported_sources'):
                        raise ValueError('missing imported-source evidence')
                    for filename in contract['imported_sources'].values():
                        expected_profile = Path('/work/aspire_validate') / manifest['profiles'][arm]['relative_root']
                        if not Path(filename).is_relative_to(expected_profile):
                            raise ValueError('simulator imported outside frozen source profile')
                    if manifest['protocol'] != LEGACY_PROTOCOL:
                        validate_bootstrap_evidence(contract.get('bootstrap_verification'), manifest,
                            roles=('entry', 'inside', 'manifest'))
                except (OSError, ValueError, KeyError) as exc:
                    full.issue(errors, contract_path, exc)
            for native_index, detail in native.items():
                eid = f"{group['group_id']}_l{detail['layout_id']}"
                item = {'episode_id': eid, 'unit_id': uid, 'family': group['family'], 'task': group['task'],
                        'layout_id': detail['layout_id'], 'native_index': int(native_index),
                        'success': detail['success'], 'score': detail['score'],
                        'native_result': str(native_path), 'initial_state_sha256': unit['initial_state_sha256']}
                if not progress_only:
                    paths = list((arm_dir / 'groups' / group['group_id'] / 'episodes' / eid).glob('attempt_*/episode_report.json'))
                    reports = [(path, full.read_json(path)) for path in paths]
                    committed = [(path, report) for path, report in reports if report.get('native_score_committed') is True]
                    if len(committed) != 1:
                        full.issue(errors, native_path, 'expected exactly one committed episode report')
                    else:
                        path, report = committed[0]
                        item.update(episode_report=str(path), episode_report_sha256=sha_file(path),
                                    **episode_metrics(report))
                        try:
                            item['perception_runtime'] = provider_evidence(report, arm, manifest)
                        except ValueError as exc:
                            full.issue(errors, path, exc)
                        if report.get('official', {}).get('official_success') is not detail['success']:
                            full.issue(errors, path, 'episode/native success mismatch')
                episodes[(arm, uid)] = item
        result['arms'][arm] = {'scored': sum(r['scored'] for r in arm_rows),
            'success': sum(r['success'] for r in arm_rows), 'score_sum': sum(r['score_sum'] for r in arm_rows), 'target': 6}
    for unit in manifest['units']:
        uid = unit['unit_id']
        baseline, treatment = episodes.get(('baseline', uid)), episodes.get(('perception', uid))
        pair = {'unit_id': uid, 'episode_id': unit['group']['episode_ids'][0],
                'arm_order': unit['arm_order'], 'baseline': baseline, 'perception': treatment,
                'both_scored': baseline is not None and treatment is not None}
        if pair['both_scored']:
            pair.update(success_delta=int(treatment['success']) - int(baseline['success']),
                        score_delta=treatment['score'] - baseline['score'])
            if not progress_only and all('wall_seconds' in x for x in (baseline, treatment)):
                pair['wall_seconds_delta'] = treatment['wall_seconds'] - baseline['wall_seconds']
        result['pairs'].append(pair)
    result['scored'] = sum(row['scored'] for row in result['arms'].values())
    result['complete_pairs'] = sum(pair['both_scored'] for pair in result['pairs'])
    result['success_transitions'] = dict(Counter(
        str(int(p['baseline']['success'])) + '->' + str(int(p['perception']['success']))
        for p in result['pairs'] if p['both_scored']))
    if not progress_only:
        usage = {arm: full.empty_usage() for arm in ARMS}
        ledgers, allocations, covered, per_episode = [], {}, set(), {}
        by_uid = {u['unit_id']: u for u in manifest['units']}
        for path in sorted((campaign_dir / 'allocations').glob('*/scored/worker0/*/usage.jsonl')):
            try:
                arm, uid = path.parent.name.split('__', 1)
                if arm not in ARMS or uid not in by_uid:
                    raise ValueError('ledger outside authorized arm/state')
                group = by_uid[uid]['group']
                allocation_path = path.parents[3] / 'allocation.json'
                if str(allocation_path) not in allocations:
                    allocations[str(allocation_path)] = allocation_model_evidence(allocation_path, manifest)
                with path.open() as handle:
                    first = json.loads(next(handle))
                config = first.get('config', {})
                if (first.get('event') != 'suite_initialized' or config.get('sampling') != PINNED_SAMPLING
                        or config.get('chat_template_kwargs') != PINNED_THINKING
                        or config.get('driver_sha256') != manifest['profiles'][arm]['files_sha256']['qwen_full/full_model_driver.py']):
                    raise ValueError('model sampling/thinking/driver differs from pinned configuration')
                measured = full.audit_usage(path, group['episode_ids'], {e: 0 for e in group['episode_ids']}, errors)
                if measured is None:
                    continue
                full.add_usage(usage[arm], measured)
                ledgers.append({'path': str(path), 'arm': arm, 'unit_id': uid, **measured})
                covered.add((arm, uid))
                metrics = ledger_episode_metrics(path, group['episode_ids'], errors)[group['episode_ids'][0]]
                key = (arm, uid)
                if key not in per_episode:
                    per_episode[key] = metrics
                else:
                    for field, value in metrics.items():
                        per_episode[key][field] += value
            except (OSError, ValueError, KeyError, IndexError, StopIteration) as exc:
                full.issue(errors, path, exc)
        for key, item in episodes.items():
            measured = per_episode.get(key)
            if measured is None or measured['model_calls_started'] + measured['tokenization_calls'] == 0:
                full.issue(errors, item['native_result'], 'scored episode lacks raw model evidence')
            else:
                item['raw_ledger_usage_all_attempts'] = measured
        result.update(raw_ledger_usage_by_arm=usage, raw_ledger_usage_details=ledgers,
                      verified_model_allocations=list(allocations.values()),
                      usage_note='All terminal and interrupted requests included, not only returned code iterations.')
        result['complete'] = (not errors and result['complete_pairs'] == 6
            and all(row['native_complete_verified'] and row['episode_reports_verified'] == 1
                    and row['video_evidence']['verified_scored_episodes'] == 1 for row in groups)
            and covered == {(arm, uid) for arm in ARMS for uid in by_uid})
        result['bootstrap_freeze_verified'] = (result['complete']
            and result['bootstrap_freeze_required'])
    result['interpretation_limits'] = [
        'Six preselected unused states, one fresh run per arm; exploratory comparison, not benchmark-wide significance.',
        'Same simulator GPU and serial arms; alternate arm order. Perception services remain resident in both arms.',
        'Perception API/docs availability is the treatment; no offline search, learned skill library or new action budget.',
        'Unavailable/invalid states remain incomplete pairs; no post-outcome substitutions.']
    if manifest['protocol'] == LEGACY_PROTOCOL:
        result['interpretation_limits'].append(
            'Legacy v1: the common launcher/entrypoint were not frozen by the manifest; '
            'their historical hashes do not establish a verified bootstrap. Scoring/usage audit remains separate.')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign-dir', type=Path, required=True)
    parser.add_argument('--simroot', type=Path)
    parser.add_argument('--progress-only', action='store_true')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit_comparison(args.campaign_dir, simroot=args.simroot, progress_only=args.progress_only)
    serialized = json.dumps(result, indent=None if args.compact else 2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n')
    print(serialized)


if __name__ == '__main__':
    main()
