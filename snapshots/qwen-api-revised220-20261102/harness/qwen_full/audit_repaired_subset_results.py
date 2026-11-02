#!/usr/bin/env python3
"""Audit the repaired-harness 767-episode campaign without importing old scores.

The established subset audit owns native scores, videos, budgets and usage.
This layer verifies frozen execution, actual inference/providers, and evidence
for each newly scored episode. Progress snapshots avoid the full evidence walk.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import audit_full_results as full
import audit_subset_results as subset
from audit_paired_results import ledger_episode_metrics
from audit_perception_results import service_provenance
from build_subset_manifest import verify_group_layout_sources
from paired_manifest import sha_file, verify_benchmark_sources
from repaired_subset_contract import load_contract, verify_profile


ALLOCATION_SOURCES = frozenset({
    'qwen_full/repaired_subset_supervisor.py', 'qwen_full/repaired_subset_contract.py',
    'qwen_full/full_supervisor.py', 'qwen_full/subset_supervisor.py',
    'qwen_full/perception_services.py',
})
GROUP_SOURCES = frozenset({
    'qwen_full/repaired_subset_entry.py', 'qwen_full/repaired_subset_inside.sh',
    'qwen_full/repaired_subset_contract.py', 'qwen_full/subset_entry.py',
    'qwen_full/full_entry.py', 'qwen_full/full_episode.py',
    'qwen_full/perception_entry.py', 'qwen_integration/aspire_glue.py',
    'qwen_integration/robodojo_adapter.py', 'qwen_integration/receipt_executor.py',
    'qwen_integration/native_stdout.py', 'qwen_integration/perception_bridge.py',
    'ASPIRE/src/cap_harness/api.py', 'ASPIRE/src/cap_harness/contracts.py',
    'ASPIRE/src/cap_harness/runtime.py',
})
PERCEPTION_TOOLS = frozenset({'segment_text', 'segment_points', 'localize_object', 'generate_grasps'})


def validate_bootstrap_record(evidence, root, contract, required):
    """Check saved container/host paths without resolving a foreign filesystem."""
    profile = contract['source_profile']
    if not isinstance(evidence, dict):
        raise ValueError('missing executed-source bootstrap evidence')
    roots = {str(Path(root) / profile['relative_root']),
             str(Path('/work/aspire_validate') / profile['relative_root'])}
    recorded_root = evidence.get('profile_root')
    if (recorded_root not in roots
            or evidence.get('files_manifest_sha256') != profile['files_manifest_sha256']):
        raise ValueError('bootstrap profile root/hash differs from frozen source')
    executed = evidence.get('executed_sources', {})
    if not isinstance(executed, dict) or not required <= executed.keys():
        raise ValueError('bootstrap lacks required executed-source evidence')
    for name, actual in executed.items():
        expected_hash = profile['files_sha256'].get(name)
        if (expected_hash is None or not isinstance(actual, dict)
                or actual.get('sha256') != expected_hash
                or actual.get('path') != str(Path(recorded_root) / name)):
            raise ValueError(f'executed source differs from frozen profile: {name}')
    return evidence


def validate_allocation(report, root, contract, contract_sha256, profile_verification):
    profile = contract['source_profile']
    expected_sources = {Path(name).name: digest for name, digest in profile['files_sha256'].items()
                        if Path(name).parent == Path('qwen_full')
                        and Path(name).suffix in {'.py', '.sh', '.sbatch'}}
    if (report.get('execution_contract_sha256') != contract_sha256
            or report.get('manifest_sha256') != contract['dataset_manifest_sha256']
            or report.get('source_profile_verification') != profile_verification
            or report.get('source_sha256') != expected_sources
            or report.get('batch_script_sha256') != profile['files_sha256']['qwen_full/repaired_subset_eval.sbatch']
            or report.get('execution_hardware') != contract['hardware']
            or report.get('providers') != contract['providers']):
        raise ValueError('allocation execution/source/hardware differs from frozen contract')
    validate_bootstrap_record(report.get('bootstrap_verification'), root, contract, ALLOCATION_SOURCES)
    pinned, command = report.get('model_manifest', {}), report.get('server_command', [])
    def option(name):
        if command.count(name) != 1:
            return None
        index = command.index(name)
        return command[index + 1] if index + 1 < len(command) else None
    inference = contract['inference']
    seed = str(inference['model_seed'])
    # The inherited launcher leaves vLLM's seed at its default. Verify the
    # loaded engine's recorded value rather than assuming that default.
    seed_verified = (option('--seed') == seed if '--seed' in command else bool(re.search(
        r'\bseed\s*=\s*' + re.escape(seed) + r'\b',
        report.get('inference_variant_evidence', {}).get('engine_config', ''))))
    if ({key: pinned.get(key) for key in ('repo', 'revision')} != contract['model']
            or report.get('actual_inference_variant') != 'bf16_mtp3'
            or report.get('context_length') != inference['native_context_tokens']
            or report.get('inference_variant_evidence', {}).get('variant') != 'bf16_mtp3'
            or option('--model') != pinned.get('snapshot_path')
            or option('--dtype') != inference['dtype']
            or option('--tensor-parallel-size') != str(inference['tensor_parallel_size'])
            or option('--max-model-len') != str(inference['native_context_tokens'])
            or option('--max-num-seqs') != str(inference['max_num_seqs'])
            or not seed_verified
            or json.loads(option('--speculative-config') or '{}') != {
                'method': 'mtp', 'num_speculative_tokens': inference['mtp_tokens']}):
        raise ValueError('actual Qwen revision/inference differs from frozen contract')
    if contract['providers']['sam3'] or contract['providers']['contact_graspnet']:
        provenance = service_provenance(report.get('perception_services', {}), contract)
        gpus = report.get('gpus', [])
        index = contract['hardware']['perception_gpu']
        if len(gpus) != 4 or provenance['gpu_uuid'] != gpus[index].get('uuid'):
            raise ValueError('perception services used a different physical GPU')
    return report


def validate_group_contract(record, group, root, contract, contract_sha256, campaign):
    expected = {'campaign': campaign, 'group_id': group['group_id'],
                'protocol': contract['protocol'], 'task': group['task'], 'eval_seed': group['eval_seed'],
                'execution_contract_sha256': contract_sha256,
                'dataset_manifest_sha256': contract['dataset_manifest_sha256'],
                'step_limit': group['step_limit'], 'target_episodes': group['target_episodes'],
                'candidate_layout_ids': group['candidate_layout_ids'],
                'source_profile': contract['source_profile'], 'providers': contract['providers']}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('executed subset group differs from dataset/runtime contract')
    validate_bootstrap_record(record.get('bootstrap_verification'), root, contract, GROUP_SOURCES)
    profile = contract['source_profile']
    expected_profile = {'profile_root': record['bootstrap_verification']['profile_root'],
                       'git_commit': profile['git_commit'], 'files_manifest_sha256': profile['files_manifest_sha256'],
                       'files_count': len(profile['files_sha256'])}
    if record.get('source_profile_verification') != expected_profile:
        raise ValueError('group source inventory differs from frozen profile verification')
    return record


def validate_ledger_header(path, contract):
    with Path(path).open() as stream:
        first = json.loads(next(stream))
    config = first.get('config', {})
    if (first.get('event') != 'suite_initialized'
            or config.get('sampling') != contract['sampling']
            or config.get('chat_template_kwargs') != contract['thinking']
            or config.get('driver_sha256') != contract['source_profile']['files_sha256']['qwen_full/full_model_driver.py']
            or any(key not in config or config[key] != value for key, value in full.UNCAPPED_CONFIG.items())
            or config.get('remote_api_calls') is not False):
        raise ValueError('model sampling/thinking/driver or native budgets changed')
    return config


def validate_episode_runtime(report, contract, provider_configs):
    actual = report.get('perception_runtime', {})
    providers = {name: contract['providers'][name] for name in ('sam3', 'contact_graspnet')}
    enabled = any(providers.values())
    if (actual.get('providers') != providers or actual.get('initialization_complete') is not True
            or actual.get('offline_skill_discovery') is not False
            or actual.get('services_resident_for_both_arms') is not False
            or actual.get('arm') != ('perception' if enabled else 'baseline')
            or actual.get('provider_config_sha256') not in provider_configs):
        raise ValueError('episode lacks matching actual provider/runtime evidence')
    saved = provider_configs[actual['provider_config_sha256']]
    if actual.get('service_provenance') != saved.get('provenance'):
        raise ValueError('episode service evidence differs from its verified allocation configuration')
    if enabled:
        service_provenance(actual.get('service_provenance', {}), contract)
        if (actual.get('provider_health') != providers
                or actual.get('grasp_calibration') != contract['grasp_calibration']):
            raise ValueError('episode provider readiness/calibration differs from frozen contract')
    elif (actual.get('provider_health') != {} or actual.get('grasp_calibration') is not None):
        raise ValueError('disabled providers unexpectedly report active readiness/calibration')
    return actual


def audit_program_evidence(report, attempt_dir):
    """Preserve raw native stdout without treating tool availability as use."""
    archives, calls = [], Counter()
    seen = set()
    for entry in report.get('iterations', []):
        if entry.get('done'):
            continue
        name = entry.get('native_stdout_log')
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or '..' in Path(name).parts or name in seen):
            raise ValueError('executed program lacks a unique relative native stdout archive')
        seen.add(name)
        path = Path(attempt_dir) / name
        if not path.is_file() or not path.resolve().is_relative_to(Path(attempt_dir).resolve()):
            raise ValueError('native stdout archive missing or outside its attempt')
        archives.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha_file(path)})
        for call in (entry.get('execution') or {}).get('tool_calls', []):
            if call.get('tool') in PERCEPTION_TOOLS:
                key = (call['tool'], str(call.get('provider')), str(call.get('ok')),
                       str(call.get('failure_code')))
                calls[key] += 1
    return {'native_stdout_archives': archives,
            'perception_calls': [{'tool': k[0], 'provider': k[1], 'ok': k[2],
                                  'failure_code': k[3], 'count': value} for k, value in sorted(calls.items())]}


def audit_campaign(campaign_dir, *, simroot=None, now=None, progress_only=False,
                   include_details=False):
    campaign_dir = Path(campaign_dir).resolve(strict=True)
    root = campaign_dir.parent.parent
    contract = load_contract(root, campaign_dir.name)
    contract_path = campaign_dir / 'execution_manifest.json'
    contract_sha256 = sha_file(contract_path)
    dataset = full.read_json(campaign_dir / 'manifest.json')
    result = subset.audit_campaign(campaign_dir, simroot=simroot, now=now,
        progress_only=progress_only, include_details=include_details)
    result.pop('reused_results', None)
    result.pop('combined_807_summary', None)
    result.update(runtime_protocol=contract['protocol'], execution_contract_sha256=contract_sha256,
                  reference_results_included=False, new_episode_target=subset.NEW_TARGET)
    result['summary_note'] = ('767 newly evaluated outcomes only. Historical scores are excluded. '
                              'This custom subset is not an official full-benchmark Avg.')
    errors = result['errors']
    if (campaign_dir / 'reused_results.json').exists():
        full.issue(errors, campaign_dir / 'reused_results.json', 'historical reuse file is forbidden in this new-only campaign')
    if progress_only:
        result['runtime_evidence_audited'] = False
        result['campaign_complete'] = False
        return result

    try:
        profile_verification = verify_profile(root, contract)
        result['verified_benchmark_source_files'] = verify_benchmark_sources(dataset)
        for group in dataset['groups']:
            verify_group_layout_sources(group)
        result['source_profile_verification'] = profile_verification
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        full.issue(errors, contract_path, exc)
        profile_verification = None

    group_map = {g['group_id']: g for g in dataset['groups']}
    per_episode, provider_configs, verified_allocations = {}, {}, []
    for allocation in sorted((campaign_dir / 'allocations').glob('*')):
        if not allocation.is_dir():
            continue
        ledgers, _ = full.allocation_usage_ledgers(allocation, errors)
        if not ledgers:
            continue  # Failed initialization without evaluated model work is retained by the base audit.
        try:
            report = full.read_json(allocation / 'allocation.json')
            validate_allocation(report, root, contract, contract_sha256, profile_verification)
            config_path = allocation / 'providers.json'
            config = full.read_json(config_path)
            if any(contract['providers'][key] for key in ('sam3', 'contact_graspnet')):
                if config.get('provenance') != report.get('perception_services'):
                    raise ValueError('retained provider configuration differs from allocation evidence')
            provider_configs[sha_file(config_path)] = config
            verified_allocations.append(str(allocation / 'allocation.json'))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            full.issue(errors, allocation / 'allocation.json', exc)
        for ledger in ledgers:
            try:
                group = group_map[ledger.parent.name]
                validate_ledger_header(ledger, contract)
                metrics = ledger_episode_metrics(ledger, group['episode_ids'], errors)
                for eid, row in metrics.items():
                    aggregate = per_episode.setdefault(eid, {key: 0 for key in row})
                    for key, value in row.items():
                        aggregate[key] += value
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration) as exc:
                full.issue(errors, ledger, exc)

    committed = []
    for group in dataset['groups']:
        group_dir = campaign_dir / 'groups' / group['group_id']
        reports = sorted(group_dir.glob('episodes/*/attempt_*/episode_report.json'))
        if not reports:
            continue
        group_verified = False
        try:
            record = full.read_json(group_dir / 'repaired_subset_contract.json')
            validate_group_contract(record, group, root, contract, contract_sha256, campaign_dir.name)
            if record.get('provider_config_sha256') not in provider_configs:
                raise ValueError('group provider configuration lacks a verified allocation')
            group_verified = True
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            full.issue(errors, group_dir / 'repaired_subset_contract.json', exc)
        for path in reports:
            try:
                report = full.read_json(path)
                if report.get('native_score_committed') is not True:
                    continue
                validate_episode_runtime(report, contract, provider_configs)
                evidence = audit_program_evidence(report, path.parent)
                metrics = per_episode.get(report['episode_id'])
                if metrics is None or metrics['model_calls_started'] + metrics['tokenization_calls'] == 0:
                    raise ValueError('scored episode lacks raw model request/tokenization evidence')
                if group_verified and profile_verification is not None:
                    committed.append({'episode_id': report['episode_id'], 'report': str(path),
                                      'raw_usage_all_attempts': metrics, **evidence})
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                full.issue(errors, path, exc)
    result.update(runtime_evidence_audited=True, verified_model_allocations=verified_allocations,
                  scored_episodes_with_verified_runtime=len(committed),
                  verified_native_stdout_archives=sum(len(row['native_stdout_archives']) for row in committed),
                  native_stdout_archive_bytes=sum(item['bytes'] for row in committed for item in row['native_stdout_archives']),
                  actual_perception_call_note='Receipt counts distinguish calls from enabled providers; argument failures do not establish backend inference.')
    totals = Counter()
    for row in committed:
        for call in row['perception_calls']:
            totals[(call['tool'], call['provider'], call['ok'], call['failure_code'])] += call['count']
    result['actual_perception_calls'] = [{'tool': key[0], 'provider': key[1], 'ok': key[2],
                                        'failure_code': key[3], 'count': count}
                                       for key, count in sorted(totals.items())]
    if include_details:
        result['runtime_episode_evidence'] = committed
    result['campaign_complete'] = (result['campaign_complete'] and not errors
                                   and len(committed) == subset.NEW_TARGET)
    result['final_evidence_audit_required'] = not result['campaign_complete']
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', type=Path, required=True)
    parser.add_argument('--simroot', type=Path)
    parser.add_argument('--progress-only', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--details', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    result = audit_campaign(args.campaign_dir, simroot=args.simroot,
                            progress_only=args.progress_only, include_details=args.details)
    if args.summary_only:
        for key in ('by_family', 'by_category_family_seed', 'native_action_budgets',
                    'group_details', 'runtime_episode_evidence'):
            result.pop(key, None)
    text = json.dumps(result, indent=None if args.compact else 2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
