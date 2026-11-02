"""Read-only native evidence and paired diagnostic metrics, including raw usage."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import audit_full_results as full
from paired_manifest import (ARMS, PINNED_MODEL, PINNED_SAMPLING, PINNED_THINKING, arm_campaign,
                             sha_file, validate_manifest, verify_layouts, verify_profiles,
                             verify_benchmark_sources)


def allocation_model_evidence(path, manifest):
    report = full.read_json(path)
    pinned = report.get('model_manifest', {})
    command = report.get('server_command', [])
    def option(name):
        return command[command.index(name) + 1] if name in command else None
    if (report.get('actual_inference_variant') != 'bf16_mtp3'
            or report.get('context_length') != 262144
            or {key: pinned.get(key) for key in ('repo', 'revision')} != PINNED_MODEL
            or option('--dtype') != 'bfloat16' or option('--tensor-parallel-size') != '2'
            or option('--max-model-len') != '262144' or option('--max-num-seqs') != '2'
            or json.loads(option('--speculative-config') or '{}') != {'method': 'mtp', 'num_speculative_tokens': 3}
            or option('--model') != pinned.get('snapshot_path')
            or report.get('inference_variant_evidence', {}).get('variant') != 'bf16_mtp3'
            or any(report.get('profile_verification', {}).get(arm, {}).get('files_manifest_sha256')
                   != manifest['profiles'][arm]['files_manifest_sha256'] for arm in ARMS)):
        raise ValueError('allocation did not verify pinned Qwen BF16+MTP3 TP2/context/source profiles')
    return {'allocation': report['allocation'], 'actual_inference_variant': report['actual_inference_variant'],
            'model_revision': pinned['revision'], 'allocation_report_sha256': sha_file(path),
            'started_utc': report.get('started_utc'), 'finished_utc': report.get('finished_utc')}


def ledger_episode_metrics(path, allowed_ids, errors):
    """All attempts/terminal calls for each state; not just returned program turns."""
    stats = {eid: {'model_calls_started': 0, 'model_calls_with_usage': 0,
                  'prompt_tokens': 0, 'completion_tokens_including_reasoning': 0,
                  'generation_wall_seconds': 0., 'tokenization_wall_seconds': 0.,
                  'calls_without_final_token_usage': 0, 'abandoned_attempts': 0,
                  'model_episode_failures': 0, 'tokenization_calls': 0} for eid in allowed_ids}
    starts, usages, runtime, tokenizations = set(), set(), {}, {}
    for row in full.jsonl_rows(path, errors):
        eid = row.get('episode_id')
        if eid is None:
            continue
        if eid not in stats:
            raise ValueError('usage episode outside paired group')
        key = (eid, row.get('attempt', 1), row.get('iteration'))
        event = row.get('event')
        if event == 'generation_started':
            starts.add(key)
        elif event == 'generation_usage':
            usages.add(key)
            usage = row['usage']
            stats[eid]['prompt_tokens'] += usage['prompt_tokens']
            stats[eid]['completion_tokens_including_reasoning'] += usage['completion_tokens']
            runtime.setdefault(key, usage['generation_wall_s'])
        elif event in ('raw_response', 'generation_transport_failure'):
            runtime[key] = row['generation_wall_s']
        elif event == 'input_token_count':
            tokenizations[key] = row['wall_s']
        elif event == 'unscored_attempt_abandoned':
            stats[eid]['abandoned_attempts'] += 1
        elif event == 'model_episode_failed':
            stats[eid]['model_episode_failures'] += 1
    for eid in stats:
        stats[eid]['model_calls_started'] = sum(key[0] == eid for key in starts)
        stats[eid]['model_calls_with_usage'] = sum(key[0] == eid for key in usages)
        stats[eid]['calls_without_final_token_usage'] = sum(key[0] == eid for key in starts - usages)
        stats[eid]['generation_wall_seconds'] = sum(value for key, value in runtime.items() if key[0] == eid)
        stats[eid]['tokenization_wall_seconds'] = sum(value for key, value in tokenizations.items() if key[0] == eid)
        stats[eid]['tokenization_calls'] = sum(key[0] == eid for key in tokenizations)
    return stats


def episode_metrics(report):
    counts, durations, failures = Counter(), Counter(), Counter()
    program_errors = []
    program_seconds = feedback_chars = 0
    for iteration in report.get('iterations', []):
        execution = iteration.get('execution') or {}
        program_seconds += iteration.get('program_wall_seconds', 0) or 0
        feedback_chars += len(json.dumps(execution, ensure_ascii=False))
        if execution.get('program_ok') is False:
            program_errors.append(str(execution.get('error')))
        for call in execution.get('tool_calls', []):
            tool = call['tool']
            counts[tool] += 1
            durations[tool] += call.get('duration_s', 0) or 0
            if call.get('ok') is False:
                failures[tool + ':' + str(call.get('failure_code'))] += 1
    return {'termination_reason': report.get('termination_reason'),
            'environment_steps': report.get('environment_steps'), 'step_limit': report.get('step_limit'),
            'wall_seconds': report.get('wall_seconds'), 'program_seconds': program_seconds,
            'feedback_chars': feedback_chars, 'program_errors': program_errors,
            'tool_counts': dict(counts), 'tool_seconds': dict(durations), 'tool_failures': dict(failures)}


def audit_comparison(campaign_dir, *, simroot=None, progress_only=False):
    campaign_dir = Path(campaign_dir).resolve(strict=True)
    root = campaign_dir.parent.parent
    simroot = Path(simroot) if simroot is not None else root.parent
    manifest_path = campaign_dir / 'manifest.json'
    manifest = validate_manifest(full.read_json(manifest_path))
    if campaign_dir.name != manifest['campaign']:
        raise ValueError('comparison directory identity differs from manifest')
    errors, groups, episodes = [], [], {}
    result = {'campaign': manifest['campaign'], 'snapshot_utc': datetime.now(timezone.utc).isoformat(),
              'audit_mode': 'progress_only' if progress_only else 'full_evidence',
              'target_pairs': 12, 'target_episodes': 24, 'selection_note': manifest['selection_note'],
              'official_leaderboard_eligible': False, 'complete': False,
              'errors': errors, 'groups': groups, 'pairs': [], 'arms': {}}
    if not progress_only:
        try:
            result['source_profiles_verified'] = verify_profiles(root, manifest)
            result['initial_state_sources_verified'] = verify_layouts(manifest)
            result['benchmark_sources_verified'] = verify_benchmark_sources(manifest)
        except (OSError, ValueError, KeyError) as exc:
            full.issue(errors, manifest_path, exc)
    for arm in ARMS:
        arm_dir = campaign_dir.parent / arm_campaign(manifest, arm)
        arm_rows = []
        for group in manifest['groups']:
            row = full.audit_group(arm_dir, simroot, group, errors, progress_only=progress_only)
            row['arm'] = arm
            groups.append(row)
            arm_rows.append(row)
            native_path = Path(row['native_result_path'])
            try:
                native = full.native_details(full.read_json(native_path), group) if native_path.exists() else {}
            except (OSError, ValueError, KeyError) as exc:
                full.issue(errors, native_path, exc)
                native = {}
            contract_path = arm_dir / 'groups' / group['group_id'] / 'paired_contract.json'
            if not progress_only and row['scored']:
                try:
                    contract = full.read_json(contract_path)
                    expected = {'protocol': manifest['protocol'], 'comparison': manifest['campaign'],
                                'arm': arm, 'manifest_sha256': sha_file(manifest_path),
                                'profile': manifest['profiles'][arm], 'group_id': group['group_id'],
                                'candidate_layout_ids': group['candidate_layout_ids'],
                                'step_limit': group['step_limit'], 'target': group['target_episodes']}
                    if any(contract.get(key) != value for key, value in expected.items()):
                        raise ValueError('executed paired contract differs from frozen comparison')
                    for filename in contract['imported_sources'].values():
                        expected_profile = Path('/work/aspire_validate') / manifest['profiles'][arm]['relative_root']
                        if not Path(filename).is_relative_to(expected_profile):
                            raise ValueError('simulator imported outside pinned profile')
                except (OSError, ValueError, KeyError) as exc:
                    full.issue(errors, contract_path, exc)
            for native_index, detail in native.items():
                eid = f"{group['group_id']}_l{detail['layout_id']}"
                item = {'episode_id': eid, 'family': group['family'], 'task': group['task'],
                        'layout_id': detail['layout_id'], 'native_index': int(native_index),
                        'success': detail['success'], 'score': detail['score'],
                        'native_result': str(native_path),
                        'initial_state_sha256': group['layout_content_sha256_by_id'][str(detail['layout_id'])]}
                if not progress_only:
                    paths = list((arm_dir / 'groups' / group['group_id'] / 'episodes' / eid).glob('attempt_*/episode_report.json'))
                    reports = [(path, full.read_json(path)) for path in paths]
                    committed = [(path, report) for path, report in reports if report.get('native_score_committed') is True]
                    if len(committed) != 1:
                        full.issue(errors, native_path, f'{eid}: expected exactly one committed episode report')
                    else:
                        path, report = committed[0]
                        item.update(episode_report=str(path), episode_report_sha256=sha_file(path),
                                    **episode_metrics(report))
                        official = report.get('official', {})
                        if official.get('official_success') is not detail['success']:
                            full.issue(errors, path, 'episode/native success mismatch')
                episodes[(arm, eid)] = item
        result['arms'][arm] = {'scored': sum(r['scored'] for r in arm_rows),
                              'success': sum(r['success'] for r in arm_rows),
                              'score_sum': sum(r['score_sum'] for r in arm_rows),
                              'target': 12}
    for group in manifest['groups']:
        for eid in group['episode_ids']:
            baseline, fixed = episodes.get(('baseline', eid)), episodes.get(('fixed', eid))
            pair = {'episode_id': eid, 'baseline': baseline, 'fixed': fixed,
                    'both_scored': baseline is not None and fixed is not None}
            if pair['both_scored']:
                pair['success_delta'] = int(fixed['success']) - int(baseline['success'])
                pair['score_delta'] = fixed['score'] - baseline['score']
                if not progress_only and 'wall_seconds' in baseline and 'wall_seconds' in fixed:
                    pair['wall_seconds_delta'] = fixed['wall_seconds'] - baseline['wall_seconds']
            result['pairs'].append(pair)
    result['scored'] = sum(r['scored'] for r in result['arms'].values())
    result['complete_pairs'] = sum(p['both_scored'] for p in result['pairs'])
    result['success_transitions'] = dict(Counter(
        str(int(p['baseline']['success'])) + '->' + str(int(p['fixed']['success']))
        for p in result['pairs'] if p['both_scored']))
    if not progress_only:
        usage = {arm: full.empty_usage() for arm in ARMS}
        ledgers = []
        covered = set()
        model_allocations = {}
        per_episode_usage = {}
        by_gid = {g['group_id']: g for g in manifest['groups']}
        for path in sorted((campaign_dir / 'allocations').glob('*/scored/worker*/*/usage.jsonl')):
            arm, gid = path.parent.name.split('__', 1)
            if arm not in ARMS or gid not in by_gid:
                full.issue(errors, path, 'usage ledger outside paired identities')
                continue
            group = by_gid[gid]
            try:
                allocation_path = path.parents[3] / 'allocation.json'
                if str(allocation_path) not in model_allocations:
                    model_allocations[str(allocation_path)] = allocation_model_evidence(allocation_path, manifest)
                with path.open() as handle:
                    first = json.loads(next(handle))
                config = first.get('config', {})
                if (first.get('event') != 'suite_initialized'
                        or config.get('sampling') != PINNED_SAMPLING
                        or config.get('chat_template_kwargs') != PINNED_THINKING
                        or config.get('driver_sha256') != manifest['profiles'][arm]['files_sha256']['qwen_full/full_model_driver.py']):
                    raise ValueError('executed model sampling/thinking/driver differs from frozen profile')
            except (OSError, ValueError, KeyError, IndexError, StopIteration) as exc:
                full.issue(errors, path, exc)
                continue
            measured = full.audit_usage(path, group['episode_ids'],
                                        {eid: 0 for eid in group['episode_ids']}, errors)
            if measured is not None:
                full.add_usage(usage[arm], measured)
                ledgers.append({'path': str(path), 'arm': arm, 'group_id': gid, **measured})
                covered.add((arm, gid))
                for eid, metrics in ledger_episode_metrics(path, group['episode_ids'], errors).items():
                    key = (arm, eid)
                    if key not in per_episode_usage:
                        per_episode_usage[key] = metrics
                    else:
                        for field, value in metrics.items():
                            per_episode_usage[key][field] += value
        for key, item in episodes.items():
            measured = per_episode_usage.get(key)
            if measured is None or measured['model_calls_started'] + measured['tokenization_calls'] == 0:
                full.issue(errors, item['native_result'], 'scored episode lacks raw model-request/tokenization evidence: ' + key[0] + ':' + key[1])
            else:
                item['raw_ledger_usage_all_attempts'] = measured
        for pair in result['pairs']:
            if pair['both_scored'] and all('raw_ledger_usage_all_attempts' in pair[arm] for arm in ARMS):
                pair['raw_usage_delta_fixed_minus_baseline'] = {
                    field: pair['fixed']['raw_ledger_usage_all_attempts'][field] - pair['baseline']['raw_ledger_usage_all_attempts'][field]
                    for field in pair['baseline']['raw_ledger_usage_all_attempts']}
        result['raw_ledger_usage_by_arm'] = usage
        result['raw_ledger_usage_details'] = ledgers
        result['verified_model_allocations'] = list(model_allocations.values())
        result['usage_note'] = 'Includes terminal requests and interrupted/unscored attempts. Report iteration sums are not used as total model usage.'
        result['complete'] = (not errors and result['complete_pairs'] == 12
            and all(row['native_complete_verified'] and row['episode_reports_verified'] == row['target']
                    and row['video_evidence']['verified_scored_episodes'] == row['target'] for row in groups)
            and covered == {(arm, gid) for arm in ARMS for gid in by_gid})
    result['interpretation_limits'] = [
        'Twelve failure-enriched states, one fresh sample per arm: a diagnostic, not benchmark-level significance.',
        'The shared model server batches variable requests; compare render/tool counts alongside wall time.',
        'A missing or invalid initial state is not silently replaced and does not form a completed pair.']
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign-dir', type=Path, required=True)
    p.add_argument('--simroot', type=Path)
    p.add_argument('--progress-only', action='store_true')
    p.add_argument('--compact', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = audit_comparison(args.campaign_dir, simroot=args.simroot, progress_only=args.progress_only)
    serialized = json.dumps(result, indent=None if args.compact else 2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n')
    print(serialized)


if __name__ == '__main__':
    main()
