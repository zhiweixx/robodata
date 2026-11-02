#!/usr/bin/env python3
"""Freeze the 220 old Terra states for a fresh revised-harness Terra comparison.

Selection uses native score existence, never success or failure. The source's
incomplete 721-state campaign is deliberately not represented as complete.
"""
from __future__ import annotations
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

SOURCE_CAMPAIGN = 'terra_paired_721_20261031'
SOURCE_AUDIT_SHA256 = 'b86c81b5f1f1c45978fdea9d6d19d8a74329cca49723d15b207785e492708679'
SOURCE_MANIFEST_SHA256 = '00867b7cf459314e1feebf787b830bcdd1136acf827c17b4406ee0b8b7954171'
SOURCE_PAIRING_SHA256 = '61c1081103cdfb08e6d537090bef153ddccfa0d5e101cbebc4438a46594703ab'
TERMINAL_ALLOCATION = '28024072'
CAMERAS = ('head', 'left_wrist', 'right_wrist')
PAIRED_EPISODES = 220

class PairingError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise PairingError(message)

def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()

def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result

def read_json(path):
    path = Path(path)
    require(not path.is_symlink(), 'Evidence must not be a symlink')
    raw = path.read_bytes()
    def nonfinite(value):
        raise PairingError('Non-finite JSON: ' + value)
    value = json.loads(raw, object_pairs_hook=_object_pairs, parse_constant=nonfinite)
    require(isinstance(value, dict), 'JSON root must be object')
    return value, hashlib.sha256(raw).hexdigest()

def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None

def _index(rows, key, label):
    require(isinstance(rows, list), label + ' must be a list')
    result = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get(key), str)
            and row[key] not in result, label + ' duplicate/missing identity')
        result[row[key]] = row
    return result

def build_pairing(manifest, audit, source_pairing, *, hashes, paths, terminal_receipt):
    """Pure validator; production build_from_files pins actual source bytes."""
    require(hashes['manifest'] == SOURCE_MANIFEST_SHA256 and hashes['audit'] == SOURCE_AUDIT_SHA256,
        'Source must be the fixed user-authorized terminal Terra220 snapshot')
    require(manifest.get('campaign') == audit.get('campaign') == SOURCE_CAMPAIGN,
        'Source campaign mismatch')
    require(manifest.get('model', {}).get('served_alias') == 'gpt-6.1-terra'
        and manifest['model'].get('reasoning_effort') == 'medium', 'Wrong Terra source model')
    require(audit.get('evidence_audited') is True and audit.get('scored_episodes') == 220
        and audit.get('verified_episode_reports') == 220
        and audit.get('episodes_with_three_videos') == 220
        and audit.get('nonempty_video_count') == 660, 'Full native/report/video source audit required')
    require(all(x.get('severity') == 'warning' for x in audit.get('issues', [])),
        'Source integrity errors are forbidden')
    require(audit.get('planned_episodes') == 721 and audit.get('paired_721_complete_verified') is False,
        'Source721 incompleteness must remain explicit')
    for key, expected in [('source_inventory', 659), ('state_inventory', 721)]:
        obj = audit.get(key, {}); noun = 'files' if key == 'source_inventory' else 'states'
        require(obj.get(noun + '_expected') == obj.get(noun + '_verified') == expected,
            'Source inventory was not fully verified')
    require(audit.get('episode_queue', {}).get('verified') is True
        and audit.get('launch_gates', {}).get('verified') is True, 'Source queue/launch contract not verified')
    allocations = _index(audit.get('allocations'), 'allocation', 'allocations')
    require(TERMINAL_ALLOCATION in allocations and all(a.get('children_stopped') is True
        and a.get('finished_utc') for a in allocations.values()), 'Source children still active or cleanup unverified')
    require(not allocations[TERMINAL_ALLOCATION].get('cleanup_errors'), 'Terminal source cleanup failed')
    queue_allocations = _index(audit['episode_queue'].get('allocations'), 'allocation', 'queue allocations')
    require(set(queue_allocations) == set(allocations) and all(a.get('remaining_active_claims') == 0
        and a.get('claims') == a.get('finalized') for a in queue_allocations.values()), 'Active or unfinalized source claims')
    require(terminal_receipt.get('full_audit_sha256') == hashes['audit']
        and terminal_receipt.get('scored_episodes') == 220
        and terminal_receipt.get('verified_episode_reports') == 220
        and terminal_receipt.get('allocations') == [{k: a[k] for k in
            ('allocation','children_stopped','cleanup_errors','error','exit_code','finished_utc','started_utc')}
            for a in audit['allocations']], 'Terminal receipt disagrees with full audit')
    require(manifest.get('pairing_sha256') == source_pairing.get('pairing_sha256') == SOURCE_PAIRING_SHA256
        and hashes['pairing'] == manifest.get('pairing_file_sha256')
        and audit.get('pairing') == source_pairing, 'Original Qwen721 exact pairing is not bound')
    require(canonical_sha256(source_pairing['groups']) == SOURCE_PAIRING_SHA256, 'Original pairing content changed')
    profile = manifest.get('source_profile', {})
    require(len(profile.get('files_sha256', {})) == 536
        and profile.get('files_sha256') == source_pairing.get('source_profile', {}).get('files_sha256')
        and canonical_sha256(profile['files_sha256']) == profile.get('files_manifest_sha256'),
        'Frozen536 harness differs from Qwen source')
    original = _index(source_pairing.get('groups'), 'group_id', 'original paired groups')
    runtime = _index(manifest.get('groups'), 'group_id', 'Terra runtime groups')
    summaries = _index(audit.get('groups'), 'group_id', 'audit summaries')
    evidence = _index(audit.get('group_evidence'), 'group_id', 'native group evidence')
    require(set(original) == set(runtime) == set(summaries) == set(evidence), 'Source groups disagree')
    comparison = _index(audit.get('paired_comparison', {}).get('pairs'), 'episode_id', 'source paired outcomes')
    selected, all_ids, selected_success = [], [], 0
    for gid, src in original.items():
        native_group = runtime[gid]
        for key in ('task','family','variant','dimension','eval_seed','step_limit','native_options','states'):
            require(native_group.get(key) == src.get(key), 'Source runtime/native budget mismatch: ' + gid + '/' + key)
        items = _index(evidence[gid].get('item_evidence'), 'episode_id', 'native item evidence')
        require(set(items) == set(src['episode_ids']), 'Missing or unexpected native item evidence')
        state_by_id = {s['episode_id']:s for s in src['states']}
        states = []
        for eid in src['episode_ids']:
            item = items[eid]; native = item.get('native', {})
            details = native.get('details', {})
            if not details:
                require(native.get('scored', 0) == 0, 'Native scored count without details')
                continue
            require(native.get('scored') == 1 and list(details) == ['0'], 'Each state needs exactly native index0')
            row = details['0']; state = copy.deepcopy(state_by_id[eid]); lid = state['layout_id']
            require(row.get('layout_id') == lid and type(row.get('success')) is bool
                and type(row.get('score')) in (int,float) and math.isfinite(row['score']), 'Invalid native exact state/outcome')
            require(native.get('three_camera_episodes') == 1 and native.get('matching_video_count') == 3
                and _digest(native.get('sha256')), 'Missing native result/video proof')
            videos = native.get('videos', [])
            require(len(videos) == 3 and all(v.get('bytes', 0)>0 for v in videos)
                and {v.get('name') for v in videos} == {f'episode_0000000_cam_{camera}_{"success" if row["success"] else "fail"}.mp4' for camera in CAMERAS},
                'Native three camera videos are missing or mixed')
            reports = [r for r in item.get('episode_reports', []) if r.get('native_score_committed') is True]
            require(len(reports) == 1 and reports[0].get('episode_id') == eid
                and _digest(reports[0].get('sha256')), 'Exactly one committed native episode report required')
            pair = comparison.get(eid, {})
            require(pair.get('terra_success') is row['success'] and type(pair.get('qwen_success')) is bool
                and pair.get('semantic_state_sha256') == state['semantic_state_sha256'], 'Outcome/state pairing mismatch')
            require(item.get('source_group_id') == gid and item.get('missing_paired_layout_ids') == [],
                'Scored item incorrectly marked missing')
            state.update(terra_native_index=0, terra_success=row['success'], terra_score=row['score'],
                qwen_success=pair['qwen_success'], terra_native_result={'path':native['path'],'sha256':native['sha256']},
                terra_episode_report=copy.deepcopy(reports[0]), terra_videos=copy.deepcopy(videos))
            states.append(state); all_ids.append(eid); selected_success += int(row['success'])
        ids = [s['layout_id'] for s in states]
        require(ids == summaries[gid]['layout_ids'] and len(ids) == summaries[gid]['scored']
            == evidence[gid]['verified_report_count'], 'Native/detail/report counts disagree')
        require({str(i):{'layout_id':r['layout_id'], 'score':r['terra_score'],'success':r['terra_success']} for i,r in enumerate(states)}
            == evidence[gid]['native']['details'], 'Group native aggregation differs from scored item details')
        if states:
            group = copy.deepcopy(src)
            group.update(states=states, candidate_count=len(ids), candidate_layout_ids=ids,
                primary_layout_ids=ids, reserve_layout_ids=[], target_episodes=len(ids),
                runtime_overrides={'eval_nums':len(ids),'eval_layout_ids':ids},episode_ids=[s['episode_id'] for s in states],
                terra_source_group_sha256=canonical_sha256(native_group))
            selected.append(group)
    require(len(all_ids) == len(set(all_ids)) == 220 and selected_success == audit.get('successes') == 9,
        'Must select every one of220 scored states regardless of outcome')
    families = len({g['family'] for g in selected})
    gate = {'artifact_type':'terra_revised_authorized_terra220_subset_gate', 'subset_complete_verified':True,
        'paired_scored_episodes':220,'configuration_count':len(selected),'family_count':families,
        'verified_episode_reports':220,'verified_camera_videos':660,'source_full_target':721,
        'source_campaign_complete_verified':False,'original_full_target':807,'original807_complete_verified':False,
        'all_allocation_children_stopped':True,'source_full_audit_sha256':hashes['audit'],
        'source_manifest_sha256':hashes['manifest'],'terminal_allocation':TERMINAL_ALLOCATION,
        'preserved_source_audit_warnings':copy.deepcopy(audit.get('issues',[])),
        'unscored_are_model_failures':False}
    value = {'schema_version':1,'artifact_type':'terra_revised_exact_state_pairing','status':'frozen_from_terra220_terminal_audit',
        'planned_scored_episodes':220,'group_count':len(selected),'family_count':families,
        'episode_ids':all_ids,'groups':selected,'source_subset_gate':gate,
        'source':{'campaign':SOURCE_CAMPAIGN,'manifest_path':paths['manifest'],'audit_path':paths['audit'],
            'pairing_path':paths['pairing'],'terminal_receipt_path':paths['terminal_receipt'],
            'manifest_sha256':hashes['manifest'],'audit_sha256':hashes['audit'],'pairing_file_sha256':hashes['pairing'],
            'pairing_sha256':SOURCE_PAIRING_SHA256,'terminal_receipt_sha256':hashes['terminal_receipt'],
            'allocation_ids':list(allocations),'terminal_allocation':TERMINAL_ALLOCATION,
            'audit_timestamp':audit['audited_utc']},
        'source_qwen':copy.deepcopy(manifest['source_qwen']),
        'source_profile':copy.deepcopy(profile),'source_bootstrap_sha256':copy.deepcopy(manifest['bootstrap_sha256']),
        'source_model':copy.deepcopy(manifest['model']),
        'deferred_group_targets':copy.deepcopy(manifest['deferred_group_targets']),
        'user_held_group_shortfalls':copy.deepcopy(manifest['user_held_group_shortfalls']),
        'perception_artifacts':copy.deepcopy(manifest['perception_artifacts']),
        'grasp_calibration':copy.deepcopy(manifest['grasp_calibration']),
        'policy':{'selection':'all220 native-scored states in the fixed terminal Terra audit, independent of success',
            'invalid_state':'unscored missing pair; never replace','scored_state':'never replay',
            'source_unscored_excluded':501,'fresh_qwen_results':True,'new_model':'Qwen/Qwen3.8-27B','reasoning_effort':'medium'},
        'limitations':['Completion-selected220-state subset, not a balanced full-suite leaderboard score.',
            'Old Terra721 and original807 remain incomplete;501 other Terra states and86 original exclusions are outside this run; all new scores remain separate from the old220 baseline.']}
    value['pairing_sha256'] = canonical_sha256(selected)
    return value

def build_from_files(manifest_path, audit_path, pairing_path, *, terminal_receipt_path):
    paths={'manifest':manifest_path,'audit':audit_path,'pairing':pairing_path,'terminal_receipt':terminal_receipt_path}
    loaded={key:read_json(path) for key,path in paths.items()}
    return build_pairing(loaded['manifest'][0], loaded['audit'][0], loaded['pairing'][0],
        hashes={k:v[1] for k,v in loaded.items()}, paths={k:str(Path(v).resolve()) for k,v in paths.items()},
        terminal_receipt=loaded['terminal_receipt'][0])


def verify_native_artifacts(pairing):
    """Read-only verification of220 original result/report bytes and660 video sizes."""
    checked = []
    for group in pairing['groups']:
        for state in group['states']:
            native, report = state['terra_native_result'], state['terra_episode_report']
            n, digest = read_json(native['path'])
            require(digest == native['sha256'] and n.get('details', {}).get('0', {}).get('layout_id') == state['layout_id'],
                'Original native result bytes/state changed: ' + state['episode_id'])
            r, digest = read_json(report['path'])
            require(digest == report['sha256'] and r.get('native_score_committed') is True,
                'Original committed report changed: ' + state['episode_id'])
            for video in state['terra_videos']:
                path = Path(native['path']).parent / video['name']
                require(path.is_file() and not path.is_symlink() and path.stat().st_size == video['bytes'],
                    'Original camera video changed: ' + str(path))
            checked.append(state['episode_id'])
    require(len(checked)==220, 'Source native artifact verification incomplete')
    return {'verified':True,'episode_ids':checked,'native_results':220,'committed_reports':220,'nonempty_camera_videos':660,
        'source_audit_sha256':SOURCE_AUDIT_SHA256,'checked_original_files_without_copying_videos':True}
