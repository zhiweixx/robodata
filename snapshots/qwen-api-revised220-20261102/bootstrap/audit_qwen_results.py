#!/usr/bin/env python3
"""Read-only exact220 Qwen audit; native evidence plus all Chat Completions attempts.

Native/state/provider/queue helpers are preserved from the completed revised
Terra runner. Qwen stream checks do not invent OpenAI Responses replay or prices.
Progress mode does not claim a terminal artifact audit.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from harness_revision import validate_revision

MODEL = 'qwen38-27b'
PROTOCOL220 = 'qwen_paired220_task_reasoning_v2'
PROTOCOL = PROTOCOL220
CAMERAS = ('head','left_wrist','right_wrist')
VIDEO = re.compile(r'episode_([0-9]{7})_cam_(head|left_wrist|right_wrist)_(success|fail)\.mp4\Z')
IMPORTED_MODULES = {'full_entry':'qwen_full/full_entry.py','full_episode':'qwen_full/full_episode.py',
 'perception_entry':'qwen_full/perception_entry.py','perception_bridge':'qwen_integration/perception_bridge.py',
 'aspire_glue':'qwen_integration/aspire_glue.py','cap_harness.contracts':'ASPIRE/src/cap_harness/contracts.py',
 'full_adapter':'qwen_full/full_adapter.py','full_model_driver':'qwen_full/full_model_driver.py',
 'model_driver':'qwen_integration/model_driver.py','task_prompt_contract':'qwen_full/task_prompt_contract.py'}

def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def planned_episodes(manifest):
    return 220

def _campaign_manifest_sha(c):
    return sha_file(Path(c)/'manifest.json')

def sha_file(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

def issue(issues, path, message, severity='error'):
    issues.append({'path':str(path), 'message':message, 'severity':severity})

def _strict_loads(text):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result: raise ValueError('duplicate JSON key')
            result[k] = v
        return result
    def nonfinite(_): raise ValueError('nonfinite JSON number')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)

def read_object(path, issues, *, required=False):
    path = Path(path)
    if not path.exists():
        if required: issue(issues, path, 'Required evidence file is missing.')
        return None
    try:
        if path.is_symlink() or not path.is_file(): raise ValueError('unsafe evidence file')
        value = _strict_loads(path.read_text())
        if not isinstance(value, dict): raise ValueError('JSON object required')
        return value
    except (OSError, ValueError, UnicodeError) as exc:
        issue(issues, path, 'Unreadable evidence: '+type(exc).__name__)
        return None

def jsonl_rows(path, issues):
    try:
        with Path(path).open() as file:
            for n, line in enumerate(file, 1):
                try:
                    row = _strict_loads(line)
                    if not isinstance(row, dict): raise ValueError('not an object')
                    yield row
                except (ValueError, UnicodeError):
                    issue(issues, path, f'Malformed JSONL record {n}.')
    except OSError as exc:
        issue(issues, path, 'Unreadable JSONL: '+type(exc).__name__)

def expected_native(simroot, campaign, group):
    return (Path(simroot)/'RoboDojo/eval_result/RoboDojo'/group['task']/'ASPIRE_Pilot/arx_x5'
            /f"{group['eval_seed']}_aspire_terra_paired"/f"{campaign}_{group['group_id']}"/'_result.json')

def group_items(group):
    """Use the runtime's deterministic projection, then check its isolation contract.

    Source groups are an aggregation only. Every physical state gets a fresh
    native output and checkpoint with exactly one native detail, at index zero.
    """
    from terra_entry import episode_items
    items = episode_items({'groups': [group]})
    if len(items) != group['target_episodes']:
        raise ValueError('episode item count differs from paired group')
    seen = set()
    for item in items:
        ids = item.get('candidate_layout_ids')
        if not isinstance(ids, list) or len(ids) != 1:
            raise ValueError('episode item is not a single exact physical state')
        layout = ids[0]; eid = f"{group['group_id']}_l{layout}"
        if (type(layout) is not int or layout not in group['candidate_layout_ids'] or layout in seen
                or item.get('group_id') != eid or item.get('episode_id') != eid
                or item.get('source_group_id') != group['group_id']
                or item.get('target_episodes') != 1 or item.get('episode_ids') != [eid]
                or item.get('reserve_layout_ids') != []):
            raise ValueError('episode item identity/isolation differs from frozen pairing')
        seen.add(layout)
    return items

def merge_native_items(natives):
    """Aggregate verified native rows without pretending the aggregate is a native file."""
    result = {'aggregation': 'independent single-state native runs', 'scored': 0,
        'successes': 0, 'score_sum': 0.0, 'details': {}, 'matching_video_count': 0,
        'three_camera_episodes': 0, 'native_items': natives}
    for native in natives:
        for row in native['details'].values():
            result['details'][str(len(result['details']))] = row
        for key in ('scored', 'successes', 'score_sum', 'matching_video_count', 'three_camera_episodes'):
            result[key] += native[key]
    return result

def native_evidence(path, group, issues, *, videos=True):
    path = Path(path)
    result = {'path':str(path), 'scored':0, 'successes':0, 'score_sum':0.0, 'details':{},
              'matching_video_count':0, 'three_camera_episodes':0, 'videos':[]}
    native = read_object(path, issues)
    if native is None: return result
    details = native.get('details')
    if (not isinstance(details, dict) or set(details) != {str(i) for i in range(len(details))}
            or type(native.get('eval_time')) is not int or native['eval_time'] != len(details)
            or len(details) > group['target_episodes']):
        issue(issues,path,'Native count/index mapping is invalid.'); return result
    seen = set()
    for row in details.values():
        if not isinstance(row,dict):
            issue(issues,path,'Malformed native detail.'); return result
        layout, score = row.get('layout_id'), row.get('score')
        if (type(layout) is not int or layout not in group['candidate_layout_ids'] or layout in seen
                or type(row.get('success')) is not bool or type(score) not in (int,float)
                or not math.isfinite(score) or not 0 <= score <= 1):
            issue(issues,path,'Native score uses a duplicate, nonpaired or invalid state.'); return result
        seen.add(layout)
    result.update(details=details, scored=len(details), successes=sum(d['success'] for d in details.values()),
                  score_sum=sum(d['score'] for d in details.values()), sha256=sha_file(path))
    if not videos: return result
    views = set()
    for video in sorted(path.parent.glob('*.mp4')):
        match = VIDEO.fullmatch(video.name)
        if video.is_symlink() or match is None or video.stat().st_size <= 0:
            issue(issues,video,'Invalid or empty camera video.'); continue
        index, camera, outcome = str(int(match[1])), match[2], match[3]
        if (index not in details or (index,camera) in views
                or outcome != ('success' if details[index]['success'] else 'fail')):
            issue(issues,video,'Camera index/outcome differs from native score.'); continue
        views.add((index,camera)); result['videos'].append({'name':video.name,'bytes':video.stat().st_size})
    result.update(matching_video_count=len(views),
        three_camera_episodes=sum(all((i,c) in views for c in CAMERAS) for i in details))
    return result

def progress_group(simroot, campaign, group, issues):
    native = merge_native_items([native_evidence(expected_native(simroot,campaign,item),item,issues,videos=False)
        for item in group_items(group)])
    ids = [r['layout_id'] for r in native['details'].values()]
    return {'group_id':group['group_id'],'family':group['family'],'dimension':group['dimension'],
            'variant':group['variant'],'target':group['target_episodes'],'scored':native['scored'],
            'successes':native['successes'],'complete':native['scored']==group['target_episodes'],
            'layout_ids':ids,'native_paths':[n['path'] for n in native['native_items']],
            'unscored_layout_ids':sorted(set(group['candidate_layout_ids'])-set(ids))}

def aggregate(rows):
    families, categories = {}, {}
    for row in rows:
        for table,key in ((families,row['family']),(categories,row['dimension'])):
            item=table.setdefault(key,{'target':0,'scored':0,'successes':0})
            for field in ('target','scored','successes'): item[field] += row[field]
    for table in (families,categories):
        for item in table.values():
            item['success_rate']=item['successes']/item['scored'] if item['scored'] else None
            item['complete']=item['scored']==item['target']
    return families,categories

def verify_inventory(root, manifest, issues):
    root=Path(root); verified=0; expected=0
    inventories=((root/manifest['source_profile']['relative_root'],manifest['source_profile']['files_sha256']),
                 (root/manifest.get('bootstrap_relative_root','terra_robodojo'),manifest['bootstrap_sha256']),
                 (Path(manifest['source']['robodojo_root']),manifest['source']['sha256']))
    profile_base=inventories[0][0]
    profile_files=set()
    for path in profile_base.rglob('*'):
        rel=path.relative_to(profile_base)
        if any(part in ('.git','__pycache__') for part in rel.parts):continue
        if path.is_symlink():issue(issues,path,'Frozen source profile contains a symlink.')
        if path.is_file():profile_files.add(str(rel))
    if profile_files!=set(inventories[0][1]):issue(issues,profile_base,'Frozen source file set differs, including unexpected files.')
    for base,files in inventories:
        expected += len(files)
        for relative,digest in files.items():
            path=base/relative
            if (Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink()
                    or not path.resolve().is_relative_to(base.resolve())):
                issue(issues,path,'Unsafe source inventory path.'); continue
            if not path.is_file() or sha_file(path)!=digest:
                issue(issues,path,'Frozen source digest differs.'); continue
            verified += 1
    return {'files_expected':expected,'files_verified':verified,'large_weights_rehashed':False}

def verify_state_files(manifest, issues):
    checked=0
    for group in manifest['groups']:
        seen=set()
        for state in group['states']:
            path=Path(state['source_layout']['source_path'])
            try:
                if path.is_symlink(): raise ValueError('symlink')
                raw=path.read_bytes(); obj=_strict_loads(raw); semantic=copy.deepcopy(obj)
                if isinstance(semantic,dict) and isinstance(semantic.get('Room'),dict):
                    semantic['Room'].pop('default_rot',None)
                    semantic['Room'].setdefault('default_ori',[1.0,0.0,0.0,0.0])
                content=canonical_sha256(semantic)
                if (hashlib.sha256(raw).hexdigest()!=state['source_sha256']
                        or canonical_sha256(obj)!=state['canonical_json_sha256']
                        or content!=state['semantic_state_sha256'] or content in seen):
                    raise ValueError('physical/source content differs')
                seen.add(content);checked+=1
            except (OSError,ValueError,KeyError,TypeError) as exc:
                issue(issues,path,'Paired source state mismatch: '+type(exc).__name__)
    return {'states_verified':checked,'states_expected':planned_episodes(manifest)}

def verify_contract(contract, campaign_dir, manifest, group, issues, manifest_sha256=None):
    profile=Path('/work/aspire_validate')/manifest['source_profile']['relative_root']
    imports={name:str(profile/relative) for name,relative in IMPORTED_MODULES.items()}
    if (not contract or contract.get('protocol')!=manifest.get('protocol',PROTOCOL) or contract.get('campaign')!=manifest['campaign']
            or contract.get('group')!=group or contract.get('manifest_sha256')!=(manifest_sha256 or _campaign_manifest_sha(campaign_dir))
            or contract.get('source_profile')!=manifest['source_profile']
            or contract.get('harness_revision')!=manifest.get('harness_revision') or contract.get('remote_model')!=manifest['model']
            or contract.get('imported_sources')!=imports or contract.get('fresh_campaign') is not True
            or contract.get('is_official_full_evaluation') is not False
            or contract.get('offline_skill_discovery') is not False
            or contract.get('episode_id')!=group['episode_id']
            or contract.get('source_group_id')!=group['source_group_id']
            or contract.get('lifecycle')!='fresh native process per exact paired physical state'):
        issue(issues,Path(campaign_dir)/'items'/group['group_id'],'Scored episode item lacks matching execution/import contract.')

def verify_item(campaign_dir, simroot, manifest, group, allocations, issues, infrastructure_lineage=None):
    c=Path(campaign_dir);gid=group['group_id'];out=c/'items'/gid
    item_manifest=manifest;item_manifest_sha=_campaign_manifest_sha(c)
    native=native_evidence(expected_native(simroot,manifest['campaign'],group),group,issues)
    bylayout={r['layout_id']:r for r in native['details'].values()}
    report=read_object(out/'group_report.json',issues)
    if native['scored']:
        if (not report or report.get('native_run_id')!=manifest['campaign']+'_'+gid
                or report.get('scored')!=native['scored'] or report.get('task')!=group['task']
                or report.get('eval_seed')!=group['eval_seed'] or report.get('mode')!='evaluation'
                or report.get('target')!=1 or report.get('complete') is not True):
            issue(issues,out,'Episode item report identity/count differs from native result.')
        contract=read_object(out/'terra_contract.json',issues,required=True)
        bound=next((b for b in (infrastructure_lineage or {}).get('allocations',{}).values()
                    if b.get('manifest_sha256')==(contract or {}).get('manifest_sha256')),None)
        if bound:
            item_manifest=bound['manifest'];item_manifest_sha=bound['manifest_sha256']
        verify_contract(contract,c,item_manifest,group,issues,item_manifest_sha)
    verified=set();calls=Counter();terminations=Counter();reports=[];episode_seconds=0.0
    for path in sorted(out.glob('episodes/*/attempt_*/episode_report.json')):
        episode=read_object(path,issues,required=True)
        if not episode:continue
        layout=episode.get('layout_id');eid=episode.get('episode_id')
        if (type(layout) is not int or layout not in group['candidate_layout_ids']
                or eid!=group['episode_id'] or path.parent.parent.name!=eid
                or episode.get('task')!=group['task'] or episode.get('eval_seed')!=group['eval_seed']):
            issue(issues,path,'Episode outside its frozen single-state native item.');continue
        terminations[str(episode.get('termination_reason'))]+=1
        for iteration in episode.get('iterations',[]):
            for call in (iteration.get('execution') or {}).get('tool_calls',[]):calls[str(call.get('tool'))]+=1
        reports.append({'path':str(path),'sha256':sha_file(path),'episode_id':eid,
                        'model_iterations':len(episode.get('iterations',[])),
                        'native_score_committed':episode.get('native_score_committed') is True})
        if episode.get('native_score_committed') is not True:continue
        official=episode.get('official',{});runtime=episode.get('perception_runtime',{})
        if (layout in verified or layout not in bylayout or episode.get('status')!='completed'
                or not valid_episode_limits(episode,group,item_manifest)
                or official.get('official_success')!=bylayout[layout]['success']
                or type(official.get('official_success')) is not bool):
            issue(issues,path,'Committed report score/model/budget identity mismatch.');continue
        if (runtime.get('initialization_complete') is not True
                or runtime.get('providers')!={'sam3':True,'contact_graspnet':True}
                or runtime.get('provider_health')!={'sam3':True,'contact_graspnet':True}
                or runtime.get('offline_skill_discovery') is not False
                or runtime.get('service_provenance',{}).get('runtime_manifest_sha256')
                    !=manifest['perception_artifacts']['runtime_manifest_sha256']):
            issue(issues,path,'Actual perception provider provenance is not verified.');continue
        timing=episode.get('terra_timing',{})
        try:
            seconds=duration_seconds(timing['episode_started_utc'],timing['episode_callback_finished_utc'])
            episode_seconds+=seconds
        except (KeyError,ValueError,TypeError):issue(issues,path,'Committed report lacks valid episode timing.')
        verified.add(layout)
    if set(bylayout)!=verified:issue(issues,out,'Native scores and unique committed episode reports differ.')
    if native['three_camera_episodes']!=native['scored']:
        issue(issues,out,'Some native scores lack three nonempty outcome-matched videos.')
    usages=[]
    for directory,allocation in allocations:
        for attempt in sorted(Path(directory).glob(f'workers/*/{gid}')):
            if attempt.is_dir():
                binding=(infrastructure_lineage or {}).get('allocations',{}).get(Path(directory).name)
                bound=binding['manifest'] if binding else manifest
                usages.append(audit_attempt_usage(attempt,allocation,group['episode_ids'],bound,issues))
    if native['scored'] and not usages:issue(issues,out,'Scored episode item has no raw usage ledger.')
    gpu=auxiliary=None
    if native['scored'] and manifest.get('planned_scored_episodes')==220:
        from terra_platform import verify_item_gpu_isolation, validate_platform_evidence
        gpu=verify_item_gpu_isolation(c,group,report,reports,allocations,issues)
        owner=(gpu or {}).get('allocation')
        if not owner:
            issue(issues,out/'terra_contract.json','Scored state has no uniquely verified simulator allocation.')
        else:
            try:
                auxiliary=validate_platform_evidence(c,item_manifest,group,owner,
                    manifest_sha256=item_manifest_sha)
            except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
                issue(issues,out/'kit_auxiliary_policy.json','Simulator platform evidence rejected: '+str(exc))
    streams={}
    for usage in usages:
        for stream in usage.get('raw_streams',[]):
            rid=stream.get('response_id')
            if rid:
                if rid in streams:issue(issues,out,'Provider response identity is reused across requests.')
                streams[rid]=stream
    for saved in reports:
        if not saved['native_score_committed']:continue
        path=Path(saved['path']);episode=read_object(path,issues,required=True)
        for iteration in (episode or {}).get('iterations',[]):
            rid=iteration.get('usage',{}).get('response_id');stream=streams.get(rid,{})
            valid=(stream.get('proposal_schema_verified') is True and stream.get('stream_complete') is True
                and stream.get('finish_reasons')==['stop']
                and stream.get('model')==item_manifest['model']['served_alias'] and stream.get('http_status')==200
                and stream.get('proposal_give_up')==iteration.get('give_up')
                and isinstance(iteration.get('summary'),str)
                and stream.get('proposal_summary_sha256')==hashlib.sha256(iteration['summary'].encode()).hexdigest())
            source=iteration.get('source')
            if iteration.get('give_up') is False:
                program=path.parent/str(source)
                valid=valid and isinstance(source,str) and re.fullmatch(r'program_[0-9]{4,}\.py',source) is not None
                valid=valid and not program.is_symlink() and program.is_file() and sha_file(program)==stream.get('proposal_code_sha256')
            if not valid:issue(issues,path,'Accepted robot program lacks matching complete raw Qwen response/schema/source.')
    return {'episode_id':gid,'source_group_id':group['source_group_id'],'native':native,
        'gpu_isolation':gpu,'kit_auxiliary_policy':auxiliary,
        'scored_provider':item_manifest['model']['provider'] if native['scored'] else None,
        'scored_model':item_manifest['model']['served_alias'] if native['scored'] else None,
        'verified_report_count':len(verified),'usage':usages,
        'tool_call_counts':dict(calls),'termination_reasons':dict(terminations),'episode_reports':reports,
        'summed_scored_episode_wall_seconds':episode_seconds,
        'missing_paired_layout_ids':sorted(set(group['candidate_layout_ids'])-set(bylayout))}

def verify_group(campaign_dir, simroot, manifest, group, allocations, issues, infrastructure_lineage=None):
    """Report source-config summaries, preserving each original native artifact."""
    items=[verify_item(campaign_dir,simroot,manifest,item,allocations,issues,infrastructure_lineage) for item in group_items(group)]
    calls=Counter();terminations=Counter()
    for item in items:
        calls.update(item['tool_call_counts']);terminations.update(item['termination_reasons'])
    return {'group_id':group['group_id'],'native':merge_native_items([i['native'] for i in items]),
        'item_evidence':items,'verified_report_count':sum(i['verified_report_count'] for i in items),
        'usage':[u for i in items for u in i['usage']],
        'tool_call_counts':dict(calls),'termination_reasons':dict(terminations),
        'episode_reports':[r for i in items for r in i['episode_reports']],
        'summed_scored_episode_wall_seconds':sum(i['summed_scored_episode_wall_seconds'] for i in items),
        'missing_paired_layout_ids':[layout for i in items for layout in i['missing_paired_layout_ids']]}

def duration_seconds(start,end):
    first=datetime.fromisoformat(start.replace('Z','+00:00'))
    last=datetime.fromisoformat(end.replace('Z','+00:00'))
    if first.tzinfo is None or last.tzinfo is None or last<first:raise ValueError('invalid chronology')
    return (last-first).total_seconds()

def verify_episode_queue(c,manifest,allocations,group_evidence,issues,prepaid_lineage=None):
    """Replay claims independently of the scheduler's final completion summary.

    A later allocation can claim an interrupted, unscored state. It may never
    claim a completed state, assign one item to two workers, or reuse a native
    output directory. Scored failures have exactly the same skip protection as
    scored successes.
    """
    c=Path(c);before=sum(i['severity']=='error' for i in issues)
    items={i['episode_id']:i for g in manifest['groups'] for i in group_items(g)}
    first_episode=group_items(manifest['groups'][0])[0]['episode_id']
    expected=set(items);scored={i['episode_id'] for g in group_evidence for i in g['item_evidence'] if i['native']['scored']==1}
    native_paths=[i['native']['path'] for g in group_evidence for i in g['item_evidence']]
    if len(native_paths)!=len(set(native_paths)) or len(native_paths)!=len(items):
        issue(issues,c/'items','Episode native paths are not one-to-one with exact states.')
    for path in (c/'items').glob('*'):
        if path.is_symlink() or not path.is_dir() or path.name not in expected:
            issue(issues,path,'Unexpected or unsafe episode output directory.')
    complete_before=set();scored_before=set();attempted_before=set();claimed_total=0;max_busy=0;attempt_count=0
    startup=active=shutdown=lifecycle=0.0;timed_attempts=0;partial_timing_attempts=0;allocation_evidence=[]
    allowed_states={'pending','running','complete','missing_pairs','financial_paused','interrupted'}
    prior=None
    for directory,a in allocations:
        old=(prepaid_lineage or {}).get('allocations',{}).get(directory.name)
        bound_manifest_hash=old['manifest_sha256'] if old else _campaign_manifest_sha(c)
        closed=bool(a.get('finished_utc'))
        if prior and a.get('started_utc'):
            try:
                if not prior.get('finished_utc'):raise ValueError('previous allocation not closed')
                duration_seconds(prior['finished_utc'],a['started_utc'])
            except (ValueError,TypeError,KeyError):issue(issues,directory,'Allocation lifetimes overlap or are not verifiably ordered.')
        prior=a
        schedule=read_object(directory/'scheduler_state.json',issues,required=True) or {}
        states=schedule.get('items',{})
        if not isinstance(states,dict):
            issue(issues,directory/'scheduler_state.json','Scheduler items must be an object.');states={}
        if (schedule.get('campaign')!=manifest['campaign'] or str(schedule.get('allocation'))!=directory.name
                or schedule.get('manifest_sha256')!=bound_manifest_hash or set(states)!=expected):
            issue(issues,directory/'scheduler_state.json','Episode scheduler identity or exact-state item set differs.')
        for eid,row in states.items():
            if not isinstance(row,dict) or row.get('state') not in allowed_states:
                issue(issues,directory/'scheduler_state.json','Invalid scheduler item state: '+str(eid));continue
            if row['state']=='complete' and eid not in scored:
                issue(issues,directory/'scheduler_state.json','Scheduler marks an unscored item complete: '+str(eid))
        claims={};started=set();finalized={};busy={};max_here=0;api_gate=False
        events=directory/'events.jsonl'
        if not events.exists() and (directory/'workers').exists():issue(issues,events,'Worker attempts lack durable claim events.')
        for event in jsonl_rows(events,issues) if events.exists() else []:
            kind=event.get('event')
            if kind=='initial_api_compatibility_verified':
                if (api_gate or event.get('episode_id')!=first_episode
                        or event.get('quality_threshold_applied') is not False):
                    issue(issues,events,'Initial API gate event identity or quality condition differs.')
                api_gate=True;continue
            if kind not in ('episode_claimed','episode_started','episode_finalized'):continue
            eid=event.get('group_id');worker=event.get('worker_id')
            if eid not in expected or worker not in {f'w{i:02d}' for i in range(9)}:
                issue(issues,events,'Queue event has an unknown item or worker.');continue
            if kind=='episode_claimed':
                if eid!=first_episode and not api_gate:
                    issue(issues,events,'Another state was admitted before first-episode API compatibility verification.')
                if eid in claims or eid in scored_before:
                    issue(issues,events,'Duplicate claim or replay of an already scored item: '+eid)
                if worker in busy:issue(issues,events,'Worker claimed a second item before finalizing the first.')
                if event.get('gpu_slot')!=[0,0,1,1,2,2,3,3,3][int(worker[1:])]:
                    issue(issues,events,'Episode claim GPU slot differs from frozen topology.')
                claims[eid]=worker;busy[worker]=eid;claimed_total+=1;max_here=max(max_here,len(busy))
            elif kind=='episode_started':
                if claims.get(eid)!=worker or busy.get(worker)!=eid or eid in started:
                    issue(issues,events,'Simulator started without a unique preceding item claim.')
                if event.get('target_episodes')!=1:issue(issues,events,'Simulator item changed the single-state target.')
                started.add(eid)
            else:
                if claims.get(eid)!=worker or busy.get(worker)!=eid or eid in finalized:
                    issue(issues,events,'Episode finalized without matching unique ownership.')
                finalized[eid]=event.get('report');busy.pop(worker,None)
        if closed and busy:issue(issues,events,'Closed allocation has claims without final lifecycle receipts.')
        attempts={}
        for path in sorted(directory.glob('workers/*/*')):
            if not path.is_dir():continue
            if path.is_symlink():issue(issues,path,'Episode attempt directory must not be a symlink.')
            eid=path.name;worker=path.parent.name;attempt_count+=1
            if eid not in expected:continue  # Other audit path retains unexpected usage.
            if eid in attempts:issue(issues,path,'Same exact state attempted twice within one allocation.')
            attempts[eid]=worker
            if claims.get(eid)!=worker:issue(issues,path,'Attempt directory lacks its unique durable claim.')
            row=read_object(path/'group_attempt.json',issues,required=closed)
            if not row:continue
            item=items[eid]
            if (row.get('episode_id')!=eid or row.get('source_group_id')!=item['source_group_id']
                    or row.get('group_id')!=eid or row.get('worker_id')!=worker
                    or row.get('target_episodes')!=1 or row.get('eligible_layout_ids')!=item['candidate_layout_ids']
                    or row.get('task')!=item['task'] or row.get('eval_seed')!=item['eval_seed']):
                issue(issues,path,'Episode lifecycle receipt differs from frozen exact-state identity.')
            if eid in finalized and finalized[eid]!=row:
                issue(issues,path,'Finalized event and saved episode lifecycle receipt differ.')
            if closed and eid not in finalized:issue(issues,path,'Closed attempt lacks a finalized event.')
            if row.get('complete') is True:
                if eid not in scored or not row.get('infrastructure_complete') or row.get('simulator_exit_code')!=0:
                    issue(issues,path,'Episode completion lacks a scored state and clean lifecycle.')
            timing=row.get('episode_timing')
            if timing:
                try:
                    if (timing.get('episode_id')!=eid or timing.get('source_group_id')!=item['source_group_id']
                            or timing.get('worker_id') not in (None,worker)):raise ValueError('timing identity')
                    start=duration_seconds(row['simulator_started_utc'],timing['episode_started_utc'])
                    expected_timing=[('simulator_startup_wall_seconds',start)]
                    if timing.get('episode_callback_finished_utc') is not None:
                        if timing.get('worker_id')!=worker:raise ValueError('terminal timing owner missing')
                        middle=duration_seconds(timing['episode_started_utc'],timing['episode_callback_finished_utc'])
                        end=duration_seconds(timing['episode_callback_finished_utc'],row['simulator_finished_utc'])
                        expected_timing.extend([('episode_callback_wall_seconds',middle),('simulator_post_episode_wall_seconds',end)])
                    elif row.get('complete'):raise ValueError('complete episode timing missing end')
                    for key,value in expected_timing:
                        if type(row.get(key)) not in (int,float) or abs(row[key]-value)>0.001:
                            raise ValueError('timing duration differs')
                    startup+=start
                    if timing.get('episode_callback_finished_utc') is None:partial_timing_attempts+=1
                    else:active+=middle;shutdown+=end;timed_attempts+=1
                except (KeyError,ValueError,TypeError):issue(issues,path,'Episode startup/active/shutdown timing is invalid.')
            if row.get('complete') and not timing:issue(issues,path,'Complete episode lacks startup/active/shutdown timing.')
            wall=row.get('lifecycle_wall_seconds')
            if type(wall) not in (int,float) or not math.isfinite(wall) or wall<0:
                issue(issues,path,'Episode lifecycle wall time is invalid.')
            else:lifecycle+=wall
        if closed and set(attempts)!=set(claims):issue(issues,directory,'Closed allocation claim and attempt sets differ.')
        reported=a.get('episode_attempts',[])
        if (not isinstance(reported,list) or len(reported)!=len({r.get('episode_id') for r in reported if isinstance(r,dict)})
                or closed and {r.get('episode_id') for r in reported if isinstance(r,dict)}!=set(finalized)):
            issue(issues,directory,'Allocation episode-attempt summary has duplicate or missing items.')
        completed=a.get('completed_items',{})
        if not isinstance(completed,dict):
            issue(issues,directory,'Completed-item summary must be an object.');completed={}
        complete=set(completed)
        if not complete<=scored or not complete_before<=complete:
            issue(issues,directory,'Completed-item summary loses prior results or invents a score.')
        for eid in complete-complete_before:
            current_receipt=isinstance(finalized.get(eid),dict) and finalized[eid].get('complete') is True
            prior_validation=states.get(eid,{}).get('validated_prior_result',{})
            verified_skip=(eid in attempted_before and isinstance(prior_validation,dict)
                and prior_validation.get('state')=='complete' and prior_validation.get('validation',{}).get('scored')==1
                and eid in scored and eid not in claims)
            if not current_receipt and not verified_skip:
                issue(issues,directory,'New completed item lacks this allocation\'s successful final receipt.')
        complete_before|=complete;scored_before|=complete;attempted_before|=set(attempts)
        # A simulator may commit a score and then fail during shutdown. Native
        # results must still be skipped on resume, even if the allocation could
        # not mark that item as infrastructure-complete.
        scored_before.update(eid for eid,row in finalized.items() if isinstance(row,dict)
            and row.get('native_state',{}).get('state')=='complete' and eid in scored)
        max_busy=max(max_busy,max_here)
        allocation_evidence.append({'allocation':directory.name,'claims':len(claims),'started':len(started),
            'finalized':len(finalized),'max_simultaneous_claims':max_here,'remaining_active_claims':len(busy)})
    return {'verified':sum(i['severity']=='error' for i in issues)==before,
        'planned_items':len(items),'claims':claimed_total,'attempts':attempt_count,'allocations':allocation_evidence,
        'max_simultaneous_claims':max_busy,'timed_attempts':timed_attempts,
        'partial_timing_attempts':partial_timing_attempts,
        'summed_simulator_startup_wall_seconds':startup,'summed_episode_callback_wall_seconds':active,
        'summed_simulator_post_episode_wall_seconds':shutdown,'summed_item_lifecycle_wall_seconds':lifecycle,
        'simultaneous_claims_are_not_proof_of_simultaneous_gpu_execution':True}

def paired220_outcomes(pairing,group_evidence,issues):
    current={g['group_id']:{d['layout_id']:d for d in g['native']['details'].values()} for g in group_evidence}
    rows=[];counts=Counter();families={};categories={}
    for group in pairing['groups']:
        gid=group['group_id']
        for state in group['states']:
            observed=current.get(gid,{}).get(state['layout_id']);baseline=state['terra_success']
            if type(baseline) is not bool:issue(issues,gid,'Baseline Terra success must be a native boolean.');continue
            outcome=('missing_pair' if observed is None else 'both_success' if baseline and observed['success']
                else 'revised_only_success' if observed['success'] else 'baseline_only_success' if baseline else 'both_failure')
            counts[outcome]+=1
            rows.append({'episode_id':state['episode_id'],'family':group['family'],'dimension':group['dimension'],
                'layout_id':state['layout_id'],'semantic_state_sha256':state['semantic_state_sha256'],
                'baseline_terra_success':baseline,'revised_qwen_success':observed['success'] if observed else None,'outcome':outcome})
            for table,key in ((families,group['family']),(categories,group['dimension'])):
                row=table.setdefault(key,{'selected_states':0,'completed_pairs':0,'baseline_successes_all_selected':0,
                    'baseline_successes_completed_pairs':0,'revised_successes':0,'missing_pairs':0})
                row['selected_states']+=1;row['baseline_successes_all_selected']+=int(baseline)
                row['missing_pairs']+=int(observed is None)
                if observed is not None:
                    row['completed_pairs']+=1;row['baseline_successes_completed_pairs']+=int(baseline)
                    row['revised_successes']+=int(observed['success'])
    for table in (families,categories):
        for row in table.values():
            n=row['completed_pairs'];row['baseline_success_rate']=row['baseline_successes_completed_pairs']/n if n else None
            row['revised_success_rate']=row['revised_successes']/n if n else None
            row['success_rate_difference']=(row['revised_successes']-row['baseline_successes_completed_pairs'])/n if n else None
    complete=[r for r in rows if r['revised_qwen_success'] is not None];n=len(complete)
    baseline=sum(r['baseline_terra_success'] for r in complete);revised=sum(r['revised_qwen_success'] for r in complete)
    return {'pairs':rows,'counts':dict(counts),'families':families,'categories':categories,
        'baseline':'original Terra medium frozen harness','comparison':'revised Qwen3.8-27B task harness',
        'missing_pairs_are_failures':False,'comparisons_are_exploratory':True,'denominator_completed_pairs':n,
        'baseline_successes_all_selected':sum(r['baseline_terra_success'] for r in rows),
        'baseline_successes_completed_pairs':baseline,'revised_successes':revised,
        'baseline_success_rate':baseline/n if n else None,'revised_success_rate':revised/n if n else None,
        'success_rate_difference':(revised-baseline)/n if n else None}


def valid_episode_limits(episode, group, manifest):
    from terra_entry import LIMITS
    official=episode.get('official',{});steps=episode.get('environment_steps')
    return (episode.get('proposal_schema')=='aspire_program_give_up_v2'
        and episode.get('task_prompt_contract',{}).get('version')=='robodojo_public_task_contract_v2'
        and episode.get('step_limit')==group['step_limit'] and type(steps) is int and 0<=steps<=group['step_limit']
        and official.get('environment_steps')==steps and official.get('step_limit')==group['step_limit']
        and episode.get('limits')==LIMITS and episode.get('remote_model')==manifest['model']
        and episode.get('model')==manifest['model']['served_alias'] and episode.get('fresh_campaign') is True
        and episode.get('is_official_full_evaluation') is False)


def verify_infrastructure_phases(c, manifest, allocations, issues):
    before=len(issues)
    try:
        validate_revision(manifest)
        from provider_transition import verify_transition
        return verify_transition(c,manifest)
    except (OSError,KeyError,TypeError,ValueError) as exc:
        issue(issues,Path(c)/'manifest.json','Provider transition rejected: '+str(exc))
    return {'verified':False,'phases':[],'allocations':{},'billing_mode':'user_managed'}


def pairing_manifest_for_preflight(manifest, infrastructure_lineage=None):
    validate_revision(manifest)
    return manifest


def verify_pairing(c,manifest,issues):
    try:
        from prepare_terra import verify_pairing_artifacts
        return verify_pairing_artifacts(c,manifest)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        issue(issues,Path(c)/'paired_states.json','Exact-state baseline binding failed: '+str(exc))
        return None


def valid_tokens(value):
    return (isinstance(value,dict) and type(value.get('prompt_tokens')) is int
        and type(value.get('completion_tokens')) is int
        and min(value['prompt_tokens'],value['completion_tokens'])>=0)


def audit_stream(path,issues,expected_model=MODEL):
    """Reconstruct accepted JSON proposal from Chat Completions; retain no text in summary."""
    path=Path(path);models=set();finishes=set();ids=set();content=[];usage=None;done=False
    invalid=0;errors=False;events=0;digest=hashlib.sha256()
    with path.open('rb') as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.startswith(b'data:'):continue
            data=raw[5:].strip()
            if data==b'[DONE]':done=True;continue
            try:
                event=_strict_loads(data)
                if not isinstance(event,dict):raise ValueError()
            except (ValueError,UnicodeError):invalid+=1;continue
            events+=1
            if isinstance(event.get('model'),str):models.add(event['model'])
            if isinstance(event.get('id'),str):ids.add(event['id'])
            errors|=event.get('error') is not None
            if valid_tokens(event.get('usage')):
                observed={k:event['usage'][k] for k in ('prompt_tokens','completion_tokens')}
                if usage is not None and usage!=observed:issue(issues,path,'Different final usage counters in one stream.')
                usage=observed
            for choice in event.get('choices',[]):
                if not isinstance(choice,dict):invalid+=1;continue
                reason=choice.get('finish_reason')
                if isinstance(reason,str):finishes.add(reason)
                delta=choice.get('delta',{})
                if isinstance(delta,dict) and isinstance(delta.get('content'),str):content.append(delta['content'])
    headers=read_object(path.with_suffix('.headers.json'),issues) or {}
    receipt=read_object(path.with_suffix('.receipt.json'),issues) or {}
    result={'path':str(path),'bytes':path.stat().st_size,'sha256':digest.hexdigest(),
        'done_marker':done,'provider_usage':usage,'finish_reasons':sorted(finishes),'model_aliases':sorted(models),
        'model':next(iter(models)) if len(models)==1 else None,'response_id':next(iter(ids)) if len(ids)==1 else None,
        'json_events':events,'invalid_data_events':invalid,'provider_error_event':errors,
        'http_status':headers.get('http_status'),'headers_retained':bool(headers),
        'request_sha256':headers.get('request_sha256'),'sent_model':headers.get('sent_model'),
        'stream_complete':done and invalid==0 and not errors and headers.get('http_status')==200,
        'proposal_schema_verified':False,'receipt_retained':bool(receipt),
        'time_to_first_token_seconds':receipt.get('time_to_first_token_seconds'),
        'request_wall_seconds':receipt.get('wall_seconds')}
    if models and models!={expected_model}:issue(issues,path,'Raw stream model differs from the pinned Qwen alias.')
    if len(ids)>1:issue(issues,path,'One request has multiple response identities.')
    if result['stream_complete'] and finishes=={'stop'}:
        try:
            proposal=_strict_loads(''.join(content))
            if (not isinstance(proposal,dict) or set(proposal)!={'code','summary','give_up'}
                    or not isinstance(proposal['code'],str) or not isinstance(proposal['summary'],str)
                    or type(proposal['give_up']) is not bool
                    or '```' in proposal['code']
                    or (proposal['give_up'] and (proposal['code'].strip() or not proposal['summary'].strip()))
                    or (not proposal['give_up'] and not proposal['code'].strip())):raise ValueError()
            result.update(proposal_schema_verified=True,proposal_give_up=proposal['give_up'],
                proposal_code_sha256=hashlib.sha256(proposal['code'].encode()).hexdigest(),
                proposal_summary_sha256=hashlib.sha256(proposal['summary'].encode()).hexdigest())
        except (ValueError,TypeError):pass  # Invalid program may legitimately end one native episode.
    return result


def request_key(row):
    return (row.get('episode_id'),row.get('attempt',1),row.get('iteration'))


def audit_attempt_usage(attempt,allocation,expected_ids,manifest,issues):
    expected_model=manifest['model']['served_alias']
    attempt=Path(attempt);path=attempt/'usage.jsonl';starts=[];known={};inputs={};failures=set();counts=Counter()
    result={'path':str(path),'allocation':str(allocation.get('allocation',attempt.parents[1].name)),
        'worker_id':attempt.parent.name,'raw_streams':[],'requests':{},'input_tokens':0,'output_tokens':0,
        'reasoning_tokens_reported':0,'reasoning_counters_missing':0,'summed_request_wall_seconds':0.0,
        'ttft_sum_seconds':0.0,'requests_with_ttft':0,'raw_archive_bytes':0}
    initialized=0
    if path.exists():
        for row in jsonl_rows(path,issues):
            event=row.get('event');counts[str(event)]+=1;key=request_key(row)
            if event=='suite_initialized':
                initialized+=1;config=row.get('config',{})
                expected={'model':expected_model,'max_iterations':None,'independent_max_output_tokens':None,
                    'proposal_schema':'aspire_program_give_up_v2','task_contract':'robodojo_public_task_contract_v2',
                    'provider':manifest['model']['provider'],'base_url':manifest['model']['base_url'],
                    'max_input_tokens':None,'expected_context_length':262144,'reasoning_effort':'medium',
                    'socket_idle_timeout_s':None,'absolute_generation_timeout_s':None,'automatic_retries':0,
                    'history_policy':'all public messages and images; no rolling truncation; prior reasoning omitted'}
                if any(config.get(k)!=v for k,v in expected.items()):issue(issues,path,'Qwen driver configuration differs.')
                if set(config.get('episode_ids',[]))!=set(expected_ids):issue(issues,path,'Controller episode identity differs.')
                if config.get('driver_sha256')!=manifest['bootstrap_sha256'].get('qwen_driver.py'):
                    issue(issues,path,'Actual Qwen driver source differs from frozen bytes.')
                if config.get('base_driver_sha256')!=manifest['source_profile']['files_sha256']['qwen_full/full_model_driver.py']:
                    issue(issues,path,'Actual shared model driver source differs.')
            elif event=='program_input':
                inputs[key]=row
                request=row.get('request',{})
                if request.get('model')!=expected_model or request.get('stream') is not True:issue(issues,path,'Generation request model or streaming changed.')
                if any(request.get(k) is not None for k in ('max_tokens','max_completion_tokens','max_output_tokens')):
                    issue(issues,path,'Independent output cap unexpectedly sent.')
                for k,v in {'temperature':1.0,'top_p':0.95,'top_k':20,'min_p':0.0,'repetition_penalty':1.0}.items():
                    if request.get(k)!=v:issue(issues,path,'Qwen sampling differs: '+k)
                if (request.get('reasoning_effort')!='medium' or request.get('chat_template_kwargs')!=
                        {'enable_thinking':True,'preserve_thinking':False,'reasoning_effort':'medium'}):
                    issue(issues,path,'Qwen thinking configuration differs.')
            elif event=='generation_started':
                if key in starts:issue(issues,path,'Duplicate generation start.')
                if row.get('episode_id') not in expected_ids:issue(issues,path,'Request outside authorized item.')
                starts.append(key)
            elif event=='generation_usage':
                usage=row.get('usage')
                if key in known:issue(issues,path,'Duplicate usage; not double counted.')
                elif valid_tokens(usage):
                    known[key]=usage;result['input_tokens']+=usage['prompt_tokens'];result['output_tokens']+=usage['completion_tokens']
                    details=usage.get('completion_tokens_details',{});reasoning=details.get('reasoning_tokens') if isinstance(details,dict) else None
                    if type(reasoning) is int and 0<=reasoning<=usage['completion_tokens']:result['reasoning_tokens_reported']+=reasoning
                    else:result['reasoning_counters_missing']+=1
            elif event in ('generation_transport_failure','generation_context_rejected'):failures.add(key)
        if initialized!=1:issue(issues,path,'Expected exactly one driver initialization record.')
    elif list((attempt/'raw_inference').glob('*.sse')):issue(issues,path,'Raw streams lack their request log.')
    archives=sorted((attempt/'raw_inference').glob('*.sse'));numbers=set()
    for raw in archives:
        match=re.fullmatch(r'request_([0-9]+)\.sse',raw.name)
        if not match:issue(issues,raw,'Unexpected request archive name.');continue
        ordinal=int(match[1]);numbers.add(ordinal);stream=audit_stream(raw,issues,expected_model)
        dt=stream.get('request_wall_seconds')
        if type(dt) in (int,float) and math.isfinite(dt) and dt>=0:result['summed_request_wall_seconds']+=dt
        ttft=stream.get('time_to_first_token_seconds')
        if type(ttft) in (int,float) and math.isfinite(ttft) and ttft>=0:
            result['ttft_sum_seconds']+=ttft;result['requests_with_ttft']+=1
        key=starts[ordinal-1] if 1<=ordinal<=len(starts) else None
        stream.update(request_ordinal=ordinal,request_identity=list(key) if key else None,usage_logged=key in known)
        if key is None:issue(issues,raw,'Raw archive lacks matching generation start.')
        else:
            payload=inputs.get(key,{})
            if stream['headers_retained'] and (stream['request_sha256']!=payload.get('request_sha256') or stream['sent_model']!=expected_model):
                issue(issues,raw,'Request header does not bind the generation payload/model.')
            if key in known and stream['provider_usage']!={k:known[key][k] for k in ('prompt_tokens','completion_tokens')}:
                issue(issues,raw,'Logged token counters differ from raw SSE.')
            result['requests'][str(ordinal)]={'episode_id':key[0],'attempt':key[1],'iteration':key[2],
                'usage':known.get(key),'raw_path':str(raw),'transport_failed':key in failures}
        result['raw_streams'].append(stream)
    missing=set(range(1,len(starts)+1))-numbers
    if missing:issue(issues,attempt/'raw_inference','Generation starts missing raw archives: '+str(sorted(missing)))
    if set(known)-set(starts):issue(issues,path,'Usage lacks a preceding generation start.')
    result.update(requests_started=len(starts),requests_with_usage=len(known),requests_without_usage=len(set(starts)-set(known)),
        token_accounting_complete=set(starts)==set(known),raw_archive_bytes=sum(r['bytes'] for r in result['raw_streams']),
        event_counts=dict(counts),token_semantics='Provider counters; missing or invalid usage is unknown, never zero.')
    return result


def verify_manual_initial_compatibility(c,simroot,manifest,first,allocations,issues,lineage=None):
    before=sum(i['severity']=='error' for i in issues)
    evidence=verify_item(c,simroot,manifest,first,allocations,issues,lineage);native=evidence['native']
    first_model=evidence['scored_model']
    if (native['scored']!=1 or native['three_camera_episodes']!=1 or evidence['verified_report_count']!=1
            or not any(r['native_score_committed'] and r['model_iterations']>0 for r in evidence['episode_reports'])
            or not any(s.get('http_status')==200 and s.get('stream_complete') is True
                and s.get('proposal_schema_verified') is True and s.get('finish_reasons')==['stop'] and s.get('model')==first_model
                for u in evidence['usage'] for s in u.get('raw_streams',[]))):
        issue(issues,Path(c)/'items'/first['episode_id'],'First Qwen episode lacks native/program/video/complete-stream proof.')
    if sum(i['severity']=='error' for i in issues)!=before:return None
    return {'schema_version':1,'campaign':manifest['campaign'],'episode_id':first['episode_id'],
        'source_group_id':first['source_group_id'],'verified':True,'quality_threshold_applied':False,
        'verified_report_count':1,'native_scored':1,'video_count':3,'model':first_model,
        'reasoning_effort':manifest['model'].get('reasoning_effort'),
        'native_result_sha256':native['sha256'],'billing_mode':'user_managed',
        'spending_control_enabled':False,'cost_monitoring_enabled':False}


def allocations_closed(allocations, infrastructure_lineage=None):
    return bool(allocations) and all(a.get('finished_utc') and a.get('children_stopped') is True
        and not a.get('cleanup_errors') and a.get('budget_broker_enabled') is False
        and a.get('control_directory_cleaned') is True for _,a in allocations)


def verify_launch_gates(c,manifest,allocations,evidence,issues,lineage=None):
    before=sum(i['severity']=='error' for i in issues);first=group_items(manifest['groups'][0])[0]
    scored=next((i for g in evidence for i in g['item_evidence'] if i['episode_id']==first['episode_id']),None)
    proof=verify_manual_initial_compatibility(c,Path(c).parent.parent.parent,manifest,first,allocations,issues,lineage) if scored and scored['native']['scored'] else None
    source=read_object(Path(c)/'source_evidence/initial_api_compatibility.json',issues,required=bool(proof))
    if source and (not proof or any(source.get(k)!=v for k,v in proof.items())):issue(issues,c,'First-episode source receipt differs.')
    count=0
    for directory,a in allocations:
        required=bool(list(directory.glob('workers/*/*/controller_config.json')))
        binding=(lineage or {}).get('allocations',{}).get(directory.name)
        active=binding['manifest'] if binding else manifest
        active_sha=binding['manifest_sha256'] if binding else _campaign_manifest_sha(c)
        if a.get('manifest_sha256')!=active_sha:issue(issues,directory,'Allocation uses a different source manifest.')
        network=read_object(directory/'network_preflight.json',issues,required=required)
        if network and network.get('verified') is not True:issue(issues,directory,'Internal Qwen endpoint readiness failed.')
        if network and active['model']['provider']=='local_vllm':
            endpoint=network.get('serving_endpoint',{})
            expected={'model':active['model']['served_alias'],'revision':active['model']['checkpoint_revision'],
                'base_url':active['model']['base_url'],'dtype':'bfloat16','tensor_parallel_size':4,
                'mtp_tokens':3,'native_context_tokens':262144,'client_output_cap':None,'ready':True}
            if (any(endpoint.get(k)!=v for k,v in expected.items())
                    or not str(endpoint.get('job_id','')).isdigit() or endpoint.get('job_id')==str(a.get('allocation'))
                    or endpoint.get('node')!='gpu-node'):
                issue(issues,directory/'network_preflight.json','A100 serving checkpoint/allocation/precision/context binding differs.')
        if network and active['model']['provider']=='professor_api':
            expected={'model':MODEL,'model_alias':MODEL,'base_url':active['model']['base_url'],
                'transport':'private_rootless_tailscale_nc','models_http_status':200,
                'exact_model_available':True,'enrollment_key_read':False,
                'provider_api_key_read':False,'system_vpn_dns_routes_changed':False,
                'underlying_checkpoint_verified':False,'manifest_sha256':active_sha}
            if any(network.get(k)!=v for k,v in expected.items()):
                issue(issues,directory/'network_preflight.json','Professor API identity/transport differs.')
            if a.get('finished_utc') and a.get('network_cleanup',{}).get('network_cleanup_verified') is not True:
                issue(issues,directory/'allocation.json','Private Tailscale cleanup not verified.')
        for worker in range(9):
            path=directory/f'container_preflight_w{worker:02d}.json';receipt=read_object(path,issues,required=required)
            if receipt and (receipt.get('verified') is not True or receipt.get('key_visible') is not False
                    or receipt.get('private_control_visible') is not False or receipt.get('relay_visible') is not True
                    or receipt.get('exit_code')!=0):issue(issues,path,'Container isolation failed.')
        path=directory/'initial_api_compatibility.json';receipt=read_object(path,issues)
        if receipt:
            if not proof or any(receipt.get(k)!=v for k,v in proof.items()):issue(issues,path,'Allocation first API proof differs.')
            else:count+=1
    return {'verified':sum(i['severity']=='error' for i in issues)==before,
        'initial_api_compatibility_allocations_verified':count,'first_paired_episode_id':first['episode_id'],
        'extra_scored_probe_episodes':0,'quality_threshold_applied':False,'token_counter_completeness_is_launch_gate':False}


def audit(campaign_dir,progress_only=False):
    from terra_entry import validate_manifest
    c=Path(campaign_dir).resolve(strict=True);root=c.parent.parent;simroot=root.parent;issues=[]
    manifest=validate_manifest(read_object(c/'manifest.json',issues,required=True))
    if manifest['campaign']!=c.name:raise ValueError('campaign name differs')
    rows=[progress_group(simroot,c.name,g,issues) for g in manifest['groups']];families,dimensions=aggregate(rows)
    allocations=[(p,read_object(p/'allocation.json',issues,required=True) or {}) for p in sorted((c/'allocations').glob('*')) if p.is_dir()]
    count=sum(r['scored'] for r in rows);successes=sum(r['successes'] for r in rows)
    life=('allocation','started_utc','finished_utc','exit_code','error','children_stopped','cleanup_errors',
        'complete','stage_complete','terminal_with_missing_pairs','memory_guard_triggered')
    result={'campaign':c.name,'protocol':manifest['protocol'],'audited_utc':datetime.now(timezone.utc).isoformat(),
        'evidence_audited':not progress_only,'planned_episodes':220,'scored_episodes':count,'successes':successes,
        'success_rate':successes/count if count else None,'groups':rows,'families':families,'dimensions':dimensions,
        'complete_groups':sum(r['complete'] for r in rows),'complete_families':sum(r['complete'] for r in families.values()),
        'family_count':len(families),'allocations':[{k:a.get(k) for k in life} for _,a in allocations],
        'unscored_paired_episodes':220-count,'paired_220_complete_verified':False,'complete_verified':False,
        'stage_complete_verified':False,'original_807_complete_verified':False,'official_full_leaderboard_score':False,'issues':issues}
    if progress_only:return result
    result['source_inventory']=verify_inventory(root,manifest,issues)
    result['state_inventory']=verify_state_files(manifest,issues)
    result['pairing']=verify_pairing(c,manifest,issues)
    lineage=verify_infrastructure_phases(c,manifest,allocations,issues)
    evidence=[verify_group(c,simroot,manifest,g,allocations,issues,lineage) for g in manifest['groups']]
    result['group_evidence']=evidence;usages=[u for g in evidence for u in g['usage']]
    expected=set(manifest['episode_ids']);unexpected=[]
    for directory,a in allocations:
        for attempt in directory.glob('workers/*/*'):
            if attempt.is_dir() and attempt.name not in expected:
                issue(issues,attempt,'Unexpected state attempt; raw usage remains included.')
                unexpected.append(audit_attempt_usage(attempt,a,manifest['episode_ids'],manifest,issues))
    usages.extend(unexpected);result['unexpected_attempt_usage']=unexpected
    keys=('requests_started','requests_with_usage','requests_without_usage','input_tokens','output_tokens',
        'reasoning_tokens_reported','reasoning_counters_missing','summed_request_wall_seconds','ttft_sum_seconds',
        'requests_with_ttft','raw_archive_bytes')
    result['total_usage']={k:sum(u[k] for u in usages) for k in keys}
    result['token_accounting_complete']=all(u['token_accounting_complete'] for u in usages)
    result['episode_queue']=verify_episode_queue(c,manifest,allocations,evidence,issues,lineage)
    result['launch_gates']=verify_launch_gates(c,manifest,allocations,evidence,issues,lineage)
    result['provider_transition']={k:v for k,v in lineage.items() if k!='allocations'}
    result['scores_by_provider']=dict(Counter(i['scored_provider'] for g in evidence for i in g['item_evidence'] if i['native']['scored']))
    result['verified_episode_reports']=sum(e['verified_report_count'] for e in evidence)
    result['episodes_with_three_videos']=sum(e['native']['three_camera_episodes'] for e in evidence)
    result['nonempty_video_count']=sum(e['native']['matching_video_count'] for e in evidence)
    result['summed_scored_episode_wall_seconds']=sum(e['summed_scored_episode_wall_seconds'] for e in evidence)
    elapsed=sum(duration_seconds(a['started_utc'],a['finished_utc']) for _,a in allocations if a.get('finished_utc'))
    result['completed_simulation_allocation_wall_seconds']=elapsed
    result['completed_allocations_episode_throughput_per_hour']=count/(elapsed/3600) if elapsed and allocations_closed(allocations) else None
    from scaling_metrics import summarize_telemetry
    result['resources']={p.name:summarize_telemetry(p/'telemetry.jsonl') for p,_ in allocations}
    if result['pairing']:result['paired_comparison']=paired220_outcomes(result['pairing'],evidence,issues)
    latest=allocations[-1][1] if allocations else {}
    result['terminal_with_missing_pairs']=latest.get('terminal_with_missing_pairs') is True
    complete=(count==220 and result['verified_episode_reports']==220 and result['nonempty_video_count']==660
        and result['episodes_with_three_videos']==220 and not any(i['severity']=='error' for i in issues)
        and result['episode_queue']['verified'] and result['launch_gates']['verified']
        and result['launch_gates']['initial_api_compatibility_allocations_verified']>0
        and latest.get('complete') is True and latest.get('exit_code')==0 and allocations_closed(allocations)
        and not result['terminal_with_missing_pairs'])
    result.update(paired_220_complete_verified=complete,complete_verified=complete,stage_complete_verified=complete)
    result['notes']=['Exactly220 baseline states; missing pairs remain unscored, with no replacement.',
        'Qwen uses full public Chat Completions history; OpenAI encrypted Responses reasoning replay does not apply.',
        'Token counters are research metadata; missing counters are unknown and do not invalidate an otherwise valid episode.',
        'Simulation time excludes the historical A100 server allocation, which was released at the authorized provider switch. No API spending is monitored.',
        'A100 and professor API phases remain distinct; the remote underlying checkpoint and server settings are unverified.']
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir',type=Path,required=True)
    parser.add_argument('--progress-only',action='store_true');parser.add_argument('--compact',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv);result=audit(args.campaign_dir,args.progress_only)
    text=json.dumps(result,sort_keys=True,indent=None if args.compact else 2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as f:f.write(text)
    print(text,end='')
    return int(any(i['severity']=='error' for i in result['issues']))

if __name__=='__main__':raise SystemExit(main())
