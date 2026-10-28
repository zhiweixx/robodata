"""Read-only native scoring, evidence and resource audit for a planned subset.

Native scores are authoritative. Full completion requires every planned score plus
committed reports, three camera videos, immutable imports and raw usage evidence.
This custom subset is not an official RoboDojo full leaderboard score.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json,math,re
from pathlib import Path
from robodata.data.provenance import verify_imports
from .runner import verify_campaign,native_result_path,sha_file,INFERENCE
MODEL='Qwen/Qwen3.8-27B'
CAMERAS=('head','left_wrist','right_wrist')
VIDEO=re.compile(r'episode_([0-9]{7})_cam_(head|left_wrist|right_wrist)_(success|fail)\.mp4')
UNCAPPED_CONFIG={'max_iterations':None,'max_input_tokens':None,'independent_max_output_tokens':None,'absolute_generation_timeout_s':None,'expected_context_length':262144}
def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite_number(value, name, *, integer=False):
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"invalid nonnegative {name}")
    return value


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def issue(errors, path, message):
    errors.append({"path": str(path), "message": str(message)})


def native_details(native, group):
    """Validate partial results with the same identity/count rules as completion."""
    details = native["details"]
    if not isinstance(details, dict) or set(details) != {str(i) for i in range(len(details))}:
        raise ValueError("noncanonical native episode indices")
    if (type(native.get("eval_time")) is not int or native["eval_time"] != len(details)
            or len(details) > group["target_episodes"]):
        raise ValueError("native scored count does not match detail mapping/target")
    layout_ids = []
    for row in details.values():
        layout = row["layout_id"]
        if type(layout) is not int or layout not in group["candidate_layout_ids"]:
            raise ValueError("native layout outside authorized candidates")
        if type(row["success"]) is not bool:
            raise ValueError("invalid native success value")
        score = finite_number(row["score"], "native score")
        if score > 1:
            raise ValueError("native score exceeds one")
        layout_ids.append(layout)
    if len(set(layout_ids)) != len(layout_ids):
        raise ValueError("duplicate native scored layout")
    return details


def video_evidence(native_dir, details):
    good, bad = set(), []
    for path in sorted(native_dir.glob("*.mp4")):
        match = VIDEO.fullmatch(path.name)
        if match is None or path.stat().st_size == 0:
            bad.append(path.name)
            continue
        index, camera, outcome = str(int(match[1])), match[2], match[3]
        key = (index, camera)
        if (index not in details or key in good
                or outcome != ("success" if details[index]["success"] else "fail")):
            bad.append(path.name)
            continue
        good.add(key)
    verified = sum(all((index, camera) in good for camera in CAMERAS) for index in details)
    return {"verified_scored_episodes": verified, "matching_nonempty_videos": len(good),
            "invalid_or_unexpected_videos": bad}


def jsonl_rows(path, errors):
    # Stream the ledger: image hashes/program bodies need not be retained.
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected JSON object")
                yield row
            except (ValueError, TypeError) as exc:
                issue(errors, path, f"line {line_number}: {exc}; incomplete line excluded")


def empty_usage():
    return {"ledgers": 0, "model_calls_started": 0, "model_calls_with_usage": 0,
            "prompt_tokens": 0, "completion_tokens_including_reasoning": 0,
            "generation_wall_seconds": 0.0, "tokenization_wall_seconds": 0.0,
            "calls_without_final_token_usage": 0, "transport_failure_calls": 0,
            "explicit_infrastructure_interrupted_attempts": 0,
            "native_unstable_abandoned_attempts": 0, "model_episode_failure_attempts": 0,
            "api_cost_usd": 0}


def audit_usage(path, allowed_ids, seeds, errors):
    totals = empty_usage()
    starts, usages, runtime, tokenizations = set(), {}, {}, {}
    infrastructure, unstable, model_failures, transport_failures = set(), set(), set(), set()
    model, initialized = None, False
    try:
        for row in jsonl_rows(path, errors):
            event = row.get("event")
            if event == "suite_initialized":
                config = row["config"]
                if (initialized or config.get("purpose") != "aspire_robodojo_full_local_qwen"
                        or config.get("model") != MODEL
                        or set(config.get("episode_ids", [])) != set(allowed_ids)
                        or config.get("episode_seeds") != seeds
                        or config.get("remote_api_calls") is not False
                        or any(key not in config or config[key] != value
                               for key, value in UNCAPPED_CONFIG.items())):
                    raise ValueError("usage ledger protocol/model/episode identity mismatch")
                initialized, model = True, config["model"]
                continue
            if not initialized:
                raise ValueError("usage events precede suite_initialized")
            if "episode_id" not in row:
                continue
            episode_id = row["episode_id"]
            if episode_id not in allowed_ids:
                raise ValueError("usage event outside authorized episodes")
            attempt = finite_number(row.get("attempt", 1), "attempt", integer=True)
            if attempt == 0:
                raise ValueError("attempt numbers start at one")
            attempt_key = (episode_id, attempt)
            key = (*attempt_key, row.get("iteration"))
            if event == "generation_started":
                if key in starts:
                    raise ValueError("duplicate generation_started request")
                starts.add(key)
            elif event == "generation_usage":
                if key in usages:
                    raise ValueError("duplicate generation_usage request")
                usage = row["usage"]
                prompt = finite_number(usage["prompt_tokens"], "prompt tokens", integer=True)
                completion = finite_number(usage["completion_tokens"], "completion tokens", integer=True)
                if usage.get("api_cost_usd") != 0:
                    raise ValueError("unexpected nonzero/missing API cost")
                usages[key] = (prompt, completion)
                runtime.setdefault(key, finite_number(usage["generation_wall_s"], "generation time"))
            elif event in {"raw_response", "generation_transport_failure"}:
                runtime[key] = finite_number(row["generation_wall_s"], "generation time")
                if event == "generation_transport_failure":
                    transport_failures.add(key)
                    infrastructure.add(attempt_key)
            elif event == "input_token_count":
                if key in tokenizations:
                    raise ValueError("duplicate tokenization request")
                tokenizations[key] = finite_number(row["wall_s"], "tokenization time")
            elif event == "infrastructure_stopped":
                infrastructure.add(attempt_key)
            elif event == "unscored_attempt_abandoned":
                if row.get("reason") == "infrastructure_interrupted":
                    infrastructure.add(attempt_key)
                elif row.get("reason") == "native_unstable":
                    unstable.add(attempt_key)
            elif event == "model_episode_failed":
                model_failures.add(attempt_key)
        if not initialized:
            raise ValueError("ledger has no initialization record")
        if usages.keys() - starts:
            raise ValueError("generation usage has no matching request start")
        totals.update(ledgers=1, model_calls_started=len(starts), model_calls_with_usage=len(usages),
                      prompt_tokens=sum(value[0] for value in usages.values()),
                      completion_tokens_including_reasoning=sum(value[1] for value in usages.values()),
                      generation_wall_seconds=sum(runtime.values()),
                      tokenization_wall_seconds=sum(tokenizations.values()),
                      calls_without_final_token_usage=len(starts - usages.keys()),
                      transport_failure_calls=len(transport_failures),
                      explicit_infrastructure_interrupted_attempts=len(infrastructure),
                      native_unstable_abandoned_attempts=len(unstable),
                      model_episode_failure_attempts=len(model_failures))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issue(errors, path, exc)
        # No partial sum from an untrusted ledger: identity may be wrong.
        return None
    return {"model": model, **totals}


def add_usage(destination, source):
    for key in empty_usage():
        destination[key] += source[key]



def _expected_imports(campaign,inventory,contract):
    imported=contract.get('imported_sources',{})
    if not imported:raise ValueError('missing executed source origins')
    remapped={}
    for rel,origin in imported.items():
        if Path(origin)!=Path('/opt/robodata')/rel:raise ValueError('module origin escapes frozen container mount')
        remapped[rel]=campaign/'source'/rel
    verify_imports(campaign/'source',inventory,remapped)
    required={'src/robodata/eval/robodojo/entry.py','src/robodata/eval/robodojo/full_entry.py',
        'src/robodata/eval/robodojo/full_episode.py','src/robodata/execution/aspire_glue.py',
        'src/robodata/execution/receipt_executor.py','src/robodata/perception/perception_bridge.py',
        'src/robodata/perception/perception_service.py','src/robodata/perception/cgn_singleton_fix.py',
        'third_party/aspire/src/cap_harness/api.py'}
    if not required<=set(imported):raise ValueError('missing core executed-source evidence')


def _providers(provenance,inventory):
    from robodata.perception.cgn_singleton_fix import expected_patch_evidence
    from robodata.perception.perception_service import SAM3_WEIGHT_SHA256,CGN_WEIGHT_SHA256
    health=provenance['service_health']
    if provenance.get('offline_skill_discovery') is not False:raise ValueError('unexpected learning workflow')
    if provenance.get('service_runner_sha256')!=inventory['files_sha256']['src/robodata/perception/perception_service.py']:
        raise ValueError('service source differs from frozen release')
    for name,weight in [('sam3',SAM3_WEIGHT_SHA256),('graspnet',CGN_WEIGHT_SHA256)]:
        row=health[name]
        if not all(row.get(k) is True for k in ('ok','loaded','model_loaded')) or row.get('checkpoint_sha256')!=weight or row.get('parameter_devices')!=['cuda:0']:
            raise ValueError('actual provider weights/device/readiness mismatch')
    if health['graspnet'].get('estimator_patch')!=expected_patch_evidence(inventory['files_sha256']['src/robodata/perception/cgn_singleton_fix.py']):
        raise ValueError('missing CGN singleton repair evidence')


def audit_campaign(campaign_dir,full=False):
    campaign=Path(campaign_dir).expanduser().resolve(strict=True)
    plan,manifest,config,inventory=verify_campaign(campaign)
    errors=[];groups=[];usage=empty_usage();smoke_usage=empty_usage();attempts=[]
    usage_ids=set();provider_hashes=set();completed_report_ids=set();seen_ledgers=set()
    if full:
        for record in sorted((campaign/'allocations').glob('*/*/attempt_*/worker.json')):
            try:
                worker=read_json(record);gid=worker['group_id']
                group=next(g for g in manifest['groups'] if g['group_id']==gid)
                if worker.get('source_tree_sha256')!=inventory['tree_sha256'] or worker.get('manifest_sha256')!=sha_file(campaign/'manifest.json') or worker.get('cluster_config_sha256')!=sha_file(campaign/'cluster_config.json'):
                    raise ValueError('worker frozen contract mismatch')
                if worker.get('inference')!=INFERENCE:raise ValueError('worker inference policy changed')
                row={'group_id':gid,'path':str(record.parent),'complete':worker.get('complete') is True,
                    'started_utc':worker.get('started_utc'),'finished_utc':worker.get('finished_utc'),'error':worker.get('error')}
                if row['finished_utc']:
                    row['worker_wall_seconds']=(timestamp(row['finished_utc'])-timestamp(row['started_utc'])).total_seconds()
                attempts.append(row)
                smoke_ledger=record.parent/'model_smoke/report.usage.jsonl'
                if smoke_ledger.exists():
                    smoke_part=audit_usage(smoke_ledger,['interface_smoke'],{'interface_smoke':0},errors)
                    if smoke_part is not None:add_usage(smoke_usage,smoke_part)
                providers=record.parent/'providers.json'
                if providers.exists():
                    _providers(read_json(providers)['provenance'],inventory)
                    provider_hashes.add(sha_file(providers))
                ledger=record.parent/'usage.jsonl'
                if ledger.exists():
                    engine=worker.get('engine_config','')
                    if (not all(t in engine for t in ('dtype=torch.bfloat16','quantization=None','tensor_parallel_size=2','max_seq_len=262144'))
                            or not re.search(r"method\s*=\s*['\"]mtp['\"]",engine)
                            or not re.search(r'num_spec(?:ulative)?_tokens\s*=\s*3\b',engine)):
                        raise ValueError('actual inference engine evidence differs')
                    model_record=worker.get('model_manifest',{})
                    if model_record.get('repo')!=MODEL or model_record.get('revision')!='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0' or not model_record.get('files'):
                        raise ValueError('model snapshot byte inventory missing')
                    seen_ledgers.add(ledger.resolve())
                    header=next(jsonl_rows(ledger,errors),{})
                    cfg=header.get('config',{})
                    if (cfg.get('driver_sha256')!=inventory['files_sha256']['src/robodata/agents/full_model_driver.py']
                        or cfg.get('sampling')!={'temperature':1.0,'top_p':.95,'top_k':20,'min_p':0.0,'presence_penalty':0.0,'repetition_penalty':1.0}
                        or cfg.get('chat_template_kwargs')!={'enable_thinking':True,'preserve_thinking':False,'reasoning_effort':'medium'}
                        or cfg.get('reasoning_effort')!='medium'
                        or cfg.get('history_policy')!='all public messages and images; no rolling truncation; prior reasoning omitted'):
                        raise ValueError('model source/sampling/history evidence differs')
                    part=audit_usage(ledger,group['episode_ids'],{x:group['eval_seed'] for x in group['episode_ids']},errors)
                    if part is not None:
                        add_usage(usage,part)
                        for event in jsonl_rows(ledger,errors):
                            if event.get('event')=='episode_closed':usage_ids.add(event.get('episode_id'))
            except (OSError,ValueError,KeyError,TypeError,StopIteration) as exc:issue(errors,record,exc)
    if full:
        for ledger in (campaign/'allocations').glob('*/*/attempt_*/usage.jsonl'):
            if ledger.resolve() not in seen_ledgers:issue(errors,ledger,'orphan or unvalidated raw usage ledger; resource totals incomplete')
    for group in manifest['groups']:
        gid=group['group_id'];native=native_result_path(campaign,group);directory=campaign/'groups'/gid
        row={'group_id':gid,'family':group['family'],'task':group['task'],'category':group['dimension'],
            'target':group['target_episodes'],'scored':0,'success':0,'score_sum':0.,'scores_complete':False,
            'evidence_verified':False,'state':'not_started','native_result_path':str(native)}
        details={}
        try:
            if native.exists():
                details=native_details(read_json(native),group)
                row.update(scored=len(details),success=sum(d['success'] for d in details.values()),score_sum=sum(d['score'] for d in details.values()),
                    scores_complete=len(details)==group['target_episodes'],state='scores_complete' if len(details)==group['target_episodes'] else 'partial')
            if full and details:
                contract=read_json(directory/'execution_contract.json')
                if (contract.get('group_id')!=gid or contract.get('manifest_sha256')!=sha_file(campaign/'manifest.json')
                        or contract.get('source_tree_sha256')!=inventory['tree_sha256'] or contract.get('step_limit')!=group['step_limit']
                        or contract.get('target_episodes')!=group['target_episodes'] or contract.get('candidate_layout_ids')!=group['candidate_layout_ids']):
                    raise ValueError('native execution contract differs')
                _expected_imports(campaign,inventory,contract)
                videos=video_evidence(native.parent,details);row['videos']=videos
                if videos['verified_scored_episodes']!=len(details) or videos['invalid_or_unexpected_videos']:raise ValueError('native video evidence incomplete/unexpected')
                report_path=directory/'group_report.json';report=read_json(report_path)
                if (report.get('mode')!='evaluation' or report.get('task')!=group['task'] or report.get('eval_seed')!=group['eval_seed']
                        or report.get('native_run_id')!=f'{campaign.name}_{gid}' or report.get('target')!=group['target_episodes']
                        or report.get('gpu_isolation',{}).get('verified') is not True):raise ValueError('native group identity/GPU evidence differs')
                verified=set()
                for ep_path in sorted(directory.glob('episodes/*/attempt_*/episode_report.json')):
                    ep=read_json(ep_path);layout=ep['layout_id'];eid=f'{gid}_l{layout}'
                    if (layout not in group['candidate_layout_ids'] or ep.get('episode_id')!=eid or ep_path.parent.parent.name!=eid
                            or ep.get('task')!=group['task'] or ep.get('eval_seed')!=group['eval_seed']):raise ValueError('episode identity mismatch')
                    if ep.get('native_score_committed') is not True:continue
                    native_row=next((r for r in details.values() if r['layout_id']==layout),None)
                    if native_row is None or layout in verified or ep.get('status')!='completed' or ep.get('step_limit')!=group['step_limit']:
                        raise ValueError('committed report mismatched/duplicate native score')
                    expected_limits={'program_count':None,'program_wall_seconds':None,'output_tokens':'remaining native model context','action_budget':'native task config'}
                    if ep.get('limits')!=expected_limits:raise ValueError('episode has changed action/model budget')
                    steps=ep.get('environment_steps')
                    if type(steps) is not int or not 0<=steps<=group['step_limit']:raise ValueError('episode native action count out of range')
                    if ep.get('official',{}).get('official_success') is not native_row['success']:raise ValueError('episode/native success receipt differs')
                    runtime=ep['perception_runtime']
                    if (runtime.get('initialization_complete') is not True or runtime.get('providers')!={'sam3':True,'contact_graspnet':True}
                            or runtime.get('provider_config_sha256') not in provider_hashes):raise ValueError('actual episode perception evidence missing')
                    _providers(runtime['service_provenance'],inventory)
                    if eid not in usage_ids:raise ValueError('native committed episode lacks model finish receipt')
                    verified.add(layout);completed_report_ids.add(eid)
                row['committed_reports_verified']=len(verified)
                if verified!={d['layout_id'] for d in details.values()}:raise ValueError('native score/report evidence incomplete')
                row['evidence_verified']=True
        except (OSError,ValueError,KeyError,TypeError) as exc:issue(errors,native,exc);row['state']='invalid_evidence'
        row['success_rate']=row['success']/row['scored'] if row['scored'] else None
        groups.append(row)
    by_family={};by_category={}
    for key,target in [('family',by_family),('category',by_category)]:
        for row in groups:
            item=target.setdefault(row[key],{'scored':0,'target':0,'success':0,'score_sum':0.})
            for field in ('scored','target','success','score_sum'):item[field]+=row[field]
        for item in target.values():item['success_rate']=item['success']/item['scored'] if item['scored'] else None
    scored=sum(g['scored'] for g in groups);success=sum(g['success'] for g in groups)
    target_episodes=sum(g['target'] for g in groups)
    return {'schema':'robodata.robodojo.audit.v1','campaign_dir':str(campaign),'full_evidence_audit':full,
        'is_official_full_evaluation':False,'target':target_episodes,'scored':scored,'success':success,
        'success_rate':success/scored if scored else None,'groups_scores_complete':sum(g['scores_complete'] for g in groups),
        'complete':bool(full and not errors and scored==target_episodes and all(g['evidence_verified'] for g in groups)),
        'source_tree_sha256':inventory['tree_sha256'],'groups':groups,'by_family':by_family,'by_category':by_category,
        'usage':usage if full else None,'unscored_model_interface_smoke_usage':smoke_usage if full else None,'worker_attempts':attempts if full else None,'errors':errors,
        'notes':[f'This plan requests {target_episodes} new scores; no historical results are merged.',
            'Worker wall times overlap across nodes: their sum is resource time, not elapsed completion time.',
            'Partial observed success is not an official five-category RoboDojo Avg.']}
