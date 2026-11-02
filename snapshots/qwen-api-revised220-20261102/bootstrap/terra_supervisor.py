"""Nine work-conserving episode slots for an exact frozen Terra state pairing.

Every atomic queue claim owns one exact state and a separate native run. A fresh
Isaac process isolates state/history/checkpoint ownership at each episode.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import concurrent.futures
import csv
import fcntl
import json
import importlib.util
import sys
import math
import os
from pathlib import Path
import queue
import pwd
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
from types import SimpleNamespace

from terra_entry import MODEL, episode_items, activate_profile, sha_file, validate_manifest, verify_sources, user_managed_billing
from prepare_terra import verify_pairing_artifacts
from budget_broker import BudgetBroker, BudgetClient
from budget_ledger import read_only_snapshot

THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
FINANCIAL_STATUS_POLL_SECONDS = 60
MEMORY_GUARD_POLL_SECONDS = 2

from terra_platform import resolve_four_gpus, capture_gpu_identity_evidence


def cpu_plan(allowed, worker_count=9):
    allowed = sorted(allowed)
    if (worker_count != 9 or len(allowed) != 32 or len(set(allowed)) != 32
            or any(type(cpu) is not int or cpu < 0 for cpu in allowed)):
        raise RuntimeError('nine-worker protocol requires 32 distinct allocated CPUs')
    return {'original_allowed': allowed, 'shared': allowed[:5],
            'simulators': {f'w{i:02d}': allowed[5 + 3*i:8 + 3*i] for i in range(9)}}


def allocated_cpu_ids(os_affinity, job_details, hostname=None):
    """Slurm's explicit CPU assignment wins over a broader batch-shell affinity."""
    assignments = re.findall(r'(?:^|\s)Nodes=(\S+)\s+CPU_IDs=([0-9,-]+)(?=\s|$)', job_details)
    matches = [value for _, value in assignments]
    if not matches:
        raise RuntimeError('cannot prove the 32 CPU job mask from Slurm CPU_IDs')
    if len(matches) != 1:
        raise RuntimeError('single-node CPU assignment is ambiguous')
    if hostname is not None and assignments[0][0] != hostname.split('.')[0]:
        raise RuntimeError('Slurm CPU assignment belongs to a different node')
    assigned = set()
    for token in matches[0].split(','):
        endpoints = [int(value) for value in token.split('-')]
        if len(endpoints) == 1:
            assigned.add(endpoints[0])
        elif len(endpoints) == 2 and endpoints[0] <= endpoints[1]:
            assigned.update(range(endpoints[0], endpoints[1] + 1))
        else:
            raise RuntimeError('invalid Slurm CPU range')
    if len(assigned) != 32 or not assigned <= set(os_affinity):
        raise RuntimeError('Slurm CPU assignment conflicts with process affinity or 32 CPU protocol')
    return sorted(assigned)


def memory_guard(sample, requested_limit_bytes):
    """Stop before a sampled host-RAM spike reaches the allocation limit.

    This operational guard is not a measured absolute hardware capacity.
    Missing cgroup counters fail closed rather than pretending memory is safe.
    """
    current = sample.get('memory_current_bytes')
    observed_limit = sample.get('memory_limit_bytes')
    if (sample.get('available') is not True or type(current) is not int or current < 0
            or type(requested_limit_bytes) is not int or requested_limit_bytes <= 12*1024**3):
        return {'safe': False, 'reason': 'valid own-cgroup memory observation required'}
    limits = [requested_limit_bytes]
    if type(observed_limit) is int and observed_limit > 0:
        limits.append(observed_limit)
    limit = min(limits)
    ceiling = min(int(.94*limit), limit-12*1024**3)
    oom = any(sample.get('events', {}).get(k, 0) > 0 for k in ('oom', 'oom_kill', 'limit_failcnt'))
    return {'safe': current < ceiling and not oom, 'memory_current_bytes': current,
            'effective_memory_limit_bytes': limit, 'ceiling_bytes': ceiling,
            'oom_event_observed': oom,
            'reason': 'own-cgroup current memory below fixed operational reserve' if current < ceiling and not oom
                      else 'host memory guard triggered; preserve unscored work, no automatic replay'}


def synthetic_broker_preflight(private_dir, durable_dir=None):
    """Exercise nine compute-host clients without touching paid ledger or API."""
    directory=Path(private_dir)
    durable=Path(durable_dir) if durable_dir is not None else directory
    if durable_dir is not None:durable.mkdir(parents=True,exist_ok=False)
    journal=durable/'synthetic-ledger.jsonl'
    broker=BudgetBroker(journal,directory/'socket',cap_usd='2000')
    started=time.monotonic();release=threading.Event();barrier=threading.Barrier(10,timeout=15)
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=9)
    futures=[];result=None
    try:
        broker.start()
        def client_work(index):
            client=BudgetClient(broker.socket_path);rid=f'synthetic-client-{index}'
            client.reserve(rid,payload_sha256='a'*64,allocation_id='synthetic-compute-preflight',worker_id=f'w{index:02d}')
            barrier.wait()
            if not release.wait(15):raise RuntimeError('synthetic compute admission phase timed out')
            client.settle(rid,usage={'input_tokens':100,'output_tokens':10,'total_tokens':110,
                'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0},
                'output_tokens_details':{'reasoning_tokens':0}},response_id=f'synthetic-response-{index}')
        futures=[pool.submit(client_work,i) for i in range(9)]
        barrier.wait()
        active=BudgetClient(broker.socket_path).status()['request_counts']['active']
        if active!=9:raise RuntimeError('nine simultaneous synthetic reservations not observed')
        release.set()
        for future in futures:future.result(timeout=15)
        status=BudgetClient(broker.socket_path).status()
        if status['request_counts']!={'active':0,'settled':9,'unknown':0}:
            raise RuntimeError('synthetic compute ledger did not settle exactly nine requests')
        result={'schema_version':1,'verified':True,'synthetic_only':True,'api_calls':0,
            'private_temporary_ledger':durable_dir is None,'isolated_synthetic_ledger':True,
            'private_node_local_socket':True,'durable_journal_sha256':sha_file(journal),
            'durable_journal_path':str(journal) if durable_dir is not None else None,
            'durable_journal_preserved':durable_dir is not None,'clients':9,'simultaneous_reservations':active,
            'settled_requests':9,'unknown_requests':0,'wall_seconds':time.monotonic()-started}
    finally:
        release.set()
        pool.shutdown(wait=True,cancel_futures=True)
        broker.stop()
        if broker.socket_path is not None and broker.socket_path.exists():
            raise RuntimeError('synthetic broker socket remained after stop')
        shutil.rmtree(directory)
    if directory.exists():raise RuntimeError('synthetic ledger cleanup failed')
    result['cleanup_verified']=True
    return result


def synthetic_worker_preflight():
    """Verify nine independent CPU workers without accounting or API calls."""
    started=time.monotonic();release=threading.Event();barrier=threading.Barrier(10,timeout=15)
    active=set();lock=threading.Lock();pool=concurrent.futures.ThreadPoolExecutor(max_workers=9)
    futures=[]
    try:
        def worker(index):
            with lock: active.add(index)
            barrier.wait()
            if not release.wait(15):raise RuntimeError('synthetic worker release timed out')
            return index
        futures=[pool.submit(worker,i) for i in range(9)]
        barrier.wait()
        with lock: observed=len(active)
        if observed!=9:raise RuntimeError('nine simultaneous CPU workers not observed')
        release.set()
        results=[f.result(timeout=15) for f in futures]
        if sorted(results)!=list(range(9)):raise RuntimeError('worker identities repeated or missing')
    finally:
        release.set();pool.shutdown(wait=True,cancel_futures=True)
    return {'schema_version':1,'verified':True,'mode':'user_managed','clients':9,
        'simultaneous_clients':observed,'api_calls':0,'financial_operations':0,
        'spending_control_enabled':False,'cleanup_verified':True,'wall_seconds':time.monotonic()-started}


def runtime_state_directories(state):
    state=Path(state)
    for name in ('runtime-home','cache/kit','cache/main','cache/computecache','logs','config',
                 'data/documents','data/kit-user','pkg','private-tmp','private-var-tmp','hidden-host-control'):
        (state/name).mkdir(parents=True,exist_ok=True)
    return state


def container_bindings(state, relay, broker_dir):
    # Explicit relay is outside /tmp. Masking default host temporary mounts does
    # not change provider HTTP services or that separately bound relay endpoint.
    return ','.join((str(relay)+':/aspire-relay',str(Path(state)/'private-tmp')+':/tmp',
                     str(Path(state)/'private-var-tmp')+':/var/tmp',
                     str(Path(state)/'hidden-host-control')+':'+str(broker_dir)+':ro'))


CONTAINER_PYTHON = '/work/scripts/python_exec.sh'


def sanitized_preflight_tail(text, *, private_values=(), max_bytes=4096):
    """Bounded bootstrap diagnostics; never log credential strings or private paths."""
    if isinstance(text,bytes):text=text.decode('utf-8',errors='replace')
    if not isinstance(text,str):text=str(text)
    for private in sorted((str(v) for v in private_values if v),key=len,reverse=True):
        text=text.replace(private,'<private-control>')
    text=re.sub(r'\b(?:sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9_-]{8,}|tskey-[A-Za-z0-9_-]{8,})\b',
                '<redacted-credential>',text)
    text=re.sub(r'(?i)(?:OPENAI_API_KEY|HF_TOKEN|TS_AUTHKEY|TAILSCALE_AUTHKEY|AUTHORIZATION)\s*[:=]\s*[^\s]+',
                '<redacted-authorization>',text)
    text=re.sub(r'(?i)\bBearer\s+[^\s]+','Bearer <redacted>',text)
    return text.encode('utf-8',errors='replace')[-max_bytes:].decode('utf-8',errors='ignore')


CONTAINER_VISIBILITY_PROBE = r"""
import json, os, pathlib, sys
key, broker, home, nonce = sys.argv[1:]
def marker(path):
    p=pathlib.Path(path)
    return p.is_file() and p.read_text()==nonce
value={'key_visible':os.path.lexists(key),'budget_socket_visible':os.path.lexists(broker),
    'relay_visible':pathlib.Path('/aspire-relay').is_dir(),
    'private_home_verified':marker(str(pathlib.Path(home)/'.terra-preflight-marker')),
    'private_tmp_verified':marker('/tmp/.terra-preflight-marker'),
    'private_var_tmp_verified':marker('/var/tmp/.terra-preflight-marker')}
print(json.dumps(value,sort_keys=True))
raise SystemExit(0 if value=={'key_visible':False,'budget_socket_visible':False,
    'relay_visible':True,'private_home_verified':True,'private_tmp_verified':True,
    'private_var_tmp_verified':True} else 2)
"""


def expected_result_path(simroot, campaign, group):
    return (Path(simroot) / 'RoboDojo/eval_result/RoboDojo' / group['task']
        / 'ASPIRE_Pilot/arx_x5' / f"{group['eval_seed']}_aspire_terra_paired"
        / f"{campaign}_{group['group_id']}" / '_result.json')


def native_details(native, group):
    details = native.get('details', {})
    if (not isinstance(details, dict) or set(details) != {str(i) for i in range(len(details))}
            or type(native.get('eval_time')) is not int or native['eval_time'] != len(details)
            or len(details) > group['target_episodes']):
        raise RuntimeError('native score indices/count inconsistent; never replay')
    ids = []
    for d in details.values():
        idx = d.get('layout_id')
        if (type(idx) is not int or idx not in group['candidate_layout_ids']
                or type(d.get('success')) is not bool or type(d.get('score')) not in (int, float)
                or not math.isfinite(d['score']) or not 0 <= d['score'] <= 1):
            raise RuntimeError('native state identity or score invalid; never replay')
        ids.append(idx)
    if len(ids) != len(set(ids)):
        raise RuntimeError('duplicate native state score; never replay')
    return details


def inspect_group(campaign_dir, group, simroot, validate_complete):
    """Return complete/pending/exhausted after verifying native resume evidence."""
    directory = Path(campaign_dir) / 'items' / group['group_id']
    native = expected_result_path(simroot, Path(campaign_dir).name, group)
    checkpoint = native.parent.parent / ('_resume_' + native.parent.name + '.json')
    details = native_details(json.loads(native.read_text()), group) if native.exists() else {}
    reports = [p for p in (directory/'group_report.json', directory/'group_progress.json') if p.is_file()]
    reports.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for report in reports:
        prior = json.loads(report.read_text())
        if prior.get('complete') is True:
            return {'state': 'complete', 'validation': validate_complete(report, group, Path(simroot),
                expected_campaign=Path(campaign_dir).name), 'report_path': str(report)}
        if prior.get('scored', 0) > len(details):
            raise RuntimeError('group report has scores absent from native evidence; never replay')
    if len(details) == group['target_episodes']:
        raise RuntimeError('scores complete but completion/video evidence requires repair; never replay')
    if details and not checkpoint.exists():
        raise RuntimeError('refusing to replay scored states without native checkpoint')
    remaining = set(group['candidate_layout_ids']) - {d['layout_id'] for d in details.values()}
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        completed = {d['layout_id'] for d in details.values()}
        saved_completed, abandoned = saved.get('completed_layout_ids', []), saved.get('abandoned_layout_ids', [])
        candidates = set(group['candidate_layout_ids'])
        for values in (saved_completed, abandoned):
            if (not isinstance(values, list) or any(type(i) is not int for i in values)
                    or len(values) != len(set(values)) or not set(values) <= candidates):
                raise RuntimeError('native checkpoint contains invalid/unauthorized state IDs')
        if set(saved_completed) != completed or completed & set(abandoned):
            raise RuntimeError('native checkpoint and committed states disagree')
        remaining = candidates - completed - set(abandoned)
    receipt_path = directory / 'terra_group_completion.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if receipt and (receipt.get('group_id') != group['group_id']
            or receipt.get('target_episodes') != group['target_episodes']
            or type(receipt.get('scored')) is not int or receipt['scored'] > len(details)):
        raise RuntimeError('native completion receipt identity/count mismatch')
    if receipt.get('native_candidate_exhausted') is True and (
            receipt.get('observed_native_queue_exhausted') is not True
            or receipt.get('native_close_seen') is not True or receipt.get('error') is not None
            or receipt['scored'] != len(details)):
        raise RuntimeError('native candidate exhaustion lacks a clean native close receipt')
    exhausted = receipt.get('native_candidate_exhausted') is True or not remaining
    return {'state': 'exhausted' if exhausted else 'pending', 'scored': len(details),
        'target': group['target_episodes'], 'native_result_path': str(native),
        'checkpoint_path': str(checkpoint) if checkpoint.exists() else None,
        'remaining_candidate_count': len(remaining),
        'missing_pair_layout_ids': sorted(set(group['candidate_layout_ids']) - {d['layout_id'] for d in details.values()}) if exhausted else [],
        'reason': 'exact paired states exhausted; preserve missing pairs without replacement or score' if exhausted else None}


def fresh_campaign_guard(root, campaign, groups):
    out = Path(root) / 'full_results' / campaign
    if any((out/'items').glob('*')):
        raise RuntimeError('full mode refuses prior group evidence; use inspected resume')
    for group in groups:
        native = expected_result_path(Path(root).parent, campaign, group)
        checkpoint = native.parent.parent / ('_resume_' + native.parent.name + '.json')
        if native.exists() or checkpoint.exists():
            raise RuntimeError('full mode refuses existing native scores/checkpoints')


def validate_prepaid_resume(campaign_dir, manifest, *, include_allocations=False):
    """Reject unverified pre-paid source lineage before creating any new attempt."""
    if manifest.get('harness_revision',{}).get('version')=='robodojo_task_reasoning_v2':
        from audit_qwen_results import verify_infrastructure_phases
        issues=[];value=verify_infrastructure_phases(campaign_dir,manifest,[],issues)
        if not value['verified']:raise ValueError('fresh revision source lineage rejected')
        return value
    if not any(k in manifest for k in ('prepaid_infrastructure_phases','transport_infrastructure_phases')):
        return {'verified':True,'phases':[]}
    from audit_qwen_results import verify_infrastructure_phases
    directory=Path(campaign_dir);allocations=[];issues=[]
    for path in sorted((directory/'allocations').glob('*/allocation.json')):
        allocations.append((path.parent,json.loads(path.read_text())))
    result=verify_infrastructure_phases(directory,manifest,allocations,issues)
    if not result.get('verified') or issues:
        raise ValueError('pre-paid infrastructure lineage must verify before allocating or admitting work')
    return result if include_allocations else {key:value for key,value in result.items() if key!='allocations'}


def save_budget_cache_integrity_diagnostics(campaign, phase):
    """Archive bounded owner diagnostics without making financial mutations.

    This runs both before and after stop (including stop failure). It cannot
    conceal cleanup failure or make a failed journal acceptable for admission.
    """
    if phase not in ('before_stop','after_stop'):
        raise ValueError('unsupported budget cache diagnostic phase')
    try:
        diagnostics=campaign.broker.cache_integrity_diagnostics()
        campaign.report.setdefault('budget_cache_integrity_diagnostics',{})[phase]=diagnostics
        campaign.full.write_json(campaign.runtime/f'budget_cache_integrity_{phase}.json',diagnostics)
        return True
    except BaseException as exc:
        campaign.report.setdefault('budget_cache_integrity_diagnostic_errors',{})[phase]={'type':type(exc).__name__}
        return False


class TerraCampaign:
    def __init__(self, args):
        self.root = args.root.resolve(strict=True)
        self.simroot = self.root.parent
        self.code = Path(__file__).resolve().parent
        self.manifest_path = args.manifest.resolve(strict=True)
        self.manifest = validate_manifest(json.loads(self.manifest_path.read_text()))
        self.user_managed_billing = user_managed_billing(self.manifest)
        self.campaign, self.mode = args.campaign, args.mode
        if self.campaign != self.manifest['campaign'] or self.mode not in ('full', 'resume'):
            raise ValueError('campaign identity or launch mode invalid')
        self.out = self.root/'full_results'/self.campaign
        if self.manifest_path != self.out/'manifest.json':
            raise ValueError('manifest must be the frozen campaign file')
        self.profile = verify_sources(self.root, self.manifest, self.code)
        if args.profile.resolve(strict=True) != self.profile:
            raise ValueError('source profile differs from frozen manifest')
        self.groups = self.manifest['groups']
        self.items = episode_items(self.manifest)
        self.execution_stage = None
        deployment = json.loads((self.out/'deployment.json').read_text())
        if deployment.get('manifest_sha256') != sha_file(self.manifest_path) or deployment.get('bootstrap_sha256') != self.manifest['bootstrap_sha256']:
            raise ValueError('deployment does not bind current manifest/source bytes')
        lineage=validate_prepaid_resume(self.out,self.manifest,include_allocations=True)
        from audit_qwen_results import pairing_manifest_for_preflight
        self.pairing=verify_pairing_artifacts(self.out,pairing_manifest_for_preflight(self.manifest,lineage))
        self.prepaid_lineage={key:value for key,value in lineage.items() if key!='allocations'}
        active = subprocess.check_output(['squeue','--noheader','--user',os.environ.get('USER','ROBODATA_USER'),'--format=%A'],text=True).split()
        if set(active) & set(self.pairing['source']['allocation_ids']):
            raise RuntimeError('source baseline allocation still active; no overlapping follow-on launch')
        self.topology = self.manifest['simulator_gpu_slots']
        if self.mode == 'full':
            fresh_campaign_guard(self.root, self.campaign, self.items)
            if any((self.out/'allocations').glob('*')):
                raise RuntimeError('full mode refuses existing allocation; use inspected resume')
        activate_profile(self.profile)
        import full_supervisor as full
        import perception_services
        spec = importlib.util.spec_from_file_location('_terra_frozen_pairing_checks', self.profile/'qwen_full/paired_manifest.py')
        pairing_checks = importlib.util.module_from_spec(spec); spec.loader.exec_module(pairing_checks)
        verify_benchmark_sources, verify_layouts = pairing_checks.verify_benchmark_sources, pairing_checks.verify_layouts
        self.full, self.perception_services = full, perception_services
        self.job = os.environ.get('SLURM_JOB_ID')
        if not self.job or not self.job.isdigit():
            raise RuntimeError('terra evaluation must run in its own Slurm allocation')
        self.runtime = self.out/'allocations'/self.job
        self.runtime.mkdir(parents=True, exist_ok=False)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.financial_pause = threading.Event()
        self.broker = self.budget = self.broker_dir = None
        self.control_dir = None
        self.api_key_file = str(args.api_key_file.expanduser().absolute())
        self.children, self.completed, self.exhausted, self.attempts = {}, {}, {}, []
        self.environment_blocked = {}
        self.env = full.sanitized_environment(os.environ, self.root)
        for key in list(self.env):
            if any(part in key.upper() for part in ('AUTHKEY', 'AUTH_KEY', 'TAILSCALE_SOCKET', 'VLLM_API_KEY')):
                self.env.pop(key)
        self.env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            PYTHONPATH=os.pathsep.join(str(self.profile/p) for p in ('qwen_full','qwen_integration','ASPIRE/src')))
        self.env.update({name: '4' for name in THREAD_VARS})
        self.controller = {'model':MODEL['served_alias'],'base_url':MODEL['base_url'],'expected_context_length':262144}
        self.network = None
        gpu_csv = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv'], text=True)
        (self.runtime/'gpus.csv').write_text(gpu_csv)
        gpu_evidence = capture_gpu_identity_evidence(gpu_csv,os.environ)
        self.full.write_json(self.runtime/'gpu_identity_evidence.json',gpu_evidence)
        if gpu_evidence['metadata_read_errors']:
            raise RuntimeError('NVIDIA proc GPU metadata could not be read; identity receipt preserved')
        self.devices = resolve_four_gpus(gpu_csv,os.environ,proc_information=gpu_evidence['proc_nvidia_gpus'])
        gpu_evidence.update(verified=True,allocated_devices=self.devices)
        self.full.write_json(self.runtime/'gpu_identity_evidence.json',gpu_evidence)
        self.original_affinity = sorted(os.sched_getaffinity(0))
        job_details = subprocess.check_output(['scontrol','show','job','-dd',self.job], text=True)
        assigned = allocated_cpu_ids(self.original_affinity, job_details, socket.gethostname())
        self.affinity = cpu_plan(assigned)
        self.affinity['inherited_os_mask'] = self.original_affinity
        (self.runtime/'slurm_job_details.txt').write_text(job_details)
        if shutil.which('taskset') is None:
            raise RuntimeError('taskset is required for allocated CPU masks')
        self.requested_limit = int(os.environ['SLURM_MEM_PER_NODE'])*1024**2
        if self.requested_limit != 240*1024**3:
            raise RuntimeError('RAM request differs from frozen 240 GiB protocol')
        os.sched_setaffinity(0, self.affinity['shared'])
        self.report = {'campaign':self.campaign,'allocation':self.job,'host':socket.gethostname(),
            'started_utc':full.utc_now(),'complete':False,'mode':self.mode,
            'worker_count':9,'continuous_task_queue':True,
            'queue_granularity':'episode',
            'simulation_lifecycle':'fresh Isaac process per exact paired state',
            'protocol':self.manifest['protocol'],'manifest_sha256':sha_file(self.manifest_path),
            'prepaid_infrastructure_lineage':self.prepaid_lineage,
            'model':self.manifest['model'],'source_profile':self.manifest['source_profile'],
            'gpu_roles':{'simulator_slots':self.topology,'shared_perception_slot':3,'model':'Professor API via private rootless Tailscale'},
            'gpu_assignments':self.devices,'gpu_identity_evidence_sha256':sha_file(self.runtime/'gpu_identity_evidence.json'),'cpu_affinity':self.affinity,
            'thread_caps':{'simulator':2,'provider_and_controller':4,'variables':list(THREAD_VARS)},
            'offline_skill_discovery':False,'is_official_full_evaluation':False,
            'planned_scored_episodes':self.manifest['planned_scored_episodes'],'episode_walltime_limit':None,'model_turn_limit':None,
            'execution_stage':self.execution_stage,'stage_complete':False,
            'client_output_token_limit':None,'server_context_tokens':262144,'server_output_tokens':None,'api_cost_usd':None,
            'financial_policy':self.manifest['financial_policy'],'financial_paused':False,
            'budget_broker_stopped':None if self.user_managed_billing else False,
            'budget_broker_enabled':not self.user_managed_billing,
            'spending_control_enabled':not self.user_managed_billing,
            'cost_monitoring_enabled':not self.user_managed_billing,
            'financial_status_poll_seconds':None if self.user_managed_billing else FINANCIAL_STATUS_POLL_SECONDS,
            'memory_guard_poll_seconds':MEMORY_GUARD_POLL_SECONDS,
            'bootstrap_sha256':self.manifest['bootstrap_sha256'],'pairing_file_sha256':self.manifest['pairing_file_sha256'],
            'requested_memory_bytes':self.requested_limit,
            'provider_rng_note':'Shared provider RNG advances by request arrival order; no cross-episode learning.',
            'verified_benchmark_files':verify_benchmark_sources(self.manifest),
            'verified_layouts':verify_layouts({'groups':self.groups})}
        self.scheduler = {'campaign':self.campaign,'allocation':self.job,'manifest_sha256':sha_file(self.manifest_path),
            'execution_stage':self.execution_stage,
            'items':{g['group_id']:{'state':'pending','worker_id':None,
                'source_group_id':g['source_group_id']} for g in self.items}}
        self.save()

    def save(self):
        with self.lock:
            self.report['completed_items'] = self.completed
            self.report['exhausted_items'] = self.exhausted
            self.report['environment_blocked_groups'] = self.environment_blocked
            self.report['episode_attempts'] = self.attempts
            self.full.write_json(self.runtime/'allocation.json',self.report)
            self.full.write_json(self.runtime/'scheduler_state.json',self.scheduler)

    def record(self,event,**values):
        with self.lock:
            with (self.runtime/'events.jsonl').open('a') as handle:
                handle.write(json.dumps({'utc':self.full.utc_now(),'event':event,**values},sort_keys=True,allow_nan=False)+'\n')
                handle.flush()
                os.fsync(handle.fileno())

    def track(self,name,process):
        with self.lock:
            if name in self.children:
                raise RuntimeError('duplicate process identity')
            self.children[name] = process
        return process

    def cleanup(self,name,grace=20):
        with self.lock:
            process = self.children.get(name)
        if process is not None:
            details = self.full.stop_process_group(process,grace=grace)
            if process.poll() is None:
                raise RuntimeError('child remained live after cleanup: '+name)
            with self.lock:
                self.children.pop(name,None)
            self.record('child_stopped',name=name,details=details)

    def process_roots(self):
        with self.lock:
            return {'supervisor':os.getpid(),**{name:p.pid for name,p in self.children.items()}}

    def wait(self,process,required):
        while process.poll() is None:
            if self.stop.is_set():
                raise RuntimeError('allocation interrupted; native checkpoints and raw usage retained')
            for name,dependency in required:
                if dependency.poll() is not None:
                    raise RuntimeError(name+' exited; no automatic API/episode replay')
            time.sleep(.5)
        return process.returncode

    def collect_pending_groups(self):
        pending = queue.Queue()
        for group in self.items:
            state = inspect_group(self.out,group,self.simroot,self.full.validate_group)
            gid = group['group_id']
            if state['state']=='complete':
                self.completed[gid] = state['validation']
                self.scheduler['items'][gid] = {'state':'complete','worker_id':None,'validated_prior_result':state}
            elif state['state']=='exhausted':
                self.exhausted[gid] = state
                self.scheduler['items'][gid] = {'state':'missing_pairs','worker_id':None,'prior_evidence':state}
            else:
                pending.put(group)
        self.save()
        return pending

    def update_completion(self, infrastructure_complete):
        """Stage completion never supplies scores for deferred or exhausted groups."""
        expected = {g['group_id'] for g in self.items}
        self.report['infrastructure_complete'] = infrastructure_complete
        self.report['complete'] = infrastructure_complete and set(self.completed) == expected
        self.report['stage_complete'] = self.report['complete']
        self.report['terminal_with_missing_pairs'] = infrastructure_complete and bool(self.exhausted)
        self.report['scored_completed_items'] = sum(r['scored'] for r in self.completed.values())

    def poll_financial_pause(self):
        if getattr(self,'user_managed_billing',False): return False
        self._last_financial_status_poll=time.monotonic()
        status = self.budget.status()
        signalled = (self.broker_dir/'pause.json').exists()
        blocked = any(status.get(k) for k in ('admission_paused','admission_blocked_unknown','admission_blocked_integrity'))
        if signalled or blocked or self.financial_pause.is_set():
            if not self.financial_pause.is_set():
                self.financial_pause.set()
                self.budget.pause_admission(reason='supervisor_financial_pause_drain')
                self.report['financial_paused'] = True
                self.record('financial_pause_started',budget_status=status)
            # Read after the broker admission latch: a reservation may have
            # entered between the first status read and pause_admission.
            status = self.budget.status()
            self.report['budget_status_during_drain'] = status
            # Already reserved calls may finish and settle. No new generation is
            # admitted. Only after this drain do we stop simulators/controllers.
            inflight = sorted(p.name for p in self.broker_dir.glob('inflight_*.json'))
            self.report['financial_inflight_markers'] = inflight
            if status['request_counts']['active'] == 0 and not inflight:
                self.report['financial_drain_complete'] = True
                self.stop.set()
            self.save()
        return self.financial_pause.is_set()

    def poll_periodic_financial_pause(self, now=None):
        """Reduce background status RPCs; reservation/settlement stays immediate.

        Explicit pause signals and an already active drain remain urgent. Startup
        and worker errors still call poll_financial_pause directly.
        """
        if getattr(self,'user_managed_billing',False):return False
        now=time.monotonic() if now is None else now
        last=getattr(self,'_last_financial_status_poll',None)
        if (last is None or now-last>=FINANCIAL_STATUS_POLL_SECONDS
                or self.financial_pause.is_set() or (self.broker_dir/'pause.json').exists()):
            return self.poll_financial_pause()
        return False

    def monitor_guard_sample(self, observed, now=None):
        decision=memory_guard(observed,self.requested_limit)
        if not decision['safe']:
            self.report['memory_guard_triggered']={'observation':observed,'decision':decision}
            self.record('memory_guard_triggered',**self.report['memory_guard_triggered'])
            self.stop.set();return False
        if not getattr(self,'user_managed_billing',False):self.poll_periodic_financial_pause(now=now)
        return True

    def private_control_directory(self):
        if getattr(self,'user_managed_billing',False):
            if self.control_dir is None:raise RuntimeError('private control directory missing')
            return self.control_dir
        return self.broker.socket_path.parent

    def private_control_probe_path(self):
        if getattr(self,'user_managed_billing',False):
            return self.private_control_directory()/'host-control-sentinel'
        return self.broker.socket_path

    def controller_configuration(self,group,worker,usage_path):
        config=dict(self.controller,usage_jsonl=usage_path,episode_ids=group['episode_ids'],
            allocation_id=self.job,worker_id=worker,
            episode_seeds={eid:group['eval_seed'] for eid in group['episode_ids']})
        if getattr(self,'user_managed_billing',False):
            config['billing_mode']='user_managed'
            config['network_socket']=str(self.network.socket)
        else:
            config.update(budget_socket=str(self.broker.socket_path),pause_file=str(self.broker_dir/'pause.json'))
        return config

    def run_group(self,group,worker,slot,services):
        gid = group['group_id']
        group_out = self.out/'items'/gid
        attempt = self.runtime/'workers'/worker/gid
        state = self.root/'full_runtime'/self.campaign/self.job/worker
        relay = sim = controller = None
        prefix = worker+'_'+gid
        started = time.monotonic()
        row = {'worker_id':worker,'group_id':gid,'episode_id':group['episode_id'],
            'source_group_id':group['source_group_id'],'task':group['task'],'eval_seed':group['eval_seed'],
            'gpu_slot':slot,'gpu_uuid':self.devices[slot]['uuid'],'native_campaign':self.campaign,
            'eligible_layout_ids':group['candidate_layout_ids'],'target_episodes':group['target_episodes'],
            'complete':False,'usage_path':str(attempt/'usage.jsonl'),'episode_report_paths':[],
            'requested_cpu_mask':self.affinity['simulators'][worker],'simulator_log_path':str(attempt/'simulator.log')}
        try:
            group_out.mkdir(parents=True,exist_ok=True)
            attempt.mkdir(parents=True,exist_ok=False)
            # Preserve previous report bytes before this explicit resume writes progress.
            prior_reports = [p for p in group_out.iterdir() if p.name in
                ('group_report.json','group_progress.json','terra_group_completion.json','terra_contract.json','terra_episode_timing.json','kit_auxiliary_policy.json',
                 'ngx_reaper_preflight.json','ngx_child_reaper.jsonl')]
            if prior_reports:
                (attempt/'prior_reports').mkdir()
                for p in prior_reports:shutil.copyfile(p,attempt/'prior_reports'/p.name)
            runtime_state_directories(state)
            relay = self.full.create_private_relay(self.job,gid)
            self.full.write_json(attempt/'relay_location.json',{'live_node_local_path':str(relay),
                'host':socket.gethostname(),'archive_path':str(attempt/'relay')})
            if self.stop.is_set() or self.financial_pause.is_set():raise RuntimeError('allocation admission stopped before group startup')
            config = self.controller_configuration(group,worker,row['usage_path'])
            self.full.write_json(attempt/'controller_config.json',config)
            with (attempt/'controller.log').open('w') as handle:
                controller = self.track(prefix+'_controller',subprocess.Popen([
                    '/usr/bin/python3.12','-B',str(self.code/'qwen_driver.py'),'--config',
                    str(attempt/'controller_config.json'),'--relay-dir',str(relay)],env=self.env,
                    cwd=self.profile,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True))
            deadline = time.monotonic()+60
            while not (relay/'ready.json').exists():
                if controller.poll() is not None or self.stop.is_set() or time.monotonic()>deadline:
                    raise RuntimeError('remote controller did not become ready')
                time.sleep(.1)
            sim_env = self.full.restore_simulator_proxies(self.env,os.environ)
            sim_env.update(CUDA_VISIBLE_DEVICES=self.devices[slot]['uuid'],ACCEPT_EULA='Y',
                ROBODOJO_STATE_DIR=str(state),SINGULARITY_BINDPATH=container_bindings(state,relay,self.private_control_directory()),
                SINGULARITYENV_ASPIRE_TERRA_ALLOCATION=self.job,
                SINGULARITYENV_ASPIRE_SIM_LOG=str(Path('/work')/(attempt/'simulator.log').relative_to(self.simroot)),
                SINGULARITYENV_ASPIRE_PERCEPTION_CONFIG=str(Path('/work')/(self.runtime/'providers.json').relative_to(self.simroot)))
            for name in THREAD_VARS:
                sim_env[name]='2';sim_env['SINGULARITYENV_'+name]='2'
            inside = Path('/work')/(self.code/'terra_inside.sh').relative_to(self.simroot)
            command = ['taskset','--cpu-list',','.join(map(str,self.affinity['simulators'][worker])),
                'bash',str(self.simroot/'scripts/container_exec.sh'),'/bin/bash',str(inside),
                self.campaign,worker,gid,group['task'],str(group['eval_seed']),self.devices[slot]['index'],
                self.devices[slot]['uuid'],str(group['target_episodes']),
                ','.join(map(str,group['candidate_layout_ids'])),self.manifest['model']['served_alias']]
            row['simulator_started_utc']=self.full.utc_now()
            self.record('episode_started',group_id=gid,episode_id=group['episode_id'],
                source_group_id=group['source_group_id'],worker_id=worker,gpu_slot=slot,
                target_episodes=group['target_episodes'],cpu_mask=row['requested_cpu_mask'],command=command)
            with (attempt/'simulator.log').open('w') as handle:
                sim=self.track(prefix+'_sim',subprocess.Popen(command,env=sim_env,cwd=self.simroot,
                    stdout=handle,stderr=subprocess.STDOUT,start_new_session=True))
            time.sleep(.15)
            if sim.poll() is None:
                row['observed_launcher_cpu_mask']=sorted(os.sched_getaffinity(sim.pid))
                if row['observed_launcher_cpu_mask']!=row['requested_cpu_mask']:
                    raise RuntimeError('simulator CPU mask differs from allocated slot')
            rc=self.wait(sim,[('remote controller',controller),*services])
            row.update(simulator_finished_utc=self.full.utc_now(),simulator_exit_code=rc)
            inspected=inspect_group(self.out,group,self.simroot,self.full.validate_group)
            row['native_state']=inspected
            if inspected['state']=='complete':
                if rc!=0:raise RuntimeError('native scores complete but process exited abnormally; preserve and audit')
                row['native_result']=inspected['validation'];row['complete']=True
            elif inspected['state']=='exhausted':
                row['candidate_exhausted']=True
                row['completion_shortfall']=group['target_episodes']-inspected['scored']
            else:
                raise RuntimeError(f'simulator exited {rc} before group completion; native checkpoint retained')
            row['execution_finished']=True
        except BaseException as exc:
            row['error']={'type':type(exc).__name__,'message':str(exc)}
            try:
                if not getattr(self,'user_managed_billing',False) and self.poll_financial_pause(): row['financial_paused']=True
                else: self.stop.set()
            except BaseException:
                self.stop.set()
        finally:
            cleanup_errors=[]
            for name,grace in ((prefix+'_sim',45),(prefix+'_controller',20)):
                try:self.cleanup(name,grace=grace)
                except BaseException as exc:
                    cleanup_errors.append({'operation':name,'type':type(exc).__name__,'message':str(exc)})
                    self.stop.set()
            row.setdefault('simulator_finished_utc',self.full.utc_now())
            row['lifecycle_wall_seconds']=time.monotonic()-started
            try:
                timing_path = group_out/'terra_episode_timing.json'
                if timing_path.exists():
                    timing = json.loads(timing_path.read_text())
                    # Earlier attempt timings are archived, not attributed to this allocation.
                    launch = datetime.fromisoformat(row['simulator_started_utc']) if row.get('simulator_started_utc') else None
                    entered = datetime.fromisoformat(timing['episode_started_utc'])
                    if launch is not None and entered >= launch:
                        row['episode_timing'] = timing
                        row['simulator_startup_wall_seconds'] = (entered-launch).total_seconds()
                        if timing.get('episode_callback_finished_utc'):
                            exited = datetime.fromisoformat(timing['episode_callback_finished_utc'])
                            row['episode_callback_wall_seconds'] = (exited-entered).total_seconds()
                            row['simulator_post_episode_wall_seconds'] = (datetime.fromisoformat(row['simulator_finished_utc'])-exited).total_seconds()
            except BaseException as exc:
                cleanup_errors.append({'operation':'episode_timing_receipt','type':type(exc).__name__,'message':str(exc)})
                self.stop.set()
            row['episode_report_paths']=[str(p) for p in sorted((group_out/'episodes').glob('*/attempt_*/episode_report.json'))]
            try:
                endpoints_stopped=all(p is None or p.poll() is not None for p in (sim,controller))
                if relay is not None and endpoints_stopped:
                    shutil.copytree(relay,attempt/'relay');shutil.rmtree(relay)
                    self.record('relay_archived',group_id=gid,worker_id=worker,path=str(attempt/'relay'))
                elif relay is not None:raise RuntimeError('relay endpoints still active; private relay retained')
            except BaseException as exc:
                cleanup_errors.append({'operation':'relay_archive','type':type(exc).__name__,'message':str(exc)})
                self.stop.set()
            if cleanup_errors:row['cleanup_errors']=cleanup_errors
            row['infrastructure_complete']=row.get('execution_finished',False) and not row.get('error') and not cleanup_errors
            if not row['infrastructure_complete'] and not row.get('financial_paused'):self.stop.set()
            # Each allocation/worker/group attempt is immutable once finalized.
            try:
                attempt.mkdir(parents=True,exist_ok=True)
                self.full.write_json(attempt/'group_attempt.json',row)
                self.full.write_json(group_out/'latest_allocation_attempt.json',{'allocation':self.job,
                    'worker_id':worker,'report':str(attempt/'group_attempt.json')})
            except BaseException as exc:
                row['persistence_error']={'type':type(exc).__name__,'message':str(exc)}
                row['infrastructure_complete']=False;self.stop.set()
            with self.lock:
                self.attempts.append(row)
                if row['infrastructure_complete'] and row['complete']:
                    self.completed[gid]=row['native_result'];status='complete'
                elif row['infrastructure_complete'] and row.get('candidate_exhausted'):
                    self.exhausted[gid]=row['native_state'];status='missing_pairs'
                else:status='financial_paused' if row.get('financial_paused') else 'interrupted'
                self.scheduler['items'][gid]={'state':status,'worker_id':worker,'attempt_report':str(attempt/'group_attempt.json')}
                self.save()
            self.record('episode_finalized',group_id=gid,episode_id=group['episode_id'],
                source_group_id=group['source_group_id'],worker_id=worker,report=row)
        return row

    def claim_and_run(self,group,worker,slot,services):
        gid=group['group_id']
        with self.lock:
            if self.scheduler['items'][gid]['state'] != 'pending':
                raise RuntimeError('duplicate or non-pending episode claim')
            self.scheduler['items'][gid]={'state':'running','worker_id':worker,
                'source_group_id':group['source_group_id'],'claimed_utc':self.full.utc_now()}
            self.save()
        self.record('episode_claimed',group_id=gid,episode_id=group['episode_id'],
            source_group_id=group['source_group_id'],worker_id=worker,gpu_slot=slot)
        return self.run_group(group,worker,slot,services)

    def run_slot(self,worker,slot,pending,services):
        while not self.stop.is_set() and not self.financial_pause.is_set():
            try:group=pending.get_nowait()
            except queue.Empty:return
            row=self.claim_and_run(group,worker,slot,services)
            pending.task_done()
            if not row['infrastructure_complete']:return

    def compute_preflight(self):
        if getattr(self,'user_managed_billing',False):
            result=synthetic_worker_preflight()
        else:
            directory=self.full.create_private_relay(self.job,'terra-synthetic')
            result=synthetic_broker_preflight(directory,self.runtime/'preflight/synthetic_budget')
        result['allocation']=self.job
        self.full.write_json(self.runtime/'compute_preflight.json',result)
        self.report['compute_node_integration_verified']=True
        self.save()

    def container_preflight(self):
        manual=getattr(self,'user_managed_billing',False)
        control_key='private_control_visible' if manual else 'budget_socket_visible'
        probe=CONTAINER_VISIBILITY_PROBE.replace('budget_socket_visible',control_key)
        probe_path=self.private_control_probe_path()
        home=str(Path(pwd.getpwuid(os.getuid()).pw_dir))
        launcher=self.simroot/'scripts/container_exec.sh'
        for index,slot in enumerate(self.topology):
            worker=f'w{index:02d}'
            state=runtime_state_directories(self.root/'full_runtime'/self.campaign/self.job/worker)
            nonce=uuid.uuid4().hex
            markers=[state/path/'.terra-preflight-marker' for path in ('runtime-home','private-tmp','private-var-tmp')]
            for marker in markers:marker.write_text(nonce)
            relay=self.full.create_private_relay(self.job,'terra-visibility-'+worker)
            process=None;stdout=_stderr='';receipt_saved=False
            try:
                env=self.full.restore_simulator_proxies(self.env,os.environ)
                env.update(CUDA_VISIBLE_DEVICES=self.devices[slot]['uuid'],ACCEPT_EULA='Y',
                    ROBODOJO_STATE_DIR=str(state),SINGULARITY_BINDPATH=container_bindings(state,relay,self.private_control_directory()))
                if self.stop.is_set():raise RuntimeError('allocation stopped before container preflight')
                process=self.track('preflight_'+worker,subprocess.Popen(
                    ['taskset','--cpu-list',','.join(map(str,self.affinity['simulators'][worker])),
                    'bash',str(launcher),CONTAINER_PYTHON,'-c',probe,
                    self.api_key_file,str(probe_path),home,nonce],env=env,cwd=self.simroot,
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True))
                deadline=time.monotonic()+180
                while True:
                    if self.stop.is_set() or time.monotonic()>deadline:
                        raise RuntimeError('container preflight interrupted or startup deadline exceeded')
                    try:
                        stdout,_stderr=process.communicate(timeout=1)
                        break
                    except subprocess.TimeoutExpired as exc:
                        stdout=exc.output or stdout
                        _stderr=exc.stderr or _stderr
                if self.stop.is_set():raise RuntimeError('allocation stopped during container preflight')
                candidates=[line for line in stdout.splitlines() if line.startswith('{')]
                parse_error=None
                try:
                    value=json.loads(candidates[-1]) if candidates else {}
                except (ValueError,TypeError) as exc:
                    value={};parse_error=type(exc).__name__
                expected={'key_visible':False,control_key:False,'relay_visible':True,
                    'private_home_verified':True,'private_tmp_verified':True,'private_var_tmp_verified':True}
                verified=process.returncode==0 and value==expected
                receipt={'schema_version':1,'allocation':self.job,'worker_id':worker,'gpu_slot':slot,
                    'verified':verified,'exit_code':process.returncode,'container_launcher_sha256':sha_file(launcher),
                    'python_launcher':CONTAINER_PYTHON,
                    **({'billing_mode':'user_managed','actual_control_kind':'non_socket_sentinel'} if manual else {}),
                    **{key:value[key] for key in expected if isinstance(value,dict) and type(value.get(key)) is bool}}
                if not verified:
                    private=(self.api_key_file,str(probe_path),str(self.private_control_directory()),nonce)
                    receipt['diagnostics']={'stdout_tail':sanitized_preflight_tail(stdout,private_values=private),
                        'stderr_tail':sanitized_preflight_tail(_stderr,private_values=private),
                        'parse_error':parse_error,'max_bytes_per_stream':4096,'credentials_redacted':True}
                # Keep boolean visibility and bounded sanitized startup diagnostics only.
                self.full.write_json(self.runtime/f'container_preflight_{worker}.json',receipt)
                receipt_saved=True
                if not verified:raise RuntimeError('container credential/control isolation preflight failed for '+worker)
            except BaseException as exc:
                if not receipt_saved:
                    private=(self.api_key_file,str(probe_path),str(self.private_control_directory()),nonce)
                    self.full.write_json(self.runtime/f'container_preflight_{worker}.json',{
                        'schema_version':1,'allocation':self.job,'worker_id':worker,'gpu_slot':slot,
                        'verified':False,'exit_code':process.poll() if process is not None else None,
                        'container_launcher_sha256':sha_file(launcher),'python_launcher':CONTAINER_PYTHON,
                        'diagnostics':{'stdout_tail':sanitized_preflight_tail(stdout,private_values=private),
                            'stderr_tail':sanitized_preflight_tail(_stderr,private_values=private),
                            'error_type':type(exc).__name__,'error_message':sanitized_preflight_tail(str(exc),private_values=private),
                            'max_bytes_per_stream':4096,'credentials_redacted':True}})
                raise
            finally:
                self.cleanup('preflight_'+worker,grace=20)
                for marker in markers:marker.unlink(missing_ok=True)
                shutil.rmtree(relay)
        self.report['container_isolation_verified']=True
        self.save()

    def verify_initial_api_compatibility(self):
        """Audit the first scored pair, independent of its success/failure outcome."""
        if getattr(self,'user_managed_billing',False):
            return self.verify_initial_api_compatibility_user_managed()
        from audit_qwen_results import verify_item,verify_infrastructure_phases
        first=self.items[0];issues=[]
        allocations=[]
        for directory in sorted((self.out/'allocations').glob('*')):
            path=directory/'allocation.json'
            if path.is_file():allocations.append((directory,json.loads(path.read_text())))
        lineage=verify_infrastructure_phases(self.out,self.manifest,allocations,issues)
        if not lineage['verified']:raise RuntimeError('initial API gate requires verified infrastructure lineage')
        acknowledged=set(lineage.get('acknowledged_unknown_request_ids',[]))
        evidence=verify_item(self.out,self.simroot,self.manifest,first,allocations,issues,lineage)
        native=evidence['native'];usages=evidence['usage']
        blocking=[i for i in issues if not (i.get('severity')=='warning'
            and i.get('message')=='Provider usage remains unknown, not zero.'
            and any(Path(i.get('path','')).name=='request_'+rid+'.sse' for rid in acknowledged))]
        if (blocking or native['scored']!=1 or native['three_camera_episodes']!=1
                or evidence['verified_report_count']!=1):
            raise RuntimeError('initial paired episode failed native/provider/evidence compatibility checks')
        committed=[r for r in evidence['episode_reports'] if r['native_score_committed']]
        report=json.loads(Path(committed[0]['path']).read_text())
        if not report.get('iterations'):
            raise RuntimeError('initial paired episode did not accept a structured model proposal')
        requests={rid:value for usage in usages for rid,value in usage['requests'].items()}
        known_requests={rid:row for rid,row in requests.items() if rid not in acknowledged}
        if not known_requests or set(requests)-set(known_requests)!=acknowledged or any(not r.get('usage') for r in known_requests.values()):
            raise RuntimeError('initial paid compatibility has missing or unknown usage')
        snapshot=read_only_snapshot(self.out/'budget/ledger.jsonl',cap_usd='2000')
        for rid,request in known_requests.items():
            saved=snapshot['requests'].get(rid,{})
            settlement=saved.get('settlement',{})
            if (saved.get('state')!='settled' or not request['settlement_event_confirmed']
                    or settlement.get('model')!=MODEL['served_alias'] or settlement.get('service_tier')!='default'
                    or not request.get('usage') or settlement.get('raw_usage')!=request['usage']['raw_usage']):
                raise RuntimeError('initial compatibility requires durable known Standard-price settlements')
        streams=[stream for usage in usages for stream in usage['raw_streams']]
        if not any(stream.get('status')=='completed' and stream.get('terminal_event')=='response.completed'
                   and stream.get('model')==MODEL['served_alias'] and stream.get('http_status')==200 for stream in streams):
            raise RuntimeError('initial compatibility lacks a real completed Terra Responses stream')
        receipt={'schema_version':1,'campaign':self.campaign,'allocation':self.job,
            'manifest_sha256':sha_file(self.manifest_path),'episode_id':first['episode_id'],
            'source_group_id':first['source_group_id'],'verified':True,'quality_threshold_applied':False,
            'verified_report_count':1,'native_scored':1,'video_count':3,
            'requests_started':len(requests),'requests_with_usage':len(known_requests),'requests_unknown':len(acknowledged),
            'acknowledged_unknown_request_ids':sorted(acknowledged),
            'held_historical_unknown_exposure_usd':lineage.get('held_historical_unknown_exposure_usd','0'),
            'settled_request_ids':sorted(known_requests),'native_result_sha256':native['sha256'],
            'model':MODEL['served_alias'],'reasoning_effort':'medium','verified_utc':self.full.utc_now()}
        initial=self.out/'source_evidence/initial_api_compatibility.json'
        if not initial.exists():self.full.write_json(initial,receipt)
        else:
            prior=json.loads(initial.read_text())
            for key in ('manifest_sha256','episode_id','source_group_id','native_result_sha256','settled_request_ids','acknowledged_unknown_request_ids','held_historical_unknown_exposure_usd'):
                if prior.get(key)!=receipt[key]:raise RuntimeError('initial paid compatibility evidence changed on resume')
        self.full.write_json(self.runtime/'initial_api_compatibility.json',receipt)
        self.report['paid_api_compatibility_verified']=True
        self.report['compatibility_scope']='Preserved native/shared harness proof from prior local phase; restored API is separately verified by network metadata and each exact-model response'
        self.report['initial_api_compatibility_episode']=first['episode_id']
        self.save()
        self.record('initial_api_compatibility_verified',episode_id=first['episode_id'],quality_threshold_applied=False)

    def verify_initial_api_compatibility_user_managed(self):
        """Reuse the scored first state while omitting all spending/usage gates."""
        from audit_qwen_results import verify_infrastructure_phases, verify_manual_initial_compatibility
        allocations=[];issues=[]
        for directory in sorted((self.out/'allocations').glob('*')):
            path=directory/'allocation.json'
            if path.is_file():allocations.append((directory,json.loads(path.read_text())))
        lineage=verify_infrastructure_phases(self.out,self.manifest,allocations,issues)
        if not lineage['verified']:raise RuntimeError('initial API gate requires verified source lineage')
        proof=verify_manual_initial_compatibility(self.out,self.simroot,self.manifest,self.items[0],allocations,issues,lineage)
        if not isinstance(proof,dict) or proof.get('verified') is not True or any(i.get('severity')=='error' for i in issues):
            raise RuntimeError('initial paired episode failed native/provider/evidence checks')
        receipt={**proof,'schema_version':1,'campaign':self.campaign,'allocation':self.job,
            'manifest_sha256':sha_file(self.manifest_path),'mode':'user_managed',
            'spending_control_enabled':False,'verified_utc':self.full.utc_now()}
        if self.manifest.get('harness_revision',{}).get('version')=='robodojo_task_reasoning_v2':
            initial=self.out/'source_evidence/initial_api_compatibility.json'
            if not initial.exists():self.full.write_json(initial,receipt)
            else:
                prior=json.loads(initial.read_text())
                if any(prior.get(k)!=v for k,v in proof.items()):
                    raise RuntimeError('initial API compatibility evidence changed on resume')
        self.full.write_json(self.runtime/'initial_api_compatibility.json',receipt)
        self.report['paid_api_compatibility_verified']=True
        self.report['compatibility_scope']='Preserved native/shared harness proof from prior local phase; restored API is separately verified by network metadata and each exact-model response'
        self.report['initial_api_compatibility_episode']=self.items[0]['episode_id']
        self.save()
        self.record('initial_api_compatibility_verified',episode_id=self.items[0]['episode_id'],quality_threshold_applied=False)

    def first_episode_gate(self,pending,services):
        first=self.items[0]
        if first['group_id'] not in self.completed:
            if first['group_id'] in self.exhausted:
                raise RuntimeError('first exact paired state is an unscored missing pair; no substitute probe')
            try:item=pending.get_nowait()
            except queue.Empty:raise RuntimeError('initial compatibility item missing from pending queue')
            if item['episode_id']!=first['episode_id']:
                raise RuntimeError('first real API gate must use static first paired state')
            row=self.claim_and_run(item,'w00',self.topology[0],services)
            pending.task_done()
            if not row['infrastructure_complete'] or not row['complete']:
                raise RuntimeError('first paid paired episode did not complete; no broad admission')
        self.verify_initial_api_compatibility()

    def network_preflight(self):
        from qwen_network import PrivateTailnet
        self.network=PrivateTailnet(self.control_dir/'tailnet')
        proof=self.network.start(stop_requested=self.stop.is_set)
        receipt=dict(proof,verified=True,base_url=MODEL['base_url'],model=MODEL['served_alias'],
            allocation=self.job,campaign=self.campaign,manifest_sha256=sha_file(self.manifest_path),
            verified_utc=self.full.utc_now(),underlying_checkpoint_verified=False)
        self.full.write_json(self.runtime/'network_preflight.json',receipt)
        self.report['network_preflight_verified']=True
        self.report['inference_endpoint']=receipt
        self.save()

    def run(self):
        from scaling_metrics import MetricsCollector,cgroup_memory_sample
        pending=self.collect_pending_groups()
        self.compute_preflight()
        if getattr(self,'user_managed_billing',False):
            self.control_dir=self.full.create_private_relay(self.job,'terra-control')
            (self.control_dir/'host-control-sentinel').write_text('private host control; no socket or financial service\n')
            self.report['private_control_kind']='non_socket_sentinel'
        else:
            self.broker_dir=self.full.create_private_relay(self.job,'terra-budget')
            self.broker=BudgetBroker(self.out/'budget/ledger.jsonl',self.broker_dir,cap_usd='2000')
            self.report['budget_broker_start']=self.broker.start()
            self.budget=BudgetClient(self.broker.socket_path)
        self.network_preflight()
        if pending.empty():
            self.verify_initial_api_compatibility()
            self.update_completion(True)
            self.save();return
        if not getattr(self,'user_managed_billing',False) and self.poll_financial_pause():
            raise RuntimeError('financial admission is paused; preserve unscored checkpoints')
        self.container_preflight()
        proxy=SimpleNamespace(root=self.root,runtime=self.runtime,source_profile=self.profile,
            manifest={'profiles':{'perception':self.manifest['source_profile']},
                'perception_artifacts':self.manifest['perception_artifacts']},env=self.env,
            gpus=self.devices,stop=self.stop,report=self.report,track=self.track,record=self.record)
        collector=MetricsCollector(self.runtime/'telemetry.jsonl',gpu_uuids=[g['uuid'] for g in self.devices],
            root_pids=self.process_roots,interval_seconds=2,memory_limit_bytes=self.requested_limit)
        monitor_stop=threading.Event()
        def monitor():
            while not monitor_stop.is_set():
                try:
                    if not self.monitor_guard_sample(cgroup_memory_sample()):return
                except BaseException as exc:
                    self.report['memory_guard_error']={'type':type(exc).__name__,'message':str(exc)}
                    self.stop.set();return
                monitor_stop.wait(MEMORY_GUARD_POLL_SECONDS)
        watcher=threading.Thread(target=monitor,name='terra-memory-guard',daemon=True)
        collector.start();watcher.start()
        try:
            providers,services=self.perception_services.start_services(proxy)
            if providers['grasp_calibration']!=self.manifest['grasp_calibration']:
                raise RuntimeError('provider calibration differs from frozen manifest')
            self.first_episode_gate(pending,services)
            with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
                futures=[pool.submit(self.run_slot,f'w{i:02d}',slot,pending,services) for i,slot in enumerate(self.topology)]
                for future in concurrent.futures.as_completed(futures):
                    try:future.result()
                    except BaseException:
                        self.stop.set();raise
        finally:
            monitor_stop.set();watcher.join(timeout=10)
            self.report['resource_summary']=collector.stop()
            self.save()
        self.update_completion(not self.stop.is_set() and not self.financial_pause.is_set() and pending.empty())
        self.save()
        if not self.report['infrastructure_complete']:
            raise RuntimeError('terra allocation interrupted; explicit inspected native resume required')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--api-key-file',type=Path,default=Path('/home/ROBODATA_USER/.config/professor-api-pilot/tailscaled.state'))
    parser.add_argument('--mode',choices=('full','resume'),default='full')
    args=parser.parse_args();os.umask(0o077)
    out=args.root/'full_results'/args.campaign
    with (out/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign=TerraCampaign.__new__(TerraCampaign)
        try:
            campaign.__init__(args)
        except BaseException as exc:
            # Initialization performs metadata/source checks only and has not
            # spawned providers/controllers/simulators. Retain a narrow receipt
            # if its allocation directory was already created.
            if hasattr(campaign,'runtime') and campaign.runtime.is_dir():
                receipt=campaign.runtime/'bootstrap_failure.json'
                with receipt.open('x') as handle:
                    json.dump({'allocation':getattr(campaign,'job',None),
                        'error_type':type(exc).__name__,'message':str(exc),
                        'children_started':False,'model_requests':0},handle,sort_keys=True)
                    handle.write('\n')
            if hasattr(campaign,'original_affinity'):
                os.sched_setaffinity(0,campaign.original_affinity)
            raise
        def interrupt(signum,_frame):
            campaign.stop.set();campaign.record('allocation_signal',signal=signum)
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,interrupt)
        rc=1
        try:
            campaign.run();rc=0
        except BaseException as exc:
            campaign.report['error']={'type':type(exc).__name__,'message':str(exc)}
            campaign.record('allocation_interrupted',error=campaign.report['error'])
        finally:
            with campaign.lock:names=list(campaign.children)
            for name in reversed(names):
                try:campaign.cleanup(name)
                except BaseException as exc:
                    campaign.report.setdefault('cleanup_errors',[]).append({'name':name,'type':type(exc).__name__,'message':str(exc)})
                    campaign.report['complete']=False
                    campaign.report['stage_complete']=False
                    campaign.report['infrastructure_complete']=False
                    rc=1
            if campaign.broker is not None:
                save_budget_cache_integrity_diagnostics(campaign,'before_stop')
                try:
                    campaign.broker.stop()
                    campaign.report['budget_broker_stopped']=True
                    campaign.report['budget_broker_final_status']=read_only_snapshot(campaign.out/'budget/ledger.jsonl',cap_usd='2000')['status']
                    remaining = sorted(p.name for p in campaign.broker_dir.glob('inflight_*.json'))
                    campaign.report['budget_remaining_inflight_markers'] = remaining
                    archive = campaign.runtime/'budget_control_archive'
                    shutil.copytree(campaign.broker_dir,archive)
                    campaign.report['budget_control_archive'] = str(archive)
                    shutil.rmtree(campaign.broker_dir)
                except BaseException as exc:
                    campaign.report.setdefault('cleanup_errors',[]).append({'name':'budget_broker','type':type(exc).__name__,'message':str(exc)})
                    campaign.report['complete']=False;rc=1
                save_budget_cache_integrity_diagnostics(campaign,'after_stop')
            if campaign.network is not None:
                try:campaign.report['network_cleanup']=campaign.network.close()
                except BaseException as exc:
                    campaign.report.setdefault('cleanup_errors',[]).append({'name':'private_tailscale','type':type(exc).__name__,'message':str(exc)})
                    campaign.report['complete']=False;rc=1
            if getattr(campaign,'user_managed_billing',False) and campaign.control_dir is not None:
                try:
                    if campaign.children:raise RuntimeError('child endpoints remain active; retain private control directory')
                    archive=campaign.runtime/'private_control_archive'
                    shutil.copytree(campaign.control_dir,archive)
                    shutil.rmtree(campaign.control_dir)
                    campaign.report['private_control_archive']=str(archive)
                    campaign.report['control_directory_cleaned']=True
                except BaseException as exc:
                    campaign.report.setdefault('cleanup_errors',[]).append({'name':'private_control','type':type(exc).__name__,'message':str(exc)})
                    campaign.report['complete']=False;rc=1
            if campaign.report.get('financial_paused') and not campaign.report.get('cleanup_errors'): rc=75
            campaign.report.update(finished_utc=campaign.full.utc_now(),exit_code=rc,
                children_stopped=not campaign.children)
            campaign.save()
            os.sched_setaffinity(0,campaign.original_affinity)
        return rc


if __name__=='__main__':raise SystemExit(main())
