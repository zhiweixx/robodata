"""One exclusive native group per Slurm worker, with node-local services.

Replica scaling uses independent group owners. No cross-node model RPC, episode
caps, result-based retries or per-layout replacement races are introduced.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from urllib import request

from robodata.data.provenance import verify_source, canonical_sha256
from .manifest import load_manifest, get_group, verify_sources

MODEL = {'repo':'Qwen/Qwen3.8-27B','revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'}
INFERENCE = {'dtype':'bfloat16','tensor_parallel_size':2,'mtp_tokens':3,'native_context_tokens':262144,'max_num_seqs':2,'model_seed':0}
CALIBRATION = {'native_closing_direction':[0,1,0], 'source_id':'X5A.urdf sha256:7aaa308a1a5b6a213d914db86ba09ddd6252f25cb498ac9a148bfa6c7166a423 joint7/8','maximum_contact_separation_m':None}


def utc_now(): return datetime.now(timezone.utc).isoformat()
def sha_file(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def server_command(python, snapshot, port):
    return [str(python),'-m','vllm.entrypoints.openai.api_server','--model',str(snapshot),
        '--served-model-name',MODEL['repo'],'--dtype','bfloat16','--tensor-parallel-size','2',
        '--max-num-seqs','2','--max-model-len','262144','--max-num-batched-tokens','8192',
        '--enable-chunked-prefill','--limit-mm-per-prompt',json.dumps({'image':262144,'video':0}),
        '--skip-mm-profiling','--gpu-memory-utilization','0.85','--reasoning-parser','qwen3',
        '--host','127.0.0.1','--port',str(port),'--speculative-config',
        json.dumps({'method':'mtp','num_speculative_tokens':3}),'--seed','0']


def unused_port():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def native_result_path(campaign_dir, group):
    campaign_dir=Path(campaign_dir)
    return (campaign_dir/'native/RoboDojo'/group['task']/'ASPIRE_Pilot/arx_x5'/
        f"{group['eval_seed']}_aspire_qwen_full"/f"{campaign_dir.name}_{group['group_id']}"/'_result.json')


def resume_status(campaign_dir, group):
    from .audit import native_details, video_evidence
    native=native_result_path(campaign_dir,group)
    checkpoint=native.parent.parent/('_resume_'+native.parent.name+'.json')
    details=native_details(json.loads(native.read_text()),group) if native.exists() else {}
    completed={r['layout_id'] for r in details.values()}
    if len(details)==group['target_episodes']:
        videos=video_evidence(native.parent,details)
        if videos['verified_scored_episodes']!=len(details) or videos['invalid_or_unexpected_videos']:
            raise RuntimeError('scores complete but native video evidence needs repair; never replay')
        return 'complete'
    if completed and not checkpoint.exists(): raise RuntimeError('scored layouts have no native checkpoint; never replay')
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text()); candidates=set(group['candidate_layout_ids'])
        done=saved.get('completed_layout_ids',[]); abandoned=saved.get('abandoned_layout_ids',[])
        for values in (done,abandoned):
            if (not isinstance(values,list) or any(type(i)is not int for i in values)
                    or len(values)!=len(set(values)) or not set(values)<=candidates):
                raise RuntimeError('checkpoint contains invalid candidate IDs')
        if set(done)!=completed or completed & set(abandoned): raise RuntimeError('checkpoint/native score mismatch')
        if not candidates-completed-set(abandoned): raise RuntimeError('unique candidate pool exhausted; no rerun or synthetic replacements')
    return 'pending'


def resolve_gpus(config, env, *, output=None):
    tokens=env.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(tokens)!=4 or len(set(tokens))!=4: raise RuntimeError('reference worker requires four distinct Slurm-assigned GPUs')
    if output is None:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True)
    available=[{'index':v[0].strip(),'uuid':v[1].strip(),'name':v[2].strip(),'driver':v[3].strip(),'memory_mib':v[4].strip()}
               for v in csv.reader(io.StringIO(output)) if v]
    selected=[]
    for token in tokens:
        matches=[g for g in available if token in (g['index'],g['uuid'])]
        if len(matches)!=1: raise RuntimeError('GPU allocation cannot be mapped to physical UUID')
        selected.append(matches[0])
    roles=config['resources']
    if (roles.get('model_gpu_indices')!=[1,2] or roles.get('simulator_gpu_index')!=0
            or roles.get('perception_gpu_index')!=3):
        raise ValueError('reference topology is model1/2, simulator0, perception3')
    renderer=selected[0]['name'].upper()
    if any(s in renderer for s in ('A100','H100','H200')):
        raise RuntimeError('Isaac RTX renderer requires a supported RTX GPU; A100/H100/H200-only nodes are unsupported')
    return selected


def container_command(config,campaign_dir,source,group,attempt,relay,gpu):
    """Explicit immutable mounts; machine locations never enter task selection."""
    paths=config['paths']; container=config['container']; campaign_dir=Path(campaign_dir)
    if container.get('accept_eula') is not True: raise ValueError('set container.accept_eula only after accepting NVIDIA Isaac Sim license')
    runtime=container.get('runtime','apptainer')
    cache=attempt/'sim_state';cache.mkdir(parents=True,exist_ok=True)
    for child in ['home','cache/kit','cache/main','cache/cuda','logs','config','data','pkg']:(cache/child).mkdir(parents=True,exist_ok=True)
    (campaign_dir/'native').mkdir(exist_ok=True)
    binds=[(source,'/opt/robodata','ro'),(campaign_dir,'/campaign','rw'),
        (paths['benchmark_root'],'/work/RoboDojo','ro'),(campaign_dir/'native','/work/RoboDojo/eval_result','rw'),
        (paths['assets_root'],'/assets','ro'),(paths['sim_venv'],'/opt/robodata-env/sim','ro'),(relay,'/aspire-relay','rw'),
        (cache/'cache/kit','/isaac-sim/kit/cache','rw'),(cache/'cache/main','/isaac-sim/.cache','rw'),
        (cache/'cache/cuda','/isaac-sim/.nv/ComputeCache','rw'),(cache/'logs','/isaac-sim/.nvidia-omniverse/logs','rw'),
        (cache/'config','/isaac-sim/.nvidia-omniverse/config','rw'),(cache/'data','/isaac-sim/.local/share/ov/data','rw'),
        (cache/'pkg','/isaac-sim/.local/share/ov/pkg','rw')]
    command=[runtime,'exec','--nv','--cleanenv','--pwd','/work/RoboDojo','--home',f'{cache / "home"}:/home/robot']
    for host,guest,mode in binds: command+=['--bind',f'{Path(host).resolve()}:{guest}:{mode}']
    for item in container.get('extra_binds',[]): command+=['--bind',str(item)]
    relative=attempt.relative_to(campaign_dir)
    environment={'PYTHONPATH':'/opt/robodata/src:/opt/robodata/third_party/aspire/src:/work/RoboDojo:/work/RoboDojo/XPolicyLab',
        'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','ROBODATA_SIM_VENV':'/opt/robodata-env/sim',
        'CUDA_VISIBLE_DEVICES':gpu['uuid'],'ASPIRE_CUDA_DEVICE_INDEX':'0','ASPIRE_VULKAN_INDEX':gpu['index'],
        'ASPIRE_SIM_GPU_UUID':gpu['uuid'],'ACCEPT_EULA':'Y','ROBODOJO_ROOT':'/work/RoboDojo',
        'ROBODATA_GROUP_ID':group['group_id'],'ASPIRE_FULL_MODE':'evaluation',
        'EVAL_NUM':str(group['target_episodes']),'EVAL_LAYOUT_IDS':','.join(map(str,group['candidate_layout_ids'])),
        'EVAL_SEED':str(group['eval_seed']),'ROBODOJO_MAX_INPROC_RESTARTS':'3','ROBODOJO_NUM_ENVS':'1',
        'ROBODOJO_UNTILED_CAMERAS':'0','ROBODOJO_PATH_TRACING':'1','ROBODOJO_ENABLE_METRIC_DEPTH':'1','ROBODOJO_ACTION_TYPE':'joint',
        'ENV_CFG':'arx_x5','ROBODOJO_ENV_CFG':'arx_x5','ROBODOJO_RUN_ID':f"{campaign_dir.name}_{group['group_id']}",
        'ASPIRE_MODEL_NAME':MODEL['repo'],'ASPIRE_RELAY_DIR':'/aspire-relay',
        'ASPIRE_OUTPUT_DIR':f"/campaign/groups/{group['group_id']}",
        'ASPIRE_SIM_LOG':f'/campaign/{relative}/simulator.log','ASPIRE_PERCEPTION_CONFIG':f'/campaign/{relative}/providers.json'}
    command += [str(paths['isaac_container']),'/usr/bin/env']+[f'{k}={v}' for k,v in environment.items()]
    command += ['/bin/bash','/opt/robodata/deploy/sim_python.sh','-m','robodata.eval.robodojo.entry',
        '--task_name',group['task'],'--env_cfg_type','arx_x5','--num_envs','1','--enable_cameras','--device_id','0',
        '--policy_name','ASPIRE_Pilot','--port','0','--protocol','ws','--additional_info','aspire_qwen_full',
        '--seed',str(group['eval_seed']),'--host','127.0.0.1','--headless','--experience','/isaac-sim/apps/isaacsim.exp.base.python.kit',
        '--kit_args=--/isaaclab/cameras_enabled=true --/crashreporter/enabled=false --/renderer/multiGpu/enabled=false --/renderer/activeGpu='+gpu['index']]
    return command


class Worker:
    def __init__(self,campaign,group,config,source,attempt):
        self.campaign,self.group,self.config,self.source,self.attempt=campaign,group,config,source,attempt
        self.children=[];self.stop=False;self.report={'started_utc':utc_now(),'group_id':group['group_id'],'complete':False,'model':MODEL,'inference':INFERENCE,'offline_skill_discovery':False,'service_rng_policy':'seed0 at each group/attempt service startup, advances per request; restart/group scheduling can change perception samples'}
        self.env={k:v for k,v in os.environ.items() if not (any(s in k.upper() for s in ('TOKEN','API_KEY','SECRET')) or k.upper() in {'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','CUDA_HOME','CUDA_PATH','TRITON_PTXAS_PATH'})}
        self.env.update(PYTHONPATH=f'{source}/src:{source}/third_party/aspire/src',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1')
    def save(self): write_json(self.attempt/'worker.json',self.report)
    def record(self,event,**values):
        row={'utc':utc_now(),'event':event,**values}
        with (self.attempt/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def spawn(self,name,command,env=None):
        log=(self.attempt/(name+'.log')).open('w')
        proc=subprocess.Popen(command,env=env or self.env,cwd=self.source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();self.children.append((name,proc));self.record('process_started',name=name,pid=proc.pid)
        return proc
    def check(self, ignore=None):
        if self.stop: raise InterruptedError('worker stopping, preserve unscored attempt')
        for name,p in self.children:
            if p is not ignore and p.poll() is not None: raise RuntimeError(f'{name} exited {p.returncode}')
    def ready(self,url,process,seconds=2400):
        opener=request.build_opener(request.ProxyHandler({}));until=time.monotonic()+seconds
        while True:
            self.check()
            try:
                with opener.open(url,timeout=5) as response:return json.load(response)
            except (OSError,ValueError):
                if time.monotonic()>until: raise RuntimeError('service readiness timeout')
                time.sleep(1)
    def stop_all(self):
        for name,p in reversed(self.children):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        self.children=[]
    def run(self):
        from deploy.manage import verify_sources as verify_native_installation
        self.report['external_sources']=verify_native_installation(self.config)
        paths=self.config['paths'];gpus=resolve_gpus(self.config,os.environ)
        self.report['gpus']=gpus;self.save()
        pinned=json.loads(Path(paths['model_manifest']).read_text())
        if {k:pinned.get(k) for k in MODEL}!=MODEL or Path(pinned['snapshot_path']).resolve()!=Path(paths['model_snapshot']).resolve():raise ValueError('model snapshot/revision differs')
        verify_artifacts(pinned.get('files'),root=paths['model_snapshot'])
        image=Path(paths['isaac_container']); image_record=json.loads(image.with_suffix('.manifest.json').read_text())
        if image_record.get('sha256')!=sha_file(image) or image_record.get('bytes')!=image.stat().st_size:raise ValueError('Isaac container content drift')
        self.report['isaac_container']=image_record
        modelconfig=json.loads((Path(paths['model_snapshot'])/'config.json').read_text())
        if modelconfig['text_config']['max_position_embeddings']!=262144:raise ValueError('native model context differs')
        model_environment=json.loads((Path(paths['model_python']).parent.parent/'installation.json').read_text())
        actual_packages=json.loads(subprocess.check_output([paths['model_python'],'-c','import importlib.metadata as m,json;print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))'],text=True,env=self.env))
        if (model_environment.get('component')!='model' or model_environment.get('packages')!=actual_packages
                or model_environment.get('lock_sha256')!=sha_file(self.source/'deploy/locks/qwen.txt')):raise ValueError('model environment drift')
        self.report['model_environment']=model_environment
        self.report['simulator_environment']=json.loads((Path(paths['sim_venv'])/'installation.json').read_text())
        self.report['model_manifest']=pinned
        port=unused_port();url=f'http://127.0.0.1:{port}/v1'
        command=server_command(paths['model_python'],paths['model_snapshot'],port)
        self.report['server_command']=command
        server=self.spawn('model',command,dict(self.env,CUDA_VISIBLE_DEVICES=','.join(gpus[i]['uuid'] for i in [1,2])))
        self.ready(url+'/models',server)
        log=(self.attempt/'model.log').read_text(errors='replace')
        engines=[x for x in log.splitlines() if 'Initializing a V1 LLM engine' in x]
        if not engines or 'dtype=torch.bfloat16' not in engines[-1] or 'quantization=None' not in engines[-1] or not re.search(r"method\s*=\s*['\"]mtp['\"]",engines[-1]) or not re.search(r'num_spec(?:ulative)?_tokens\s*=\s*3\b',engines[-1]):raise RuntimeError('missing actual BF16 MTP3 engine evidence')
        if not all(x in engines[-1] for x in ('tensor_parallel_size=2','max_seq_len=262144')):raise RuntimeError('actual TP2/native context not confirmed')
        self.report['engine_config']=engines[-1]
        smoke=self.spawn('model_smoke',[paths['model_python'],'-B','-m','robodata.agents.full_model_smoke',
            '--base-url',url,'--output',str(self.attempt/'model_smoke/report.json')])
        while smoke.poll() is None:self.check(ignore=smoke);time.sleep(.2)
        self.children=[(n,p) for n,p in self.children if p is not smoke]
        if smoke.returncode or json.loads((self.attempt/'model_smoke/report.json').read_text()).get('ok') is not True:
            raise RuntimeError('unscored model interface check failed')
        runtime=json.loads(Path(paths['perception_runtime_manifest']).read_text())
        for item in runtime['files']:
            if sha_file(item['path'])!=item['sha256'] or Path(item['path']).stat().st_size!=item['bytes']:raise ValueError('perception artifact changed')
        urls={};health={}
        for kind,key in [('sam3','sam3_python'),('graspnet','graspnet_python')]:
            actual=json.loads(subprocess.check_output([paths[key],'-c','import importlib.metadata as m,json;print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))'],text=True,env=self.env))
            if actual!=runtime['environments'][kind]:raise ValueError('perception environment drift')
            p=unused_port();base=f'http://127.0.0.1:{p}'
            proc=self.spawn(kind,[paths[key],'-B','-m','robodata.perception.perception_service','--kind',kind,'--host','127.0.0.1','--port',str(p),
                '--sam3-manifest',paths['sam3_manifest'],'--vendor-root',paths['perception_vendor_root']],dict(self.env,CUDA_VISIBLE_DEVICES=gpus[3]['uuid']))
            health[kind]=self.ready(base+'/healthz',proc,1200)
            if not all(health[kind].get(k) is True for k in ('ok','loaded','model_loaded')):raise RuntimeError('perception service not loaded')
            if kind=='graspnet':
                from robodata.perception.cgn_singleton_fix import expected_patch_evidence
                patch=health[kind].get('estimator_patch',{})
                if patch!=expected_patch_evidence(sha_file(self.source/'src/robodata/perception/cgn_singleton_fix.py')):raise RuntimeError('CGN repair not active')
            urls[kind+'_url']=base
        providers={**urls,'grasp_calibration':CALIBRATION,'provenance':{'runtime_manifest':runtime,'runtime_manifest_sha256':sha_file(paths['perception_runtime_manifest']),
            'service_health':health,'gpu_uuid':gpus[3]['uuid'],'service_runner_sha256':sha_file(self.source/'src/robodata/perception/perception_service.py'),'offline_skill_discovery':False}}
        write_json(self.attempt/'providers.json',providers);self.report['providers']=providers['provenance'];self.save()
        relay=Path(tempfile.mkdtemp(prefix='robodata-',dir=os.environ.get('SLURM_TMPDIR') or '/tmp'));relay.chmod(0o700)
        write_json(self.attempt/'relay_location.json',{'node_local':str(relay),'host':socket.gethostname()})
        try:
            cfg={'usage_jsonl':str(self.attempt/'usage.jsonl'),'episode_ids':self.group['episode_ids'],
                'episode_seeds':{eid:self.group['eval_seed'] for eid in self.group['episode_ids']},'base_url':url,'expected_context_length':262144,'socket_idle_timeout_s':None}
            write_json(self.attempt/'controller_config.json',cfg)
            self.spawn('controller',[paths['model_python'],'-B','-m','robodata.agents.full_model_driver','--config',str(self.attempt/'controller_config.json'),'--relay-dir',str(relay)])
            until=time.monotonic()+60
            while not (relay/'ready.json').exists():
                self.check()
                if time.monotonic()>until:raise RuntimeError('controller readiness timeout')
                time.sleep(.2)
            command=container_command(self.config,self.campaign,self.source,self.group,self.attempt,relay,gpus[0])
            self.report['simulator_command']=command;self.save()
            sim=self.spawn('simulator',command,dict(self.env,CUDA_VISIBLE_DEVICES=gpus[0]['uuid']))
            while sim.poll() is None:self.check(ignore=sim);time.sleep(1)
            self.children=[(n,p) for n,p in self.children if p is not sim]
            if sim.returncode:raise RuntimeError(f'simulator exited {sim.returncode}, preserve native checkpoint')
            if resume_status(self.campaign,self.group)!='complete':raise RuntimeError('group ended without complete native evidence')
            self.report['complete']=True
        finally:
            self.stop_all()
            shutil.copytree(relay,self.attempt/'relay');shutil.rmtree(relay)


def verify_campaign(campaign_dir):
    """Validate the published plan before any execution or aggregation."""
    campaign=Path(campaign_dir).resolve(strict=True)
    plan=json.loads((campaign/'plan.json').read_text())
    manifest=load_manifest(campaign/'manifest.json')
    config=json.loads((campaign/'cluster_config.json').read_text())
    inventory=json.loads((campaign/'source_inventory.json').read_text())
    if (plan.get('schema')!=1 or plan.get('manifest_sha256')!=canonical_sha256(manifest)
            or plan.get('cluster_config_sha256')!=canonical_sha256(config)
            or plan.get('source_tree_sha256')!=inventory.get('tree_sha256')):
        raise ValueError('published plan/config/manifest/source digest mismatch')
    ids=[g['group_id'] for g in manifest['groups']]
    assignments=plan.get('assignments',[])
    if (plan.get('groups')!=ids or not assignments
            or [a.get('shard_index') for a in assignments]!=list(range(len(assignments)))
            or any(not isinstance(a.get('group_ids'),list) or not a['group_ids'] for a in assignments)):
        raise ValueError('published shard topology is invalid')
    owned=[gid for a in assignments for gid in a['group_ids']]
    if len(owned)!=len(ids) or len(set(owned))!=len(owned) or set(owned)!=set(ids):
        raise ValueError('native group ownership must cover every group exactly once')
    verify_source(campaign/'source',inventory)
    return plan,manifest,config,inventory


def validate_agent(config,source):
    path=Path(config.get('agent_config','configs/agents/qwen38_27b.json'))
    if path.is_absolute(): raise ValueError('agent_config must select the frozen repository-relative profile')
    if '..' in path.parts: raise ValueError('agent profile escapes frozen source')
    profile=json.loads((source/path).read_text())
    expected={'model':MODEL,'inference':INFERENCE,
        'sampling':{'temperature':1.0,'top_p':.95,'top_k':20,'min_p':0.0,'presence_penalty':0.0,'repetition_penalty':1.0},
        'thinking':{'enable_thinking':True,'preserve_thinking':False,'reasoning_effort':'medium'},
        'limits':{'model_turn_cap':None,'output_token_cap':None,'episode_walltime_cap':None,'history_compaction':False},
        'perception':{'sam3':True,'contact_graspnet':True,'offline_skill_discovery':False}}
    if any(profile.get(k)!=v for k,v in expected.items()):raise ValueError('agent profile differs from reviewed unchanged Qwen protocol')
    return profile


def verify_artifacts(items, *, root=None):
    if not isinstance(items,list) or not items:raise ValueError('artifact inventory is empty')
    paths=set()
    for item in items:
        path=Path(item['path']).resolve(strict=True)
        if path in paths or (root is not None and not path.is_relative_to(Path(root).resolve())):raise ValueError('duplicate/outside artifact inventory')
        paths.add(path)
        if path.stat().st_size!=item['bytes'] or sha_file(path)!=item['sha256']:raise ValueError('artifact content drift: '+str(path))
    return len(paths)


def run_campaign_group(campaign_dir,group_id,resume=False,*,frozen=False):
    campaign=Path(campaign_dir).resolve(strict=True)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',campaign.name):raise ValueError('campaign name must use letters digits underscores hyphens')
    plan,manifest,config,inventory=verify_campaign(campaign)
    source=campaign/'source'
    validate_agent(config,source)
    if not frozen:
        env=dict(os.environ,PYTHONPATH=f'{source}/src:{source}/third_party/aspire/src',PYTHONDONTWRITEBYTECODE='1')
        command=[sys.executable,'-B','-m','robodata.eval.robodojo.runner','--campaign-dir',str(campaign),'--group-id',group_id,'--frozen']
        if resume:command+=['--resume']
        return subprocess.call(command,env=env,cwd=source)
    if Path(__file__).resolve()!=source/'src/robodata/eval/robodojo/runner.py':raise ValueError('runner not imported from frozen source')
    if not os.environ.get('SLURM_JOB_ID','').isdigit():raise RuntimeError('GPU/model/simulator execution requires Slurm allocation')
    group=get_group(manifest,group_id)
    verify_sources(manifest,config['paths']['benchmark_root'],config['paths']['assets_root'],group_id)
    locks=campaign/'locks';locks.mkdir(exist_ok=True)
    with (locks/(group_id+'.lock')).open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        directory=campaign/'groups'/group_id
        prior=list((campaign/'allocations').glob('*/'+group_id+'/attempt_*'))
        if not resume and (directory.exists() or prior or native_result_path(campaign,group).exists()):raise RuntimeError('existing group evidence requires inspected --resume')
        if resume_status(campaign,group)=='complete':return 0
        parent=campaign/'allocations'/os.environ['SLURM_JOB_ID']/group_id;parent.mkdir(parents=True,exist_ok=True)
        index=1
        while (parent/f'attempt_{index:03d}').exists():index+=1
        attempt=parent/f'attempt_{index:03d}';attempt.mkdir()
        worker=Worker(campaign,group,config,source,attempt)
        batch_sha=os.environ.get('ROBODATA_BATCH_SHA256')
        if batch_sha is not None and batch_sha!=inventory['files_sha256'].get('deploy/worker.sbatch'):raise ValueError('executing Slurm script differs from frozen source')
        worker.report['batch_script_sha256']=batch_sha
        worker.report.update(source_tree_sha256=inventory['tree_sha256'],manifest_sha256=sha_file(campaign/'manifest.json'),cluster_config_sha256=sha_file(campaign/'cluster_config.json'),job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname())
        def stop(*_):worker.stop=True
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,stop)
        code=1
        try:worker.run();code=0
        except BaseException as exc:
            worker.report['error']={'type':type(exc).__name__,'message':str(exc)};worker.record('worker_interrupted',**worker.report['error'])
        finally:worker.stop_all();worker.report['finished_utc']=utc_now();worker.save()
        return code


def run_group(config_path,manifest_path,output_root,campaign,group_id,resume=False):
    directory=Path(output_root)/campaign
    if sha_file(config_path)!=sha_file(directory/'cluster_config.json') or sha_file(manifest_path)!=sha_file(directory/'manifest.json'):
        raise ValueError('arguments differ from planned campaign snapshots')
    return run_campaign_group(directory,group_id,resume)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign-dir',type=Path,required=True)
    selection=p.add_mutually_exclusive_group(required=True);selection.add_argument('--group-id');selection.add_argument('--shard-index',type=int)
    p.add_argument('--resume',action='store_true');p.add_argument('--frozen',action='store_true');args=p.parse_args(argv)
    groups=[args.group_id]
    if args.shard_index is not None:
        plan,_,_,_=verify_campaign(args.campaign_dir)
        assignments=[a for a in plan['assignments'] if a['shard_index']==args.shard_index]
        if len(assignments)!=1:raise ValueError('unknown shard index')
        groups=assignments[0]['group_ids']
    for gid in groups:
        code=run_campaign_group(args.campaign_dir,gid,args.resume,frozen=args.frozen)
        if code:return code
    return 0

if __name__=='__main__':raise SystemExit(main())
