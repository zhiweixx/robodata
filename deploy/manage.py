#!/usr/bin/env python3
"""Reproducible source/artifact/environment setup and Slurm submission.

This utility never launches evaluation directly on the machine running it.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re,shlex,shutil,subprocess,sys
from pathlib import Path
try:
    from .common import REPO,agent,artifact,load_config,require_slurm,run,sha256,write_json
except ImportError:
    from common import REPO,agent,artifact,load_config,require_slurm,run,sha256,write_json


def source_destinations(config):
    p=config['paths'];b=Path(p['benchmark_root']);v=Path(p['perception_vendor_root'])
    return {'robodojo':b,'xpolicylab':b/'XPolicyLab','isaaclab':b/'third_party/IsaacLab',
            'curobo':b/'third_party/curobo','sam3':v/'sam3','graspnet':v/'contact_graspnet_pytorch'}


def fetch_source(name,spec,destination):
    d=Path(destination)
    if d.exists() and any(d.iterdir()):
        head=subprocess.check_output(['git','-C',str(d),'rev-parse','HEAD'],text=True).strip()
        if head!=spec['revision']:raise RuntimeError(f'{name}: existing checkout has a different revision; refusing reset')
    else:
        d.mkdir(parents=True,exist_ok=True)
        run(['git','init',str(d)])
        run(['git','-C',str(d),'remote','add','origin',spec['url']])
        run(['git','-C',str(d),'fetch','--depth','1','origin',spec['revision']])
        run(['git','-C',str(d),'checkout','--detach','FETCH_HEAD'])
    if spec.get('patch'):
        patch=REPO/spec['patch']
        if sha256(patch)!=spec['patch_sha256']:raise RuntimeError('redistributed patch checksum mismatch')
        applied=subprocess.run(['git','-C',str(d),'apply','--reverse','--check',str(patch)],capture_output=True).returncode==0
        if not applied:
            run(['git','-C',str(d),'apply','--check',str(patch)])
            run(['git','-C',str(d),'apply',str(patch)])
    return {'path':str(d.resolve()),'revision':spec['revision'],'patch_sha256':spec.get('patch_sha256')}


def source_files(root,include_utils=False):
    rows=subprocess.check_output(['git','-C',str(root),'ls-files','--stage','-z']).split(b'\0')
    result=[]
    for row in rows:
        if not row:continue
        metadata,name=row.split(b'\t',1);mode=metadata.split()[0]
        if mode==b'160000':continue  # Explicitly pinned nested projects have their own records.
        relative=os.fsdecode(name);path=root/relative
        if mode==b'120000':
            payload=os.readlink(path).encode();result.append({'relative_path':relative,'kind':'symlink','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
        else:result.append({'relative_path':relative,'kind':'file','bytes':path.stat().st_size,'sha256':sha256(path)})
    if include_utils and not any(x['relative_path']=='utils/__init__.py' for x in result):
        path=root/'utils/__init__.py';result.append({'relative_path':'utils/__init__.py','kind':'file','bytes':path.stat().st_size,'sha256':sha256(path)})
    return sorted(result,key=lambda row:row['relative_path'])


def verify_sources(config):
    record_path=Path(config['paths']['work_root']).parent/'source-installation.json'
    records=json.loads(record_path.read_text());destinations=source_destinations(config)
    pins=json.loads((REPO/'deploy/sources.lock.json').read_text())
    if set(records)!=set(pins['sources']):raise ValueError('external source inventory is incomplete')
    for name,record in records.items():
        root=destinations[name];spec=pins['sources'][name]
        if Path(record['path']).resolve()!=root.resolve() or record['revision']!=spec['revision'] or record.get('patch_sha256')!=spec.get('patch_sha256'):raise ValueError('external source identity differs: '+name)
        if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=spec['revision'] or source_files(root,name=='robodojo')!=record['files']:raise ValueError('external source content drift: '+name)
    return {'path':str(record_path),'sha256':sha256(record_path),'sources':records}


def fetch(config,config_path,components,token_file):
    pins=json.loads((REPO/'deploy/sources.lock.json').read_text());p=config['paths']
    if 'sources' in components:
        destinations=source_destinations(config)
        records={n:fetch_source(n,s,destinations[n]) for n,s in pins['sources'].items()}
        package=Path(p['benchmark_root'])/'utils/__init__.py';package.parent.mkdir(parents=True,exist_ok=True)
        content=(REPO/'deploy/robodojo_utils_init.py').read_bytes()
        if package.exists() and package.read_bytes()!=content:raise RuntimeError('unexpected benchmark utils initializer')
        package.write_bytes(content)
        link=Path(p['benchmark_root'])/'Assets'
        if not link.exists() and not link.is_symlink():link.symlink_to('/assets',target_is_directory=True)
        elif not link.is_symlink() or str(link.readlink())!='/assets':raise RuntimeError('benchmark Assets must resolve to container /assets; refusing replacement')
        for name,record in records.items():record['files']=source_files(Path(record['path']), include_utils=name=='robodojo')
        write_json(Path(p['work_root']).parent/'source-installation.json',records)
    hf=[c for c in components if c!='sources']
    if hf:
        command=[sys.executable,str(REPO/'deploy/fetch_hf.py'),'--config',str(Path(config_path).resolve()),'--components',*hf]
        if token_file:command+=['--hf-token-file',str(Path(token_file).resolve())]
        # Only the file path is an argument; no token appears in process listings/logs.
        run(command)


def doctor(config,gpu=False):
    checks=[]
    for command in ('git','uv',config['container']['runtime'],'sbatch','squeue'):
        checks.append({'check':'command:'+command,'ok':shutil.which(command) is not None})
    for key in ('benchmark_root','assets_root','isaac_container','sim_venv','model_snapshot','model_python','sam3_python','graspnet_python','perception_runtime_manifest','sam3_manifest','model_manifest'):
        checks.append({'check':'installed:'+key,'ok':Path(config['paths'][key]).exists()})
    checks.append({'check':'explicit_isaac_license_acceptance','ok':config['container'].get('accept_eula') is True})
    scheduler=config.get('scheduler',{})
    checks.append({'check':'scheduler_configured','ok':all(scheduler.get(k) and 'REPLACE' not in str(scheduler[k]) for k in ('account','partition','gres'))})
    if gpu:
        require_slurm()
        sys.path.insert(0,str(REPO/'src'))
        from robodata.eval.robodojo.runner import resolve_gpus
        try:
            selected=resolve_gpus(config,os.environ)
            checks.append({'check':'allocated_renderer_gpu','ok':True,'evidence':selected,'note':'GPU allocation check only; actual RGB-D smoke remains mandatory.'})
        except (ValueError,RuntimeError) as exc:
            checks.append({'check':'allocated_renderer_gpu','ok':False,'error':str(exc)})
    return {'checks':checks,'ready':all(c['ok'] for c in checks),'gpu_smoke_validated':False}


def build_container(config):
    if config['container'].get('accept_eula') is not True:raise ValueError('Read NVIDIA Isaac Sim license and explicitly set container.accept_eula=true before downloading/building')
    image=Path(config['paths']['isaac_container']);image.parent.mkdir(parents=True,exist_ok=True)
    source=config['container']['source']
    if not source.startswith('docker://nvcr.io/nvidia/isaac-sim'):raise ValueError('expected NVIDIA Isaac Sim OCI source')
    if '@sha256:' not in source:
        if not shutil.which('skopeo'):raise RuntimeError('Install skopeo to resolve the OCI tag digest, or configure an immutable docker://...@sha256:... source')
        digest=subprocess.check_output(['skopeo','inspect','--format','{{.Digest}}',source],text=True).strip()
        if not re.fullmatch(r'sha256:[a-f0-9]{64}',digest):raise ValueError('invalid OCI digest')
        source=source.rsplit(':',1)[0]+'@'+digest
    record=image.with_suffix('.manifest.json')
    if image.exists():
        if not record.exists() or json.loads(record.read_text()).get('sha256')!=sha256(image):raise RuntimeError('existing image lacks matching checksum record; refusing overwrite')
        existing=json.loads(record.read_text())
        if existing.get('source')!=source:raise RuntimeError('existing image source digest differs; use a new installation path')
        return existing
    run([config['container']['runtime'],'build',str(image),source])
    result=dict(artifact(image),source=source,isaac_sim_version='6.0.0',license_accepted=True)
    write_json(record,result);return result


def setup_host_environment(config,kind):
    p=config['paths'];python=Path(p[{'model':'model_python','sam3':'sam3_python','graspnet':'graspnet_python'}[kind]])
    env=python.parent.parent;env.parent.mkdir(parents=True,exist_ok=True)
    for variable in ('CUDA_HOME','CUDA_PATH','TRITON_PTXAS_PATH','CUROBO_USE_PYBIND'):os.environ.pop(variable,None)
    run(['uv','venv','--allow-existing','--python',config.get('setup',{}).get('python','3.12.13'),str(env)])
    lock=REPO/'deploy/locks'/({'model':'qwen','sam3':'sam3','graspnet':'graspnet'}[kind]+'.txt')
    command=['uv','pip','install','--python',str(python),'--torch-backend','cu130' if kind=='model' else 'cu128','-r',str(lock)]
    # Providers are full observed environment snapshots. Reconstruct every pinned
    # distribution without asking a resolver to rewrite historical combinations.
    if kind!='model':command.append('--no-deps')
    run(command)
    if kind!='model':
        vendor=Path(p['perception_vendor_root'])/('sam3' if kind=='sam3' else 'contact_graspnet_pytorch')
        if kind=='sam3':run(['uv','pip','install','--python',str(python),'--no-deps','--no-build-isolation','-e',str(vendor)])
        # CGN is imported by the service from its pinned source tree, not a new wheel.
        run(['uv','pip','install','--python',str(python),'--no-deps','-e',str(REPO/'third_party/aspire')])
    packages=json.loads(subprocess.check_output([str(python),'-c','import importlib.metadata as m,json;print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))'],text=True))
    write_json(env/'installation.json',{'component':kind,'lock_sha256':sha256(lock),'packages':packages})
    return packages


def provider_manifest(config):
    p=config['paths'];vendors=Path(p['perception_vendor_root'])
    paths=[]
    for vendor in ('sam3','contact_graspnet_pytorch'):
        paths.extend(sorted((vendors/vendor).rglob('*.py')))
    paths.extend([vendors/'contact_graspnet_pytorch/checkpoints/contact_graspnet/config.yaml',vendors/'contact_graspnet_pytorch/checkpoints/contact_graspnet/checkpoints/model.pt'])
    sam=json.loads(Path(p['sam3_manifest']).read_text());paths.extend(Path(f['path']) for f in sam['files'])
    environments={}
    for kind in ('sam3','graspnet'):
        python=p[kind+'_python']
        environments[kind]=json.loads(subprocess.check_output([python,'-c','import importlib.metadata as m,json;print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))'],text=True))
    write_json(p['perception_runtime_manifest'],{'schema_version':1,'files':[dict(artifact(x),file=str(x.relative_to(vendors)) if x.is_relative_to(vendors) else x.name) for x in paths], 'environments':environments,'offline_skill_discovery':False,'source_lock_sha256':sha256(REPO/'deploy/sources.lock.json')})


def setup_simulator(config):
    require_slurm()
    if config['container'].get('accept_eula') is not True:raise ValueError('explicit NVIDIA license acceptance required')
    p=config['paths'];venv=Path(p['sim_venv']);venv.mkdir(parents=True,exist_ok=True)
    state=Path(os.environ.get('SLURM_TMPDIR','/tmp'))/('robodata-setup-'+os.environ['SLURM_JOB_ID']);state.mkdir(parents=True,exist_ok=True)
    command=[config['container']['runtime'],'exec','--nv','--cleanenv','--bind',str(REPO)+':/opt/robodata:ro','--bind',p['benchmark_root']+':/work/RoboDojo','--bind',str(venv)+':/opt/robodata-env/sim','--bind',str(state)+':/tmp/robodata-setup','--env','ACCEPT_EULA=Y','--env','ROBODATA_SIM_VENV=/opt/robodata-env/sim','--env','TORCH_CUDA_ARCH_LIST='+config.get('setup',{}).get('cuda_arch_list','8.9'),p['isaac_container'],'/bin/bash','/opt/robodata/deploy/setup_sim_inside.sh']
    run(command)


def verified_campaign(campaign_dir):
    sys.path.insert(0,str(REPO/'src'))
    from robodata.data.provenance import verify_source,canonical_sha256
    campaign=Path(campaign_dir).resolve(strict=True)
    if not re.fullmatch(r'[A-Za-z0-9_]+',campaign.name):raise ValueError('campaign name must use letters, digits, underscores')
    source=campaign/'source';inventory=json.loads((campaign/'source_inventory.json').read_text())
    verify_source(source,inventory)
    plan=json.loads((campaign/'plan.json').read_text());config=json.loads((campaign/'cluster_config.json').read_text());manifest=json.loads((campaign/'manifest.json').read_text())
    if (plan['source_tree_sha256']!=inventory['tree_sha256'] or plan['cluster_config_sha256']!=canonical_sha256(config) or plan['manifest_sha256']!=canonical_sha256(manifest)):raise ValueError('frozen plan/config/manifest digest mismatch')
    assignments=plan['assignments'];expected=[g['group_id'] for g in manifest['groups']]
    actual=[g for row in assignments for g in row['group_ids']]
    if ([r['shard_index'] for r in assignments]!=list(range(len(assignments))) or len(actual)!=len(set(actual)) or sorted(actual)!=sorted(expected) or plan['groups']!=expected):raise ValueError('plan assignments must own each native group exactly once')
    return campaign,source,plan,config,inventory


def submit(campaign_dir,resume=False,dry_run=False):
    campaign,source,plan,config,inventory=verified_campaign(campaign_dir)
    scheduler=config['scheduler']
    if any('REPLACE' in str(scheduler.get(k,'')) or not scheduler.get(k) for k in ('account','partition','gres')):raise ValueError('configure scheduler account/partition/gres before planning')
    if type(scheduler['max_concurrent_shards']) is not int or scheduler['max_concurrent_shards']<1:raise ValueError('positive array concurrency required')
    logs=campaign/'scheduler'
    args=['sbatch','--parsable','--job-name=robodata-'+campaign.name,'--nodes=1','--ntasks=1','--cpus-per-task='+str(scheduler['cpus_per_task']),'--mem='+scheduler['memory'],'--gres='+scheduler['gres'],'--time='+scheduler['time'],'--account='+scheduler['account'],'--partition='+scheduler['partition'],'--array=0-'+str(len(plan['assignments'])-1)+'%'+str(scheduler['max_concurrent_shards']),'--output='+str(logs/'%A_%a.log'),'--chdir='+str(source),str(source/'deploy/worker.sbatch'),str(source),str(campaign)]
    if resume:args+=['resume']
    if dry_run:return {'command':args,'scope':'one array element per disjoint native group assignment; no job submitted'}
    logs.mkdir(parents=True,exist_ok=True)
    active=subprocess.check_output(['squeue','-u',os.environ.get('USER',''),'-h','-o','%j'],text=True).splitlines()
    if 'robodata-'+campaign.name in active:raise RuntimeError('same campaign already has active jobs; inspect before resubmission')
    submission_env=dict(os.environ,ROBODATA_PYTHON=sys.executable)
    job=subprocess.check_output(args,text=True,env=submission_env).strip();record={'job_id':job,'command':args,'resume':resume,'plan_sha256':sha256(campaign/'plan.json'),'source_tree_sha256':inventory['tree_sha256']}
    write_json(logs/('submission-'+job.split(';')[0]+'.json'),record);return record


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    for name in ('doctor','fetch','setup','build-container','submit'):
        parser=sub.add_parser(name)
        if name!='submit':parser.add_argument('--config',required=True)
        if name=='doctor':parser.add_argument('--gpu',action='store_true')
        if name=='fetch':parser.add_argument('--components',nargs='+',choices=['sources','assets','model','providers'],default=['sources','assets','model','providers']);parser.add_argument('--hf-token-file')
        if name=='setup':parser.add_argument('--components',nargs='+',choices=['model','sam3','graspnet','simulator'],default=['model','sam3','graspnet','simulator'])
        if name=='submit':
            parser.add_argument('--campaign-dir',required=True)
            parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=ap.parse_args(argv)
    if args.command=='submit':
        print(json.dumps(submit(args.campaign_dir,args.resume,args.dry_run),indent=2));return 0
    config=load_config(args.config)
    if args.command=='doctor':
        result=doctor(config,args.gpu);print(json.dumps(result,indent=2));return 0 if result['ready'] else 2
    if args.command=='fetch':fetch(config,args.config,args.components,args.hf_token_file)
    elif args.command=='build-container':print(json.dumps(build_container(config),indent=2))
    elif args.command=='setup':
        require_slurm()
        for kind in args.components:
            if kind=='simulator':setup_simulator(config)
            else:setup_host_environment(config,kind)
        if all(Path(config['paths'][key]).exists() for key in ('sam3_python','graspnet_python','sam3_manifest')):provider_manifest(config)
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError) as exc:print('Deployment error: '+str(exc),file=sys.stderr);raise SystemExit(2)
