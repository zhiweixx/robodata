"""Portable installation helpers; never infer paths from the developer's machine."""
from __future__ import annotations
import hashlib,json,os,re,stat,subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = ('benchmark_root','assets_root','isaac_container','sim_venv','model_snapshot','model_manifest','model_python','sam3_python','graspnet_python','perception_runtime_manifest','sam3_manifest','perception_vendor_root','work_root')

def load_config(path):
    value=json.loads(Path(path).read_text())
    if value.get('schema_version') != 1: raise ValueError('unsupported cluster config version')
    def expand(v):
        if isinstance(v,str):
            v=os.path.expanduser(os.path.expandvars(v))
            if re.search(r'\$\{?\w+',v): raise ValueError('unresolved environment variable in cluster config')
            return v
        if isinstance(v,list):return [expand(x) for x in v]
        if isinstance(v,dict):return {k:expand(x) for k,x in v.items()}
        return v
    value=expand(value)
    for key in REQUIRED_PATHS:
        if key not in value['paths'] or not Path(value['paths'][key]).is_absolute():raise ValueError(f'paths.{key} must be absolute after expansion')
    roles=value['resources']; slots=list(roles['model_gpu_indices'])+[roles['simulator_gpu_index'],roles['perception_gpu_index']]
    if slots!=[1,2,0,3]:raise ValueError('reference topology requires model slots 1/2, simulator slot 0, perception slot 3')
    if value['container']['runtime'] not in ('apptainer','singularity'):raise ValueError('supported container runtimes: apptainer, singularity')
    return value

def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)

def run(args,**kwargs):
    print(json.dumps({'command':list(map(str,args))}),flush=True)
    return subprocess.run(list(map(str,args)),check=True,**kwargs)

def private_token(path):
    if path is None:return None
    p=Path(path);s=p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)&0o077:raise ValueError('HF token file must be a private user-owned regular file (chmod 600)')
    return p.read_text().strip()

def require_slurm():
    if not os.environ.get('SLURM_JOB_ID','').isdigit():raise RuntimeError('GPU setup/warmup must run inside a Slurm allocation, never on a laptop/login node')

def artifact(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha256(p)}

def agent(config):
    p=Path(config.get('agent_config','configs/agents/qwen38_27b.json'))
    if not p.is_absolute():p=REPO/p
    return json.loads(p.read_text())
