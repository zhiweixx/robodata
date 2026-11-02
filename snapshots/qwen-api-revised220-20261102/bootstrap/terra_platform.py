"""Pinned Isaac/NVIDIA infrastructure fixes reused from the deployed Astra baseline.

No model, robot action, observation or score semantics are changed. GPU identity
inspection reads metadata only; actual NGX validation runs inside the Slurm
simulator and collects only its verified failed-updater children.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def gpu_device_minor_mapping(proc_information):
    """Join global NVIDIA device minors to UUIDs using read-only driver metadata.

    Slurm's global GRES device IDs and nvidia-smi's cgroup-local indices are
    separate namespaces. Never infer a global ID from a visible row position.
    """
    if not isinstance(proc_information, dict) or not proc_information:
        raise RuntimeError('NVIDIA proc UUID/device-minor identity evidence is required')
    by_minor = {}; seen_uuids = set(); seen_buses = set()
    for path, raw in proc_information.items():
        if not isinstance(path, str) or not isinstance(raw, str):
            raise RuntimeError('invalid NVIDIA proc GPU identity record')
        fields = {}
        for line in raw.splitlines():
            if ':' not in line:continue
            key, value = line.split(':', 1)
            key, value = key.strip(), value.strip()
            if key in ('GPU UUID', 'Device Minor', 'Bus Location', 'Model'):
                if key in fields:raise RuntimeError('duplicate NVIDIA proc GPU identity field')
                fields[key] = value
        uuid_value, minor, bus = (fields.get(key, '') for key in ('GPU UUID', 'Device Minor', 'Bus Location'))
        if (not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', uuid_value)
                or not re.fullmatch(r'0|[1-9][0-9]*', minor)
                or not re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', bus)
                or Path(path).name != 'information' or Path(path).parent.name.lower() != bus.lower()
                or not fields.get('Model')):
            raise RuntimeError('incomplete or inconsistent NVIDIA proc GPU identity')
        if minor in by_minor or uuid_value in seen_uuids or bus.lower() in seen_buses:
            raise RuntimeError('ambiguous NVIDIA UUID/device-minor identity')
        by_minor[minor] = {'uuid': uuid_value, 'device_minor': int(minor),
            'pci_bus_id': bus.lower(), 'model': fields['Model'], 'proc_information_path': path}
        seen_uuids.add(uuid_value); seen_buses.add(bus.lower())
    return by_minor


def capture_gpu_identity_evidence(csv_text, environment, *, proc_root=Path('/proc/driver/nvidia/gpus')):
    """Capture only public GPU metadata before validation; no contexts or GPU work."""
    records = {}; errors = []
    try:
        paths = sorted(Path(proc_root).glob('*/information'))
        for path in paths:
            try:records[str(path)] = path.read_text()
            except OSError as exc:errors.append({'path':str(path),'type':type(exc).__name__})
    except OSError as exc:
        errors.append({'path':str(proc_root),'type':type(exc).__name__})
    return {'schema_version':1,'verified':False,'gpu_operations':'read-only identity metadata; no CUDA context or GPU work',
        'environment':{key:environment.get(key) for key in
            ('SLURM_JOB_ID','CUDA_VISIBLE_DEVICES','SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_GPUS_ON_NODE')},
        'gpu_csv':csv_text,'proc_nvidia_gpus':records,'metadata_read_errors':errors,
        'mapping_rule':'CUDA visible index or UUID -> nvidia-smi UUID; Slurm global device minor -> proc UUID; require exact UUID set'}


def resolve_four_gpus(csv_text, environment, *, proc_information=None):
    if not environment.get('SLURM_JOB_ID'):
        raise RuntimeError('this experiment requires a Slurm allocation')
    tokens = environment.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if (len(tokens) != 4 or len(set(tokens)) != 4
            or any(not re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+)', t) for t in tokens)):
        raise RuntimeError('expected exactly four allocated GPU tokens')
    rows = [{key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(csv_text.splitlines())]
    def resolve_visible(token):
        found = [r for r in rows if token in (r.get('index'), r.get('uuid'))]
        if len(found) != 1 or 'L40S' not in found[0].get('name', ''):
            raise RuntimeError('allocated visible GPU is not uniquely identified as an L40S')
        return found[0]
    devices = [resolve_visible(t) for t in tokens]
    if len({gpu['uuid'] for gpu in devices}) != 4:
        raise RuntimeError('allocated GPU tokens resolve to duplicate UUIDs')
    global_tokens = environment.get('SLURM_JOB_GPUS', '').split(',')
    if (len(global_tokens) != 4 or len(set(global_tokens)) != 4
            or any(not re.fullmatch(r'0|[1-9][0-9]*', token) for token in global_tokens)):
        raise RuntimeError('exactly four distinct Slurm global device IDs are required')
    by_minor = gpu_device_minor_mapping(proc_information)
    if any(token not in by_minor for token in global_tokens):
        raise RuntimeError('Slurm global device ID lacks NVIDIA proc identity evidence')
    slurm_devices = [by_minor[token] for token in global_tokens]
    if {d['uuid'] for d in slurm_devices} != {d['uuid'] for d in devices}:
        raise RuntimeError('visible GPU UUIDs disagree with Slurm global device allocation')
    by_uuid = {d['uuid']:d for d in slurm_devices}
    return [{**device, 'slurm_global_device_id': str(by_uuid[device['uuid']]['device_minor']),
        'device_minor':by_uuid[device['uuid']]['device_minor'],
        'pci_bus_id':by_uuid[device['uuid']]['pci_bus_id'],
        'identity_source':by_uuid[device['uuid']]['proc_information_path']} for device in devices]


NGX_LIBRARY_SHA256 = '0566e7071e0f3fa641a41ee3cf9555b2683cfc99bae346a96e6e8d9391626bda'


NGX_CALLER_RETURN_OFFSET = 0x81c72


AUXILIARY_PROCESS_POLICY = ('disable optional Hub/telemetry; synchronously reap only failed-exec '
    'children from pinned NGX fork callsite; no SIGCHLD or wildcard wait override')


def strict_json_object(text):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    def nonfinite(_):raise ValueError('nonfinite JSON value')
    value=json.loads(text,object_pairs_hook=pairs,parse_constant=nonfinite)
    if not isinstance(value,dict):raise ValueError('JSON object required')
    return value


def verify_ngx_library(path):
    """Pin the inspected driver and missing-updater condition before preload."""
    import shutil
    library=Path(path).resolve(strict=True)
    digest=sha_file(library)
    if digest!=NGX_LIBRARY_SHA256:
        raise RuntimeError('NGX driver differs from the inspected fork callsite')
    candidates=[str(Path(part or '.')/'nvidia-ngx-updater')
        for part in os.environ.get('PATH',os.defpath).split(os.pathsep)]
    if shutil.which('nvidia-ngx-updater') is not None or any(os.path.lexists(p) for p in candidates):
        raise RuntimeError('NGX updater is executable; failed-exec workaround is not applicable')
    st=library.stat()
    if os.getuid()!=os.geteuid():raise RuntimeError('NGX workaround requires identical real/effective UID')
    return {'path':str(library),'sha256':digest,'caller_return_offset':NGX_CALLER_RETURN_OFFSET,
        'device':st.st_dev,'inode':st.st_ino,'updater_executable_absent_from_path':True,
        'updater_path_candidates_all_absent':True,'updater_path_candidates':candidates,
        'uid':os.getuid(),'euid':os.geteuid(),'kernel':os.uname().release}


def validate_ngx_reaper_events(rows, *, expected_parent_pid):
    """Validate exact-child receipts without sending signals or waiting on PIDs."""
    if type(expected_parent_pid) is not int or expected_parent_pid<=0 or not isinstance(rows,list):
        raise RuntimeError('invalid NGX receipt parent identity')
    fields={'event','parent_pid','child_pid','errno','exit_status','exit_code_type','caller_offset'}
    registered=set();reaped=set();initializations=0;available_default_checks=0
    for row in rows:
        if (not isinstance(row,dict) or set(row)!=fields or not isinstance(row['event'],str)
                or any(type(row[k]) is not int for k in fields-{'event'})
                or row['parent_pid']<=0 or row['caller_offset']!=NGX_CALLER_RETURN_OFFSET):
            raise RuntimeError('NGX receipt schema/callsite differs')
        event=row['event'];child=row['child_pid']
        if event not in ('initialized','sigchld_default','registered','reaped'):
            raise RuntimeError('NGX reaper reported unknown/error event')
        if row['parent_pid']!=expected_parent_pid and event!='initialized':
            raise RuntimeError('NGX receipt contains actions by another parent')
        if event=='initialized':
            if any(row[k]!=0 for k in ('child_pid','errno','exit_status','exit_code_type')):
                raise RuntimeError('NGX initialization receipt is invalid')
            initializations+=int(row['parent_pid']==expected_parent_pid)
            continue
        if not initializations:raise RuntimeError('NGX action precedes parent initialization')
        if event=='sigchld_default':
            if any(row[k]!=0 for k in ('child_pid','errno','exit_status','exit_code_type')):
                raise RuntimeError('SIGCHLD observation receipt is invalid')
            available_default_checks+=1
        elif event=='registered':
            if (available_default_checks<=0 or child<=0 or child==expected_parent_pid or child in registered
                    or any(row[k]!=0 for k in ('errno','exit_status','exit_code_type'))):
                raise RuntimeError('NGX exact-child registration is invalid')
            available_default_checks-=1;registered.add(child)
        elif event=='reaped':
            if (child not in registered or child in reaped or row['errno']!=0
                    or row['exit_status']!=1 or row['exit_code_type']!=1):
                raise RuntimeError('NGX receipt did not reap its registered failed-exec child')
            reaped.add(child)
    if not initializations or not registered or registered!=reaped or available_default_checks:
        raise RuntimeError('NGX startup lacks complete exact-child reap receipts')
    return {'verified':True,'parent_pid':expected_parent_pid,
        'registered_child_pids':sorted(registered),'reaped_child_pids':sorted(reaped)}


def validate_ngx_library_transition(preflight,active):
    """Vendor Python may extend PATH; driver identity must remain identical."""
    for receipt in (preflight,active):
        if (not isinstance(receipt,dict) or receipt.get('updater_executable_absent_from_path') is not True
                or receipt.get('updater_path_candidates_all_absent') is not True
                or not isinstance(receipt.get('updater_path_candidates'),list)
                or not receipt['updater_path_candidates']
                or any(not isinstance(p,str) or Path(p).name!='nvidia-ngx-updater' for p in receipt['updater_path_candidates'])):
            raise RuntimeError('NGX updater absence evidence is invalid')
    stable=lambda value:{k:v for k,v in value.items() if k!='updater_path_candidates'}
    if canonical_sha256(stable(preflight))!=canonical_sha256(stable(active)):
        raise RuntimeError('NGX library identity changed since preflight')


def ngx_shim_is_loaded(path):
    """Check mapped device/inode rather than merely trusting LD_PRELOAD text."""
    st=Path(path).stat();device=(os.major(st.st_dev),os.minor(st.st_dev))
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts=line.split(None,5)
        if len(parts)<5:continue
        try:
            mapped=tuple(int(v,16) for v in parts[3].split(':'))
            if int(parts[4])==st.st_ino and mapped==device:return True
        except ValueError:continue
    return False


def noload_library_link_map(path):
    """Inspect an existing loader entry; never load a missing target library."""
    import ctypes
    class LinkMap(ctypes.Structure):
        _fields_=[('l_addr',ctypes.c_void_p),('l_name',ctypes.c_char_p),
            ('l_ld',ctypes.c_void_p),('l_next',ctypes.c_void_p),('l_prev',ctypes.c_void_p)]
    loader=ctypes.CDLL(None)
    loader.dlopen.argtypes=[ctypes.c_char_p,ctypes.c_int];loader.dlopen.restype=ctypes.c_void_p
    loader.dlinfo.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];loader.dlinfo.restype=ctypes.c_int
    loader.dlclose.argtypes=[ctypes.c_void_p];loader.dlclose.restype=ctypes.c_int
    handle=loader.dlopen(os.fsencode(path),os.RTLD_NOW|os.RTLD_NOLOAD)
    if not handle:return None
    try:
        pointer=ctypes.POINTER(LinkMap)()
        if loader.dlinfo(handle,2,ctypes.byref(pointer))!=0 or not pointer:
            raise RuntimeError('loaded library has no usable link-map identity')
        info=pointer.contents
        if not info.l_name or not info.l_addr:raise RuntimeError('loaded library identity is empty')
        return {'path':os.fsdecode(info.l_name),'base_address':int(info.l_addr)}
    finally:
        if loader.dlclose(handle)!=0:raise RuntimeError('failed to balance NOLOAD inspection handle')


def vendor_carb_is_loaded(path):
    """Handle the verified SIF stat/maps device mismatch for vendor libcarb only."""
    if str(path)!='/isaac-sim/kit/libcarb.so':return False
    try:
        linked=noload_library_link_map(path)
        if (not linked or not os.path.samefile(str(path),linked['path'])
                or type(linked.get('base_address')) is not int or linked['base_address']<=0):
            return False
        for line in Path('/proc/self/maps').read_text().splitlines():
            parts=line.split(None,5)
            if len(parts)<6 or Path(parts[5]).name!='libcarb.so':continue
            if not os.path.samefile(str(path),parts[5]):continue
            start,end=(int(v,16) for v in parts[0].split('-'))
            if start<=linked['base_address']<end:return True
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,AttributeError):
        return False
    return False


def verify_ngx_reaper_runtime(environment):
    code=Path(__file__).resolve().parent;shim=code/'ngx_child_reaper.so'
    vendor_carb=Path('/isaac-sim/kit/libcarb.so')
    output=Path(environment['ASPIRE_OUTPUT_DIR'])
    expected_log=output/'ngx_child_reaper.jsonl'
    if (environment.get('NGX_REAPER_LOG')!=str(expected_log)
            or environment.get('NGX_REAPER_LIBRARY')!='/opt/robodojo-driver-libs/libnvidia-ngx.so.1'
            or environment.get('NGX_REAPER_SHIM')!=str(shim)
            or environment.get('LD_PRELOAD')!=str(shim)+':'+str(vendor_carb) or shim.is_symlink()
            or not ngx_shim_is_loaded(shim) or not vendor_carb_is_loaded(vendor_carb)):
        raise RuntimeError('NGX reaper preload/mapping/private log contract differs')
    library=verify_ngx_library(environment['NGX_REAPER_LIBRARY'])
    if not ngx_shim_is_loaded(library['path']):
        raise RuntimeError('inspected NGX library is not actually mapped in simulator')
    preflight=output/'ngx_reaper_preflight.json'
    if preflight.is_symlink() or not preflight.is_file():raise RuntimeError('NGX preflight receipt missing')
    saved=strict_json_object(preflight.read_text())
    validate_ngx_library_transition(saved,library)
    if expected_log.is_symlink() or not expected_log.is_file():raise RuntimeError('NGX reaper log missing')
    raw=expected_log.read_bytes()
    if not raw or not raw.endswith(b'\n'):raise RuntimeError('NGX reaper log is empty/incomplete')
    rows=[strict_json_object(line) for line in raw.decode('utf-8').splitlines()]
    checked=validate_ngx_reaper_events(rows,expected_parent_pid=os.getpid())
    return {'schema_version':1,**checked,'library':library,'library_mapped_device_inode_verified':True,
        'shim':{'path':str(shim),'source_sha256':sha_file(code/'ngx_child_reaper.c'),
            'binary_sha256':sha_file(shim),'mapped_device_inode_verified':True,
            'vendor_carb_path':str(vendor_carb),'vendor_carb_mapped':True,
            'vendor_carb_mapping_method':'rtld_noload_dlinfo_samefile_and_mapped_base',
            'preload_paths':[str(shim),str(vendor_carb)]},
        'log_path':str(expected_log),'log_prefix_bytes':len(raw),
        'log_prefix_sha256':hashlib.sha256(raw).hexdigest(),'events':rows,
        'preflight_sha256':sha_file(preflight)}


def verify_kit_auxiliary_policy(settings, environment):
    """Verify optional helper prevention and scoped NGX cleanup before inference."""
    observed = {
        'omniclient_hub_mode': environment.get('OMNICLIENT_HUB_MODE'),
        'anonymous_data_disabled': environment.get('OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA'),
        'structured_log_enabled': settings.get('/structuredLog/enable'),
        'anonymous_telemetry_enabled': settings.get('/telemetry/enableAnonymousData'),
    }
    expected = {'omniclient_hub_mode': 'disabled', 'anonymous_data_disabled': '1',
        'structured_log_enabled': False, 'anonymous_telemetry_enabled': False}
    if canonical_sha256(observed) != canonical_sha256(expected):
        raise RuntimeError('Isaac optional helper prevention settings were not applied')
    identity = {'uid': os.getuid(), 'euid': os.geteuid(), 'gid': os.getgid(), 'egid': os.getegid(), 'pid':os.getpid()}
    if identity['uid'] != identity['euid'] or identity['gid'] != identity['egid']:
        raise RuntimeError('NGX scoped cleanup requires matching real/effective IDs')
    return {'schema_version': 1, 'verified': True, 'observed': observed,
        'identity': identity, 'verified_utc': utc_now(),
        'process_policy': AUXILIARY_PROCESS_POLICY,
        'ngx_child_reaper':verify_ngx_reaper_runtime(environment)}



def issue(issues,path,message):
    issues.append({"severity":"error","path":str(path),"message":message})


def read_object(path,issues,*,required=False):
    path=Path(path)
    try:
        if path.is_symlink() or not path.is_file():
            if required:raise ValueError("missing or unsafe JSON evidence")
            return None
        return strict_json_object(path.read_text())
    except (OSError,UnicodeError,ValueError) as exc:
        issue(issues,path,str(exc));return None


_strict_loads=strict_json_object


def validate_gpu_identity_evidence(directory,allocation):
    """Independently rejoin Slurm global IDs to UUIDs from archived metadata."""
    issues=[];value=verify_allocation_gpu_identity(directory,allocation,issues)
    if issues or value.get('verified') is not True:
        raise RuntimeError('GPU allocation identity evidence rejected: '+json.dumps(issues,sort_keys=True))
    if str(allocation.get('allocation'))!=Path(directory).name:
        raise RuntimeError('GPU evidence belongs to another allocation')
    return {**value,'gpu_assignments':allocation['gpu_assignments'],
            'sha256':sha_file(Path(directory)/'gpu_identity_evidence.json')}


def validate_platform_evidence(campaign_dir,manifest,group,owner,*,manifest_sha256):
    """Require the fresh Terra simulator's measured helper and exact-child proof."""
    c=Path(campaign_dir);out=c/'items'/group['episode_id'];path=out/'kit_auxiliary_policy.json'
    issues=[];row=read_object(path,issues,required=True)
    expected={'schema_version':1,'verified':True,'campaign':manifest['campaign'],
        'episode_id':group['episode_id'],'allocation':owner,'manifest_sha256':manifest_sha256,
        'process_policy':AUXILIARY_PROCESS_POLICY,
        'observed':{'omniclient_hub_mode':'disabled','anonymous_data_disabled':'1',
                    'structured_log_enabled':False,'anonymous_telemetry_enabled':False}}
    if not row or any(canonical_sha256(row.get(k))!=canonical_sha256(v) for k,v in expected.items()):
        raise RuntimeError('Simulator helper settings/source identity differ')
    try:
        when=datetime.fromisoformat(row['verified_utc'].replace('Z','+00:00'))
        if when.tzinfo is None:raise ValueError('naive timestamp')
    except (KeyError,TypeError,AttributeError,ValueError) as exc:
        raise RuntimeError('Kit helper verification timestamp is invalid') from exc
    checked=verify_ngx_reaper_evidence(c,manifest,group,row,issues)
    if issues or checked.get('verified') is not True:
        raise RuntimeError('Scoped NGX child-reap evidence rejected: '+json.dumps(issues,sort_keys=True))
    return {'required':True,'verified':True,'allocation':owner,'path':str(path),'ngx_child_reaper':checked}


def verify_ngx_reaper_evidence(campaign_dir,manifest,group,row,issues):
    c=Path(campaign_dir);out=c/'items'/group['episode_id'];path=out/'kit_auxiliary_policy.json'
    before=sum(i['severity']=='error' for i in issues)
    try:
        reaper=row['ngx_child_reaper'];identity=row['identity'];library=reaper['library'];shim=reaper['shim']
        expected_output=Path('/work/aspire_validate/full_results')/manifest['campaign']/'items'/group['episode_id']
        expected_shim=Path('/work/aspire_validate')/manifest.get('bootstrap_relative_root','terra_robodojo')/'ngx_child_reaper.so'
        if (reaper.get('schema_version')!=1 or reaper.get('verified') is not True
                or reaper.get('library_mapped_device_inode_verified') is not True
                or type(reaper.get('parent_pid')) is not int or reaper['parent_pid']<=0
                or identity.get('pid')!=reaper['parent_pid']
                or any(type(identity.get(k)) is not int for k in ('pid','uid','euid','gid','egid'))
                or identity['uid']!=identity['euid'] or identity['gid']!=identity['egid']
                or library.get('sha256')!=NGX_LIBRARY_SHA256
                or type(library.get('caller_return_offset')) is not int
                or library['caller_return_offset']!=NGX_CALLER_RETURN_OFFSET
                or library.get('updater_executable_absent_from_path') is not True
                or library.get('updater_path_candidates_all_absent') is not True
                or not isinstance(library.get('updater_path_candidates'),list)
                or not library['updater_path_candidates']
                or any(not isinstance(p,str) or Path(p).name!='nvidia-ngx-updater' for p in library['updater_path_candidates'])
                or library.get('uid')!=identity['uid'] or library.get('euid')!=identity['euid']
                or any(type(library.get(k)) is not int or library[k]<=0 for k in ('device','inode'))
                or not isinstance(library.get('path'),str) or not Path(library['path']).is_absolute()
                or not isinstance(library.get('kernel'),str) or not library['kernel']
                or shim.get('path')!=str(expected_shim) or shim.get('mapped_device_inode_verified') is not True
                or shim.get('vendor_carb_path')!='/isaac-sim/kit/libcarb.so'
                or shim.get('vendor_carb_mapped') is not True
                or shim.get('vendor_carb_mapping_method')!='rtld_noload_dlinfo_samefile_and_mapped_base'
                or shim.get('preload_paths')!=[str(expected_shim),'/isaac-sim/kit/libcarb.so']
                or shim.get('source_sha256')!=manifest['bootstrap_sha256']['ngx_child_reaper.c']
                or shim.get('binary_sha256')!=manifest['bootstrap_sha256']['ngx_child_reaper.so']
                or reaper.get('log_path')!=str(expected_output/'ngx_child_reaper.jsonl')):
            raise ValueError('NGX parent/library/callsite/loaded shim binding differs')
        preflight=read_object(out/'ngx_reaper_preflight.json',issues,required=True)
        if (not preflight or sha_file(out/'ngx_reaper_preflight.json')!=reaper.get('preflight_sha256')):
            raise ValueError('NGX library preflight differs from active simulator receipt')
        validate_ngx_library_transition(preflight,library)
        log=out/'ngx_child_reaper.jsonl'
        if log.is_symlink() or not log.is_file():raise ValueError('NGX raw receipt log is missing/unsafe')
        raw=log.read_bytes();size=reaper.get('log_prefix_bytes')
        if (type(size) is not int or size<=0 or size>len(raw) or not raw.endswith(b'\n')
                or not raw[:size].endswith(b'\n') or hashlib.sha256(raw[:size]).hexdigest()!=reaper.get('log_prefix_sha256')):
            raise ValueError('NGX immutable startup log prefix differs')
        startup=[_strict_loads(line) for line in raw[:size].decode('utf-8').splitlines()]
        rows=[_strict_loads(line) for line in raw.decode('utf-8').splitlines()]
        if canonical_sha256(startup)!=canonical_sha256(reaper.get('events')):
            raise ValueError('NGX embedded events differ from original startup bytes')
        checked=validate_ngx_reaper_events(startup,expected_parent_pid=reaper['parent_pid'])
        if any(canonical_sha256(checked[k])!=canonical_sha256(reaper.get(k))
               for k in ('registered_child_pids','reaped_child_pids')):
            raise ValueError('NGX startup child identity summary differs')
        final=validate_ngx_reaper_events(rows,expected_parent_pid=reaper['parent_pid'])
        return {'verified':sum(i['severity']=='error' for i in issues)==before,
            'parent_pid':reaper['parent_pid'],'registered_child_pids':final['registered_child_pids'],
            'reaped_child_pids':final['reaped_child_pids'],'log_bytes':len(raw),'startup_prefix_bytes':size}
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,UnicodeError) as exc:
        issue(issues,path,'Scoped NGX child-reap evidence failed: '+str(exc))
        return {'verified':False}


def verify_item_gpu_isolation(campaign_dir,group,report,episode_reports,allocations,issues):
    """Bind renderer/CUDA/planner receipts to the unique scoring process's GPU."""
    out=Path(campaign_dir)/'items'/group['episode_id'];before=sum(i['severity']=='error' for i in issues)
    gpu=read_object(out/'gpu_isolation.json',issues,required=True)
    result={'verified':False,'episode_id':group['episode_id']}
    try:
        if not isinstance(gpu,dict) or not isinstance(report,dict) or canonical_sha256(report.get('gpu_isolation'))!=canonical_sha256(gpu):
            raise ValueError('group report and saved GPU isolation receipt differ')
        committed=[r for r in episode_reports if r['native_score_committed']]
        if len(committed)!=1:raise ValueError('a unique committed episode report is required')
        episode=read_object(Path(committed[0]['path']),issues,required=True) or {}
        timing=episode['terra_timing'];worker=timing['worker_id']
        if not re.fullmatch(r'w0[0-8]',str(worker)):raise ValueError('unknown scoring worker')
        owners=[]
        for directory,allocation in allocations:
            path=Path(directory)/'workers'/worker/group['episode_id']/'group_attempt.json'
            if not path.exists():continue
            row=read_object(path,issues,required=True) or {}
            try:
                duration_seconds(row['simulator_started_utc'],timing['episode_started_utc'])
                duration_seconds(timing['episode_started_utc'],timing['episode_callback_finished_utc'])
                duration_seconds(timing['episode_callback_finished_utc'],row['simulator_finished_utc'])
            except (KeyError,ValueError,TypeError):continue
            owners.append((Path(directory),allocation,row))
        if len(owners)!=1:raise ValueError('committed report timing has no unique scoring allocation/attempt')
        directory,allocation,row=owners[0];slot=[0,0,1,1,2,2,3,3,3][int(worker[1:])]
        devices=allocation['gpu_assignments']
        if not isinstance(devices,list) or len(devices)!=4:raise ValueError('allocation must identify four GPUs')
        device=devices[slot];uuid=device['uuid'];normalized=uuid.lower().removeprefix('gpu-')
        if not re.fullmatch('[0-9a-f-]{36}',normalized):raise ValueError('invalid allocation GPU UUID')
        contract=read_object(out/'terra_contract.json',issues,required=True) or {}
        if (row.get('worker_id')!=worker or contract.get('worker')!=worker
                or row.get('episode_id')!=group['episode_id'] or row.get('group_id')!=group['group_id']
                or row.get('gpu_slot')!=slot or row.get('gpu_uuid')!=uuid):
            raise ValueError('scoring attempt/contract/GPU slot differ')
        active=gpu.get('active_vulkan',{});planners=gpu.get('planner_devices')
        if (gpu.get('verified') is not True or gpu.get('expected_uuid')!=uuid
                or gpu.get('cuda_uuid')!=normalized or gpu.get('cuda_visible_devices')!=uuid
                or type(gpu.get('cuda_device_index')) is not int or gpu['cuda_device_index']!=0
                or type(gpu.get('cuda_device_count')) is not int or gpu['cuda_device_count']!=1
                or 'L40S' not in str(gpu.get('cuda_name',''))
                or not isinstance(active,dict) or active.get('uuid_prefix')!=normalized[:8]
                or type(active.get('vulkan_index')) is not int or active['vulkan_index']!=int(device['index'])
                or not isinstance(planners,dict) or not planners
                or any(v not in ('cuda','cuda:0') for v in planners.values())):
            raise ValueError('actual CUDA/Vulkan/planner identity differs from the allocated physical GPU')
        result.update(verified=sum(i['severity']=='error' for i in issues)==before,
            allocation=directory.name,worker_id=worker,gpu_slot=slot,gpu_uuid=uuid,
            vulkan_index=active['vulkan_index'],cuda_device_index=0,cuda_device_count=1,
            planner_devices=planners,gpu_receipt_sha256=sha_file(out/'gpu_isolation.json'))
    except (OSError,KeyError,TypeError,ValueError) as exc:
        issue(issues,out/'gpu_isolation.json','Scored simulator GPU isolation failed: '+str(exc))
    return result


def duration_seconds(start,end):
    first=datetime.fromisoformat(start.replace('Z','+00:00'))
    last=datetime.fromisoformat(end.replace('Z','+00:00'))
    if first.tzinfo is None or last.tzinfo is None or last<first:raise ValueError('invalid chronology')
    return (last-first).total_seconds()


def verify_allocation_gpu_identity(directory,allocation,issues):
    """Reconcile saved Slurm global minors, visible indices, and physical UUIDs."""
    directory=Path(directory);path=directory/'gpu_identity_evidence.json'
    before=sum(i['severity']=='error' for i in issues)
    receipt=read_object(path,issues,required=True)
    try:
        if (not receipt or receipt.get('schema_version')!=1 or receipt.get('verified') is not True
                or receipt.get('metadata_read_errors')!=[]
                or allocation.get('gpu_identity_evidence_sha256')!=sha_file(path)):
            raise ValueError('GPU identity metadata receipt/digest is unverified')
        environment=receipt['environment']
        if (set(environment)!={'SLURM_JOB_ID','CUDA_VISIBLE_DEVICES','SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_GPUS_ON_NODE'}
                or environment.get('SLURM_JOB_ID')!=directory.name
                or receipt.get('mapping_rule')!='CUDA visible index or UUID -> nvidia-smi UUID; Slurm global device minor -> proc UUID; require exact UUID set'
                or (directory/'gpus.csv').read_text()!=receipt.get('gpu_csv')):
            raise ValueError('GPU identity allocation/visibility/raw CSV provenance differs')
        mapped=resolve_four_gpus(receipt['gpu_csv'],environment,proc_information=receipt['proc_nvidia_gpus'])
        if (canonical_sha256(mapped)!=canonical_sha256(allocation.get('gpu_assignments'))
                or canonical_sha256(mapped)!=canonical_sha256(receipt.get('allocated_devices'))):
            raise ValueError('GPU assignments differ from saved physical UUID reconciliation')
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
        issue(issues,path,'Allocation GPU identity audit failed: '+str(exc))
    return {'verified':sum(i['severity']=='error' for i in issues)==before,'allocation':directory.name}
