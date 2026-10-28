"""CPU-only checks of reproducible install and frozen Slurm boundaries."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deploy'))
import common
import manage
from robodata.data.provenance import canonical_sha256,freeze_source


def config(tmp_path,monkeypatch):
    monkeypatch.setenv('ROBODATA_DATA',str(tmp_path/'data'))
    return common.load_config(ROOT/'configs/clusters/rtx-slurm.example.json')


def campaign(tmp_path):
    path=tmp_path/'campaign';source=path/'source'
    (source/'src/robodata').mkdir(parents=True)
    (source/'src/robodata/__init__.py').write_text('')
    (source/'deploy').mkdir();(source/'deploy/worker.sbatch').write_bytes((ROOT/'deploy/worker.sbatch').read_bytes())
    inventory=freeze_source(source)
    manifest={'groups':[{'group_id':'g0'},{'group_id':'g1'}]}
    cfg={'scheduler':{'account':'lab','partition':'rtx','gres':'gpu:l40s:4','cpus_per_task':32,'memory':'240G','time':'24:00:00','max_concurrent_shards':2}}
    plan={'groups':['g0','g1'],'assignments':[{'shard_index':0,'group_ids':['g0']},{'shard_index':1,'group_ids':['g1']}],
          'source_tree_sha256':inventory['tree_sha256'],'cluster_config_sha256':canonical_sha256(cfg),'manifest_sha256':canonical_sha256(manifest)}
    for name,value in [('source_inventory',inventory),('manifest',manifest),('cluster_config',cfg),('plan',plan)]:
        (path/(name+'.json')).write_text(json.dumps(value))
    return path


def test_config_expands_external_paths_and_rejects_missing_variables(tmp_path,monkeypatch):
    c=config(tmp_path,monkeypatch)
    assert all(Path(value).is_absolute() for value in c['paths'].values())
    monkeypatch.delenv('ROBODATA_DATA')
    with pytest.raises(ValueError,match='unresolved'):common.load_config(ROOT/'configs/clusters/rtx-slurm.example.json')


def test_token_must_be_private_regular_file(tmp_path):
    secret=tmp_path/'credential';secret.write_text('test-value');secret.chmod(0o600)
    assert common.private_token(secret)=='test-value'
    secret.chmod(0o644)
    with pytest.raises(ValueError,match='private'):common.private_token(secret)
    link=tmp_path/'link';link.symlink_to(secret)
    with pytest.raises(ValueError):common.private_token(link)


def test_gpu_operations_require_slurm(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='Slurm'):common.require_slurm()


def test_patch_and_source_pins_are_immutable():
    pins=json.loads((ROOT/'deploy/sources.lock.json').read_text())
    for spec in pins['sources'].values():
        assert len(spec['revision'])==40 and spec['url'].startswith('https://github.com/')
        if 'patch' in spec:assert common.sha256(ROOT/spec['patch'])==spec['patch_sha256']
    assert pins['graspnet_checkpoint']['bytes']==26628146


def test_license_gate_precedes_container_download(tmp_path,monkeypatch):
    c=config(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='license'):manage.build_container(c)


def test_existing_container_requires_same_digest(tmp_path,monkeypatch):
    c=config(tmp_path,monkeypatch);c['container']['accept_eula']=True
    c['container']['source']='docker://nvcr.io/nvidia/isaac-sim@sha256:'+'a'*64
    image=Path(c['paths']['isaac_container']);image.parent.mkdir(parents=True);image.write_bytes(b'image')
    record=dict(common.artifact(image),source='docker://nvcr.io/nvidia/isaac-sim@sha256:'+'b'*64)
    common.write_json(image.with_suffix('.manifest.json'),record)
    with pytest.raises(RuntimeError,match='source digest differs'):manage.build_container(c)


def test_submit_uses_frozen_script_and_real_runner_interface(tmp_path):
    path=campaign(tmp_path);result=manage.submit(path,resume=True,dry_run=True);args=result['command']
    assert str(path/'source/deploy/worker.sbatch') in args
    assert args[-3:]==[str(path/'source'),str(path),'resume']
    worker=(path/'source/deploy/worker.sbatch').read_text()
    assert '--campaign-dir' in worker and '--shard-index' in worker and '--frozen' in worker
    assert 'ROBODATA_BATCH_SHA256' in worker and 'TRITON_PTXAS_PATH' in worker
    assert '--config' not in worker


def test_submit_rejects_source_drift(tmp_path):
    path=campaign(tmp_path)
    (path/'source/deploy/worker.sbatch').write_text('different')
    with pytest.raises(ValueError,match='drift'):manage.submit(path,dry_run=True)


def test_submit_rejects_duplicate_group_ownership(tmp_path):
    path=campaign(tmp_path);plan=json.loads((path/'plan.json').read_text())
    plan['assignments'][1]['group_ids']=['g0'];(path/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='exactly once'):manage.submit(path,dry_run=True)


def test_submit_inherits_control_python_and_blocks_active_campaign(tmp_path,monkeypatch):
    path=campaign(tmp_path)
    def fake(args,**kwargs):
        if args[0]=='squeue':return 'robodata-campaign\n'
        raise AssertionError('must not submit duplicate job')
    monkeypatch.setattr(manage.subprocess,'check_output',fake)
    with pytest.raises(RuntimeError,match='already has active'):manage.submit(path)


def test_source_inventory_detects_tracked_changes_and_symlink_targets(tmp_path):
    root=tmp_path/'external';root.mkdir()
    subprocess.run(['git','init','-q',str(root)],check=True)
    (root/'a.py').write_text('original');(root/'link').symlink_to('a.py')
    subprocess.run(['git','-C',str(root),'add','.'],check=True)
    before=manage.source_files(root)
    assert {x['kind'] for x in before}=={'file','symlink'}
    (root/'a.py').write_text('modified')
    assert manage.source_files(root)!=before


def test_shell_entrypoints_are_executable_and_parse():
    scripts=list((ROOT/'deploy').glob('*.sh'))+list((ROOT/'deploy').glob('*.sbatch'))
    for script in scripts:
        assert os.access(script,os.X_OK)
        subprocess.run(['bash','-n',str(script)],check=True)


def test_provider_snapshot_reconstruction_and_model_resolver_are_distinct(tmp_path,monkeypatch):
    c=config(tmp_path,monkeypatch);commands=[]
    monkeypatch.setattr(manage,'run',lambda args,**kw:commands.append(args))
    monkeypatch.setattr(manage.subprocess,'check_output',lambda *a,**kw:'{}')
    monkeypatch.setenv('CUDA_HOME','/unrelated/toolkit')
    manage.setup_host_environment(c,'sam3')
    assert '--no-deps' in next(x for x in commands if '-r' in x)
    assert 'CUDA_HOME' not in os.environ
    commands.clear();manage.setup_host_environment(c,'model')
    assert '--no-deps' not in next(x for x in commands if '-r' in x)
