"""CPU-only campaign publication and frozen-origin regressions."""
import json
from pathlib import Path
import shutil

import pytest

from robodata.cli.main import plan_campaign, publish_campaign
from robodata.data.provenance import verify_source, verify_imports

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'experiments/manifests/robodojo_subset767.json'


def checkout(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'src/robodata').mkdir(parents=True)
    (repo / 'src/robodata/__init__.py').write_text('# synthetic package\n')
    (repo / 'pyproject.toml').write_text('[project]\nname="synthetic"\n')
    (repo / 'deploy/locks').mkdir(parents=True)
    shutil.copyfile(ROOT / 'deploy/common.py', repo / 'deploy/common.py')
    (repo / 'deploy/locks/provider.txt').write_text('dependency==1.0\n')
    (repo / 'patches').mkdir()
    (repo / 'patches/runtime.patch').write_text('synthetic patch\n')
    config = repo / 'configs/clusters/local.json'
    config.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / 'configs/clusters/rtx-slurm.example.json', config)
    return repo, config


def test_plan_expands_config_and_verifies_exact_frozen_origin(tmp_path, monkeypatch):
    repo, config = checkout(tmp_path)
    monkeypatch.setenv('ROBODATA_DATA', str(tmp_path / 'data'))
    destination = tmp_path / 'campaign'
    result = plan_campaign(repo, MANIFEST, config, destination, shards=4)
    assert result['groups'] == 51
    assert result['shards'] == 4
    frozen_config = json.loads((destination / 'cluster_config.json').read_text())
    assert frozen_config['paths']['benchmark_root'] == str(tmp_path / 'data/sources/RoboDojo')
    assert not (destination / 'source/configs/clusters/local.json').exists()
    record = json.loads((destination / 'source_inventory.json').read_text())
    assert 'patches/runtime.patch' in record['files_sha256']
    assert 'deploy/locks/provider.txt' in record['files_sha256']
    frozen = destination / 'source'
    assert verify_source(frozen, record)
    relative = 'src/robodata/__init__.py'
    assert verify_imports(frozen, record, {relative: frozen / relative})
    with pytest.raises(ValueError, match='origin'):
        verify_imports(frozen, record, {relative: repo / relative})
    plan = json.loads((destination / 'plan.json').read_text())
    assert plan['source_tree_sha256'] == record['tree_sha256']
    assigned = [g for a in plan['assignments'] for g in a['group_ids']]
    assert len(assigned) == len(set(assigned)) == 51


def test_plan_rejects_unresolved_environment_before_publication(tmp_path, monkeypatch):
    repo, config = checkout(tmp_path)
    monkeypatch.delenv('ROBODATA_DATA', raising=False)
    with pytest.raises(ValueError, match='unresolved'):
        plan_campaign(repo, MANIFEST, config, tmp_path / 'campaign')
    assert not (tmp_path / 'campaign').exists()


@pytest.mark.parametrize('folder', ['src', 'deploy', 'configs', 'patches', 'third_party/aspire'])
def test_plan_rejects_recursive_snapshot_destination(tmp_path, folder):
    repo, config = checkout(tmp_path)
    with pytest.raises(ValueError, match='snapshot directories'):
        plan_campaign(repo, MANIFEST, config, repo / folder / 'campaign')


def test_publication_never_overwrites_existing_empty_directory(tmp_path):
    temporary = tmp_path / 'temporary'
    temporary.mkdir()
    (temporary / 'plan.json').write_text('{}')
    output = tmp_path / 'campaign'
    output.mkdir()
    with pytest.raises(FileExistsError):
        publish_campaign(temporary, output)
    assert list(output.iterdir()) == []
    assert (temporary / 'plan.json').is_file()


def test_snapshot_rejects_symlink_before_copying_external_bytes(tmp_path, monkeypatch):
    repo, config = checkout(tmp_path)
    monkeypatch.setenv('ROBODATA_DATA', str(tmp_path / 'data'))
    outside = tmp_path / 'outside.py'
    outside.write_text('not part of checkout')
    (repo / 'src/robodata/external.py').symlink_to(outside)
    with pytest.raises(ValueError, match='regular in-tree'):
        plan_campaign(repo, MANIFEST, config, tmp_path / 'campaign')
    assert not (tmp_path / 'campaign').exists()
