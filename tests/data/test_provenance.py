from pathlib import Path
import pytest
from robodata.data.provenance import freeze_source, verify_source, verify_imports


def tree(tmp_path):
    path = tmp_path / "src/robodata/__init__.py"
    path.parent.mkdir(parents=True)
    path.write_text("# code\n")
    return path


def test_drift_and_foreign_import_rejected(tmp_path):
    path = tree(tmp_path)
    record = freeze_source(tmp_path)
    assert verify_source(tmp_path, record) == record
    assert verify_imports(tmp_path, record, {"src/robodata/__init__.py": path})
    foreign = tmp_path / "foreign.py"
    foreign.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="origin"):
        verify_imports(tmp_path, record, {"src/robodata/__init__.py": foreign})
    path.write_text("# changed\n")
    with pytest.raises(ValueError, match="drift"):
        verify_source(tmp_path, record)


def test_extra_executable_rejected_and_local_bindings_excluded(tmp_path):
    tree(tmp_path)
    record = freeze_source(tmp_path)
    config = tmp_path / "configs/clusters/local.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"private_path":"/local"}')
    assert verify_source(tmp_path, record)
    extra = tmp_path / "src/robodata/new.py"
    extra.write_text("# extra code\n")
    with pytest.raises(ValueError, match="drift"):
        verify_source(tmp_path, record)


def test_inventory_tampering_rejected(tmp_path):
    tree(tmp_path)
    record = freeze_source(tmp_path)
    record["files_sha256"]["src/robodata/__init__.py"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        verify_source(tmp_path, record)


@pytest.mark.parametrize('relative', [
    'patches/native.patch', 'deploy/locks/model.txt',
    'third_party/aspire/src/cap_harness/robodata_patches/gripper.patch',
    'third_party/aspire/src/cap_harness/description/robot.urdf',
])
def test_packaged_data_and_deployment_pins_are_inventoried(tmp_path, relative):
    tree(tmp_path)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('original\n')
    record = freeze_source(tmp_path)
    assert relative in record['files_sha256']
    path.write_text('changed\n')
    with pytest.raises(ValueError, match='drift'):
        verify_source(tmp_path, record)


@pytest.mark.parametrize('directory', [False, True])
def test_symlink_sources_are_rejected_even_within_checkout(tmp_path, directory):
    tree(tmp_path)
    target = tmp_path / ('target' if directory else 'target.py')
    if directory:
        target.mkdir()
        (target / 'module.py').write_text('content')
    else:
        target.write_text('content')
    (tmp_path / 'src/robodata/link').symlink_to(target, target_is_directory=directory)
    with pytest.raises(ValueError, match='regular in-tree'):
        freeze_source(tmp_path)
