"""Commit optimizer state before publishing progress. Only scalars go to W&B."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys
import uuid


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


@contextlib.contextmanager
def locked(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / '.lock').open('a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read_json(path):
    return json.loads(path.read_text())


def atomic(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    temp = Path(raw)
    try:
        with os.fdopen(fd, 'wb') as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, data):
    body = json.dumps(data, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n'
    atomic(path, lambda f: f.write(body))


def write_metrics(run, rows):
    body = ''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in rows).encode()
    atomic(run / 'metrics.jsonl', lambda f: f.write(body))


class OfflineMetrics:
    """Offline W&B cannot resume. Rebuild its history from committed updates on restart."""
    def __init__(self, root, config, rows):
        import wandb
        # Isolate each offline session; only the canonical session should be synced.
        session = root / 'offline' / uuid.uuid4().hex
        session.mkdir(parents=True)
        self.root = root
        # W&B exposes sys.argv as a computed backend setting independently of console
        # capture. Keep CLI input/output paths out of the offline event stream.
        original_argv = sys.argv
        sys.argv = ['robodata-sft']
        try:
            self.run = wandb.init(
                mode='offline', project='robodojo-qwen-sft', dir=str(session),
                id=config['run_id'], config=config,
                settings=wandb.Settings(console='off', disable_code=True, save_code=False,
                                        disable_git=True, x_disable_meta=True, x_disable_stats=True,
                                        x_disable_machine_info=True, x_save_requirements=False,
                                        disable_job_creation=True, x_disable_viewer=True),
            )
        finally:
            sys.argv = original_argv
        self.path = Path(self.run.dir).parent
        for row in rows:
            self.update(row)

    def update(self, row):
        # A later step flushes this row. Leave the last one open for validation.
        self.run.log(row, step=row['optimizer_step'], commit=False)

    def finish(self, summary=None):
        if summary is None:
            self.run.finish(exit_code=1)
            return
        validation = {key: value for key, value in summary.items() if key.startswith('val/')}
        self.run.log(validation, step=summary['epoch/updates'], commit=True)
        self.run.summary.update(summary)
        self.run.finish()
        write_json(self.root / 'wandb_offline.json', {
            'run_id': self.run.id, 'path': str(self.path.relative_to(self.root)),
            'project': 'robodojo-qwen-sft', 'mode': 'offline',
        })
