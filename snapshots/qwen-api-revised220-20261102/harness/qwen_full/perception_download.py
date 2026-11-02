"""Download gated SAM3 assets without exposing credentials to an evaluated agent."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--credential', type=Path, required=True)
    parser.add_argument('--revision', default='3c879f39826c281e95690f02c7821c4de09afae7')
    args = parser.parse_args()
    meta = args.credential.lstat()
    if (not stat.S_ISREG(meta.st_mode) or meta.st_uid != os.getuid()
            or stat.S_IMODE(meta.st_mode) & 0o077):
        raise ValueError('credential must be a private, user-owned regular file')
    token = args.credential.read_text().strip()
    from huggingface_hub import HfApi, hf_hub_download
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    info = HfApi(token=token).model_info('facebook/sam3', revision=args.revision)
    files = {item.rfilename for item in info.siblings}
    required = ['config.json', 'sam3.pt']
    if not set(required) <= files:
        raise RuntimeError('pinned SAM3 revision lacks files required by upstream provider')
    manifest = {'repo': 'facebook/sam3', 'revision': info.sha, 'files': []}
    for filename in required:
        print(json.dumps({'event': 'download_start', 'file': filename}), flush=True)
        path = Path(hf_hub_download('facebook/sam3', filename, revision=info.sha,
                                   token=token, cache_dir=str(root / 'hf-cache')))
        manifest['files'].append({'file': filename, 'path': str(path),
                                 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    manifest['wall_seconds'] = time.monotonic() - started
    (root / 'sam3_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    args.credential.unlink()
    print(json.dumps({'event': 'download_complete', 'revision': info.sha,
                      'manifest': str(root / 'sam3_manifest.json'),
                      'temporary_credential_removed': True}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Avoid embedding remote URLs or credential-bearing exception messages.
        print(json.dumps({'event': 'download_failed', 'error_type': type(exc).__name__,
                          'status': getattr(getattr(exc, 'response', None), 'status_code', None)}), flush=True)
        raise SystemExit(1)
