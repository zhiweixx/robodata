"""Trusted model download and gated-access check; credentials never reach eval."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, snapshot_download

ROOT = Path('/srv/ROBODATA_RUNTIME/isaac6/aspire_validate/qwen')
secret = Path('/home/ROBODATA_USER/.config/qwen-pilot/hf.token')
meta = secret.lstat()
if not stat.S_ISREG(meta.st_mode) or meta.st_uid != os.getuid() or meta.st_mode & 0o077:
    raise RuntimeError('HF credential must be private and user owned')
token = secret.read_text().strip()
api = HfApi(token=token)
began = time.monotonic()
sam = {'repo': 'facebook/sam3', 'downloaded_weights': False, 'enabled_in_eval': False}
try:
    info = api.model_info('facebook/sam3', files_metadata=True)
    weights = next(f.rfilename for f in info.siblings if f.rfilename.endswith(('.pt', '.safetensors')))
    metadata = get_hf_file_metadata(hf_hub_url('facebook/sam3', weights, revision=info.sha), token=token)
    sam.update(access_confirmed=True, revision=info.sha, checked_weight=weights, size=metadata.size)
except Exception as exc:
    sam.update(access_confirmed=False, error_type=type(exc).__name__,
               http_status=getattr(getattr(exc, 'response', None), 'status_code', None))
(ROOT/'sam3_access.json').write_text(json.dumps(sam, indent=2))
print(json.dumps({'sam3_access': sam}), flush=True)
info = api.model_info('Qwen/Qwen3.8-27B', files_metadata=True)
manifest = {'repo': info.id, 'revision': info.sha,
    'files': [{'path': x.rfilename, 'bytes': x.size,
               'sha256': getattr(x.lfs, 'sha256', None) if x.lfs else None} for x in info.siblings]}
(ROOT/'model_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({'download_start': info.id, 'revision': info.sha,
                  'listed_bytes': sum(x.size or 0 for x in info.siblings)}), flush=True)
path = snapshot_download(info.id, revision=info.sha, token=token,
                         cache_dir=str(ROOT/'hf-cache'), max_workers=8,
                         ignore_patterns=['*.md', '*.png', '*.jpg', '*.mp4'])
manifest.update(snapshot_path=path, download_wall_seconds=time.monotonic()-began)
(ROOT/'model_manifest.json').write_text(json.dumps(manifest, indent=2))
secret.unlink()
print(json.dumps({'download_complete': True, 'snapshot_path': path,
                  'wall_seconds': manifest['download_wall_seconds'],
                  'temporary_credential_removed': True}), flush=True)
