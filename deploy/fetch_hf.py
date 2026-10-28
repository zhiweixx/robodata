"""Explicit pinned Hugging Face downloads; credentials never enter manifests."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from common import REPO,agent,artifact,load_config,private_token,sha256,write_json

def fetch(config,components,token_file=None):
    from huggingface_hub import snapshot_download,hf_hub_download
    token=private_token(token_file)
    pins=json.loads((REPO/'deploy/sources.lock.json').read_text());p=config['paths']
    if 'assets' in components:
        spec=pins['assets'];destination=Path(p['assets_root']).parent
        snapshot_download(spec['repo'],repo_type=spec['repo_type'],revision=spec['revision'],local_dir=destination,allow_patterns=spec['allow_patterns'],token=token)
        if not Path(p['assets_root']).is_dir():raise RuntimeError('asset download did not produce required Assets directory')
        write_json(destination/'assets.manifest.json',spec)
    if 'model' in components:
        spec=agent(config)['model'];destination=Path(p['model_snapshot'])
        snapshot_download(spec['repo'],revision=spec['revision'],local_dir=destination,token=token,ignore_patterns=['*.md','*.png','*.jpg','*.mp4'])
        files=[artifact(x) for x in sorted(destination.rglob('*')) if x.is_file() and '.cache' not in x.parts]
        for name in ('config.json','chat_template.jinja'):
            if not (destination/name).is_file():raise RuntimeError(f'pinned model missing {name}')
        write_json(p['model_manifest'],dict(spec,snapshot_path=str(destination.resolve()),files=files))
    if 'providers' in components:
        spec=pins['sam3'];destination=Path(p['sam3_manifest']).parent/'sam3'
        files=[]
        for name,expected in spec['files'].items():
            file=Path(hf_hub_download(spec['repo'],name,revision=spec['revision'],local_dir=destination,token=token))
            if sha256(file)!=expected:raise RuntimeError(f'SAM3 artifact checksum mismatch: {name}')
            files.append(dict(artifact(file),file=name))
        write_json(p['sam3_manifest'],{'repo':spec['repo'],'revision':spec['revision'],'files':files})
        check=pins['graspnet_checkpoint'];file=Path(p['perception_vendor_root'])/check['relative_path']
        if not file.is_file() or file.stat().st_size!=check['bytes'] or sha256(file)!=check['sha256']:raise RuntimeError('CGN checkpoint differs from pinned source; run fetch --components sources first')
        write_json(Path(p['sam3_manifest']).parent/'weights.manifest.json',{'sam3':files,'graspnet':artifact(file)})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--components',nargs='+',choices=['assets','model','providers'],required=True);ap.add_argument('--hf-token-file')
    args=ap.parse_args()
    try:fetch(load_config(args.config),args.components,args.hf_token_file)
    except Exception as exc:
        # Third-party HTTP errors may embed signed URLs; never print their text.
        print(json.dumps({'error_type':type(exc).__name__,'message':'Pinned download failed; check access, free space, and artifact identity. Credentials were not stored or deleted.'}));return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
