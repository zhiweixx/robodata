"""Real services and RGB-D smoke on previously exposed general_pickup/seed0/layout0."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess

import full_supervisor as full
from perception_services import start_services


class Preflight(full.Campaign):
    def prepare(self):
        if not self.job.isdigit():
            raise RuntimeError('GPU preflight requires a Slurm allocation')
        self.source_profile = Path(os.environ['QWEN_PERCEPTION_PREFLIGHT_PROFILE']).resolve(strict=True)
        if os.environ.get('QWEN_REPAIRED_CAMPAIGN'):
            from repaired_subset_contract import load_contract, verify_profile, verify_executed_sources
            contract = load_contract(self.root, os.environ['QWEN_REPAIRED_CAMPAIGN'])
            self.report['source_profile_verification'] = verify_profile(self.root, contract)
            self.report['bootstrap_verification'] = verify_executed_sources(self.root, contract, {
                'qwen_full/perception_preflight.py': __file__})
            if self.source_profile != self.root / contract['source_profile']['relative_root']:
                raise RuntimeError('preflight profile differs from repaired campaign')
            self.manifest = {'profiles': {'perception': contract['source_profile']},
                             'perception_artifacts': contract['perception_artifacts']}
        from paired_manifest import source_files
        self.report['smoke_source_profile'] = str(self.source_profile)
        self.report['smoke_source_files_sha256'] = source_files(self.source_profile)
        text = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv'], text=True)
        (self.runtime / 'gpus.csv').write_text(text)
        self.gpus = full.resolve_gpu_uuids(text, self.devices)
        self.report.update(gpus=self.gpus, scope='perception_only_interface_smoke',
                           model_requests=0, scored_comparison_episodes=0)
        state = self.root / 'full_runtime/worker0'
        for child in ('runtime-home', 'cache/kit', 'cache/main', 'cache/computecache',
                      'logs', 'config', 'data/documents', 'data/kit-user', 'pkg'):
            (state / child).mkdir(parents=True, exist_ok=True)

    def run(self):
        _, services = start_services(self)
        output = self.out / 'groups/general_pickup_s0'
        output.mkdir(parents=True, exist_ok=True)
        env = full.restore_simulator_proxies(self.env, os.environ)
        env.update(CUDA_VISIBLE_DEVICES=self.gpus[0]['uuid'],
            ROBODOJO_STATE_DIR=str(self.root / 'full_runtime/worker0'),
            SINGULARITYENV_ASPIRE_PERCEPTION_PROFILE=str(Path('/work') / self.source_profile.relative_to(self.simroot)),
            SINGULARITYENV_ASPIRE_PERCEPTION_VENDOR_ROOT='/work/aspire_validate/ASPIRE/third_party/contact_graspnet_pytorch',
            SINGULARITYENV_ASPIRE_SIM_LOG=str(Path('/work') / (self.runtime / 'simulator.log').relative_to(self.simroot)),
            SINGULARITYENV_ASPIRE_PERCEPTION_CONFIG=str(Path('/work') / (self.runtime / 'providers.json').relative_to(self.simroot)))
        if os.environ.get('QWEN_PERCEPTION_SMOKE_QUERY'):
            env['SINGULARITYENV_ASPIRE_PERCEPTION_SMOKE_QUERY'] = os.environ['QWEN_PERCEPTION_SMOKE_QUERY']
        command = ['bash', str(self.simroot / 'scripts/container_exec.sh'), '/bin/bash',
                   str(Path('/work') / self.source_profile.relative_to(self.simroot) / 'qwen_full/perception_smoke_inside.sh'),
                   'general_pickup', self.campaign, '0', str(self.gpus[0]['index']), self.gpus[0]['uuid']]
        with (self.runtime / 'simulator.log').open('w') as handle:
            process = self.track('smoke_sim', subprocess.Popen(command, env=env, cwd=self.simroot,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        rc = self.wait(process, required_processes=services)
        if rc:
            raise RuntimeError(f'perception simulator smoke exited {rc}')
        reports = list(output.rglob('smoke_report.json'))
        if len(reports) != 1:
            raise RuntimeError(f'expected one excluded smoke report, found {len(reports)}')
        report = json.loads(reports[0].read_text())
        if report.get('ok') is not True or report.get('model_requests') != 0:
            raise RuntimeError('perception interface smoke did not pass')
        self.report.update(complete=True, smoke_report=str(reports[0]), smoke=report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=full.DEFAULT_ROOT)
    args = parser.parse_args()
    args.campaign = 'qwen38_perception_preflight_' + os.environ.get('SLURM_JOB_ID', 'invalid')
    args.manifest = args.root / 'qwen_full/full_manifest.json'
    args.mode, args.mtp_tokens = 'preflight', 3
    os.umask(0o077)
    campaign = Preflight(args)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda *_: campaign.stop.set())
    rc = 1
    try:
        campaign.prepare()
        campaign.run()
        rc = 0
    except BaseException as exc:
        campaign.report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        campaign.record('preflight_failed', **campaign.report['error'])
    finally:
        campaign.finish()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
