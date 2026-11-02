"""Two immutable harness profiles share one Qwen server on four allocated L40S."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

import full_supervisor as full
from audit_full_results import expected_result_path, native_details
from paired_manifest import (ARMS, PINNED_MODEL, arm_campaign, gpu_assignment,
                             validate_manifest, verify_layouts, verify_profiles, verify_benchmark_sources)


def pending_group(arm_dir, group, simroot):
    """Never replay a scored failure or an exhausted exact-state candidate queue."""
    path = arm_dir / 'groups' / group['group_id'] / 'group_report.json'
    progress = path.with_name('group_progress.json')
    snapshots = [p for p in (path, progress) if p.exists()]
    prior = json.loads(max(snapshots, key=lambda p: p.stat().st_mtime_ns).read_text()) if snapshots else {}
    if prior.get('complete'):
        return full.validate_group(max(snapshots, key=lambda p: p.stat().st_mtime_ns), group,
                                   simroot, expected_campaign=arm_dir.name)
    native = expected_result_path(simroot, arm_dir.name, group)
    details = native_details(json.loads(native.read_text()), group) if native.exists() else {}
    if len(details) == group['target_episodes']:
        raise RuntimeError('native scores complete but completion/video evidence needs repair; never replay')
    checkpoint = native.parent.parent / ('_resume_' + native.parent.name + '.json')
    if details and not checkpoint.exists():
        raise RuntimeError('refusing to replay scored states without native checkpoint')
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        abandoned = set(saved.get('abandoned_layout_ids', []))
        selected = set(group['candidate_layout_ids'])
        if selected & abandoned:
            raise RuntimeError('exact paired state was abandoned by native initialization; diagnose missing pair, do not replace or restart')
        completed = {row['layout_id'] for row in details.values()}
        if set(saved.get('completed_layout_ids', [])) != completed:
            raise RuntimeError('native checkpoint and scored layout IDs disagree')
    if prior.get('unstable', 0):
        raise RuntimeError('native invalid paired state needs diagnosis; no substitute is authorized')
    return None


class PairedCampaign(full.Campaign):
    def __init__(self, args):
        manifest = validate_manifest(json.loads(args.manifest.read_text()))
        if args.campaign != manifest['campaign'] or args.mode not in ('full', 'resume'):
            raise ValueError('comparison identity or mode mismatch')
        args.mtp_tokens = 3
        super().__init__(args)
        self.arm_dirs = {arm: self.out.parent / arm_campaign(self.manifest, arm) for arm in ARMS}
        if self.mode == 'full' and any((directory / 'groups').exists() for directory in self.arm_dirs.values()):
            raise ValueError('fresh paired run cannot reuse existing arm groups; use inspected resume')

    def prepare(self):
        profile_evidence = verify_profiles(self.root, self.manifest)
        layouts = verify_layouts(self.manifest)
        target_manifest = self.out / 'manifest.json'
        if target_manifest.exists() and json.loads(target_manifest.read_text()) != self.manifest:
            raise RuntimeError('comparison manifest changed')
        full.write_json(target_manifest, self.manifest)
        self.report['manifest_sha256'] = full.sha_file(target_manifest)
        self.report['profile_verification'] = profile_evidence
        self.report['verified_layouts'] = layouts
        self.report['verified_benchmark_source_files'] = verify_benchmark_sources(self.manifest)
        pinned = json.loads((self.root / 'qwen/model_manifest.json').read_text())
        if {key: pinned[key] for key in ('repo', 'revision')} != PINNED_MODEL:
            raise RuntimeError('model snapshot differs from paired manifest')
        self.snapshot = Path(pinned['snapshot_path']).resolve(strict=True)
        if json.loads((self.snapshot / 'config.json').read_text())['text_config']['max_position_embeddings'] != full.CONTEXT:
            raise RuntimeError('native model context drifted')
        self.report['model_manifest'] = pinned
        experiment = {'campaign': self.campaign, 'protocol': self.manifest['protocol'],
            'target_scored_episodes': 24, 'paired_states': 12, 'model': PINNED_MODEL,
            'inference': self.manifest['inference'], 'profiles': self.manifest['profiles'],
            'selection_note': self.manifest['selection_note'],
            'native_action_budgets': True, 'model_turn_cap': None,
            'output_token_cap': None, 'episode_walltime_cap': None,
            'history': self.manifest['history'], 'api_cost_usd': 0,
            'is_official_full_evaluation': False,
            'gpu_schedule': [{'group_id': group['group_id'], **gpu_assignment(i)}
                             for i, group in enumerate(self.manifest['groups'])]}
        full.write_json(self.out / 'experiment_config.json', experiment)
        for arm, directory in self.arm_dirs.items():
            directory.mkdir(parents=True, exist_ok=True)
            arm_manifest = dict(self.manifest, arm=arm, campaign=directory.name,
                                comparison=self.campaign, group_count=7, total_target_episodes=12)
            existing = directory / 'manifest.json'
            if existing.exists() and json.loads(existing.read_text()) != arm_manifest:
                raise RuntimeError('existing arm manifest differs from comparison')
            full.write_json(existing, arm_manifest)
        gpu_text = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv'], text=True)
        (self.runtime / 'gpus.csv').write_text(gpu_text)
        self.gpus = full.resolve_gpu_uuids(gpu_text, self.devices)
        self.report['gpus'] = self.gpus
        for index in (0, 3):
            state = self.root / 'full_runtime' / f'worker{index}'
            for child in ('runtime-home', 'cache/kit', 'cache/main', 'cache/computecache',
                          'logs', 'config', 'data/documents', 'data/kit-user', 'pkg'):
                (state / child).mkdir(parents=True, exist_ok=True)
        self.port = full.unused_loopback_port()
        self.url = f'http://127.0.0.1:{self.port}/v1'
        self.env['ASPIRE_LOCAL_BASE_URL'] = self.url
        self.env['PYTHONDONTWRITEBYTECODE'] = '1'
        full.write_json(self.runtime / 'allocation.json', self.report)
        self.update_inference_phases()

    def run_group(self, group, arm, gpu_index):
        gid = group['group_id']
        key = arm + '__' + gid
        profile = self.root / self.manifest['profiles'][arm]['relative_root']
        # Check immutable source bytes immediately before each group too.
        verify_profiles(self.root, self.manifest)
        group_out = self.arm_dirs[arm] / 'groups' / gid
        group_out.mkdir(parents=True, exist_ok=True)
        attempt_dir = self.runtime / 'scored' / f'worker{gpu_index}' / key
        attempt_dir.mkdir(parents=True, exist_ok=True)
        relay = full.create_private_relay(self.job, key)
        full.write_json(attempt_dir / 'relay_location.json', {
            'live_node_local_path': str(relay), 'host': socket.gethostname(),
            'archive_path': str(attempt_dir / 'relay')})
        full.write_json(attempt_dir / 'paired_group.json', {
            'arm': arm, 'group_id': gid, 'campaign': self.arm_dirs[arm].name,
            'profile': self.manifest['profiles'][arm], 'gpu': gpu_index})
        controller = sim = None
        sim_env = full.restore_simulator_proxies(self.env, os.environ)
        sim_env.update(CUDA_VISIBLE_DEVICES=self.gpus[gpu_index]['uuid'],
            ROBODOJO_STATE_DIR=str(self.root / 'full_runtime' / f'worker{gpu_index}'),
            SINGULARITY_BINDPATH=str(relay) + ':/aspire-relay',
            SINGULARITYENV_ASPIRE_SIM_LOG=str(Path('/work') /
                (attempt_dir / 'simulator.log').relative_to(self.simroot)))
        started = time.monotonic()
        try:
            config = {'usage_jsonl': str(attempt_dir / 'usage.jsonl'),
                'episode_ids': group['episode_ids'],
                'episode_seeds': {eid: 0 for eid in group['episode_ids']},
                'base_url': self.url, 'expected_context_length': full.CONTEXT,
                'socket_idle_timeout_s': None}
            full.write_json(attempt_dir / 'controller_config.json', config)
            with (attempt_dir / 'controller.log').open('w') as f:
                controller = self.track(key + '_controller', subprocess.Popen([
                    '/usr/bin/python3.12', '-B', str(profile / 'qwen_full/full_model_driver.py'),
                    '--config', str(attempt_dir / 'controller_config.json'),
                    '--relay-dir', str(relay)], env=self.env, cwd=profile,
                    stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
            ready_until = time.monotonic() + 60
            while not (relay / 'ready.json').exists():
                if controller.poll() is not None or time.monotonic() > ready_until:
                    raise RuntimeError('paired controller failed readiness')
                time.sleep(.2)
            command = ['bash', str(self.simroot / 'scripts/container_exec.sh'), '/bin/bash',
                '/work/aspire_validate/qwen_full/paired_inside.sh', group['task'], self.campaign,
                arm, str(self.gpus[gpu_index]['index']), self.gpus[gpu_index]['uuid'],
                str(group['target_episodes']), ','.join(map(str, group['candidate_layout_ids']))]
            self.record('group_started', arm=arm, group_id=gid, gpu=gpu_index,
                        target_episodes=group['target_episodes'], profile=str(profile))
            with (attempt_dir / 'simulator.log').open('w') as f:
                sim = self.track(key + '_sim', subprocess.Popen(command, env=sim_env,
                    cwd=self.simroot, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
            rc = self.wait(sim, required_processes=(('inference controller', controller),))
            if rc != 0:
                raise RuntimeError(f'{key} exited {rc}; exact-state checkpoint preserved')
            result = full.validate_group(group_out / 'group_report.json', group, self.simroot,
                                         expected_campaign=self.arm_dirs[arm].name)
            result.update(arm=arm, comparison_group_id=key)
            with self.lock:
                self.completed[key] = result
            self.save_progress()
            self.record('group_finished', arm=arm, group_id=gid, gpu=gpu_index,
                        wall_seconds=time.monotonic()-started, result=result)
        finally:
            if sim is not None:
                self.cleanup_child(key + '_sim', sim, grace=45)
            if controller is not None:
                self.cleanup_child(key + '_controller', controller)
            shutil.copytree(relay, attempt_dir / 'relay')
            self.record('relay_archived', arm=arm, group_id=gid, path=str(attempt_dir / 'relay'))
            shutil.rmtree(relay)

    def collect_pending(self):
        pending = []
        for index, group in enumerate(self.manifest['groups']):
            pair = []
            for arm in ARMS:
                result = pending_group(self.arm_dirs[arm], group, self.simroot)
                if result is None:
                    pair.append((arm, gpu_assignment(index)[arm]))
                else:
                    key = arm + '__' + group['group_id']
                    self.completed[key] = dict(result, arm=arm, comparison_group_id=key)
            if pair:
                pending.append((group, pair))
        return pending

    def run(self):
        pending = self.collect_pending()
        self.save_progress()
        if not pending:
            self.report['complete'] = True
            return
        self.start_server()
        # A task/config pair occupies the two GPUs together. Alternate arm/GPU
        # assignment between groups; state queues are identical within a pair.
        for group, arms in pending:
            if self.stop.is_set():
                break
            def run_one(arm, gpu):
                try:
                    self.run_group(group, arm, gpu)
                except BaseException as exc:
                    failure = {'group_id': group['group_id'], 'arm': arm,
                               'type': type(exc).__name__, 'message': str(exc)}
                    with self.lock:
                        self.failed.append(failure)
                    self.stop.set()
                    self.record('worker_interrupted', **failure)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run_one, arm, gpu) for arm, gpu in arms]
                for future in futures:
                    future.result()
        self.report['failures'] = self.failed
        self.report['complete'] = len(self.completed) == 14
        if not self.report['complete']:
            raise RuntimeError('paired comparison incomplete; inspect cause/checkpoint before resume')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=full.DEFAULT_ROOT)
    p.add_argument('--campaign', required=True)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--mode', choices=['full', 'resume'], default='full')
    args = p.parse_args()
    os.umask(0o077)
    campaign = PairedCampaign(args)
    with (campaign.out / 'campaign.lock').open('a') as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def interrupt(signum, frame):
            campaign.stop.set()
            campaign.record('allocation_signal', signal=signum)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(sig, interrupt)
        rc = 1
        try:
            campaign.prepare()
            campaign.run()
            rc = 0
        except BaseException as exc:
            campaign.report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            campaign.record('allocation_incomplete', **campaign.report['error'])
        finally:
            campaign.finish()
        return rc


if __name__ == '__main__':
    raise SystemExit(main())
