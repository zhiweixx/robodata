"""Sequential unused-state perception comparison with pinned Qwen and native budgets."""
from __future__ import annotations

import argparse
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
from perception_manifest import (ARMS, HARDWARE, PINNED_MODEL, arm_campaign,
    validate_manifest, verify_layouts, verify_profiles, verify_benchmark_sources,
    require_frozen_bootstrap, verify_bootstrap, bootstrap_source_hashes)
import perception_manifest
from paired_supervisor import pending_group
from perception_services import start_services




class PerceptionCampaign(full.Campaign):
    def __init__(self, args):
        manifest = validate_manifest(json.loads(args.manifest.read_text()))
        bootstrap = require_frozen_bootstrap(manifest)
        batch_sha256 = os.environ.get('QWEN_PERCEPTION_BATCH_SHA256')
        if batch_sha256 != bootstrap['files_sha256']['qwen_full/perception_eval.sbatch']:
            raise ValueError('submitted Slurm batch script differs from frozen bootstrap')
        bootstrap_evidence = verify_bootstrap(args.root, manifest, {
            'supervisor': __file__, 'manifest': perception_manifest.__file__,
            'full_supervisor': full.__file__,
            'paired_supervisor': pending_group.__code__.co_filename,
            'perception_services': start_services.__code__.co_filename})
        if args.campaign != manifest['campaign'] or args.mode not in ('full', 'resume'):
            raise ValueError('comparison identity or mode mismatch')
        args.mtp_tokens = 3
        super().__init__(args)
        # Full.Campaign defaults to the mutable checkout; its model startup helpers
        # must also come from the designated immutable shared profile.
        self.code = self.root / bootstrap['relative_root'] / 'qwen_full'
        self.report['bootstrap_verification'] = bootstrap_evidence
        self.report['batch_script_sha256'] = batch_sha256
        self.report['source_sha256'] = {
            p.name: full.sha_file(p) for p in self.code.iterdir()
            if p.is_file() and p.suffix in {'.py', '.sh', '.sbatch'}}
        if self.report['source_sha256'] != bootstrap_source_hashes(manifest):
            raise ValueError('actual shared supervisor source inventory differs from freeze')
        self.arm_dirs = {(arm, unit['unit_id']): self.out.parent / arm_campaign(self.manifest, arm, unit['unit_id'])
                         for unit in self.manifest['units'] for arm in ARMS}
        self.service_processes = []
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
        self.report['execution_hardware'] = HARDWARE
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
            'target_scored_episodes': 12, 'paired_states': 6, 'model': PINNED_MODEL,
            'inference': self.manifest['inference'], 'profiles': self.manifest['profiles'],
            'bootstrap': self.manifest['bootstrap'],
            'selection_note': self.manifest['selection_note'],
            'native_action_budgets': True, 'model_turn_cap': None,
            'output_token_cap': None, 'episode_walltime_cap': None,
            'history': self.manifest['history'], 'api_cost_usd': 0,
            'is_official_full_evaluation': False,
            'execution_hardware': HARDWARE, 'offline_skill_discovery': False}
        full.write_json(self.out / 'experiment_config.json', experiment)
        for (arm, uid), directory in self.arm_dirs.items():
            directory.mkdir(parents=True, exist_ok=True)
            arm_manifest = dict(self.manifest, arm=arm, campaign=directory.name,
                                comparison=self.campaign, unit_id=uid, group_count=1, total_target_episodes=1)
            existing = directory / 'manifest.json'
            if existing.exists() and json.loads(existing.read_text()) != arm_manifest:
                raise RuntimeError('existing arm manifest differs from comparison')
            full.write_json(existing, arm_manifest)
        gpu_text = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv'], text=True)
        (self.runtime / 'gpus.csv').write_text(gpu_text)
        self.gpus = full.resolve_gpu_uuids(gpu_text, self.devices)
        self.report['gpus'] = self.gpus
        for index in (0,):
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

    def run_group(self, unit, arm):
        group, uid, gpu_index = unit['group'], unit['unit_id'], 0
        gid = group['group_id']
        key = arm + '__' + uid
        profile = self.root / self.manifest['profiles'][arm]['relative_root']
        # Check immutable source bytes immediately before each group too.
        verify_profiles(self.root, self.manifest)
        group_out = self.arm_dirs[(arm, uid)] / 'groups' / gid
        group_out.mkdir(parents=True, exist_ok=True)
        attempt_dir = self.runtime / 'scored' / f'worker{gpu_index}' / key
        attempt_dir.mkdir(parents=True, exist_ok=True)
        relay = full.create_private_relay(self.job, key)
        full.write_json(attempt_dir / 'relay_location.json', {
            'live_node_local_path': str(relay), 'host': socket.gethostname(),
            'archive_path': str(attempt_dir / 'relay')})
        full.write_json(attempt_dir / 'perception_group.json', {
            'arm': arm, 'unit_id': uid, 'group_id': gid, 'campaign': self.arm_dirs[(arm, uid)].name,
            'profile': self.manifest['profiles'][arm], 'gpu': gpu_index})
        controller = sim = None
        sim_env = full.restore_simulator_proxies(self.env, os.environ)
        sim_env.update(CUDA_VISIBLE_DEVICES=self.gpus[gpu_index]['uuid'],
            ROBODOJO_STATE_DIR=str(self.root / 'full_runtime' / f'worker{gpu_index}'),
            SINGULARITY_BINDPATH=str(relay) + ':/aspire-relay',
            SINGULARITYENV_ASPIRE_SIM_LOG=str(Path('/work') /
                (attempt_dir / 'simulator.log').relative_to(self.simroot)),
            SINGULARITYENV_ASPIRE_PERCEPTION_CONFIG=str(Path('/work') /
                (self.runtime / 'providers.json').relative_to(self.simroot)))
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
            inside = Path('/work/aspire_validate') / self.manifest['bootstrap']['relative_root'] / 'qwen_full/perception_inside.sh'
            command = ['bash', str(self.simroot / 'scripts/container_exec.sh'), '/bin/bash',
                str(inside), group['task'], self.campaign,
                arm, uid, str(self.gpus[gpu_index]['index']), self.gpus[gpu_index]['uuid'],
                str(group['candidate_layout_ids'][0])]
            self.record('group_started', arm=arm, group_id=gid, gpu=gpu_index,
                        target_episodes=group['target_episodes'], profile=str(profile))
            with (attempt_dir / 'simulator.log').open('w') as f:
                sim = self.track(key + '_sim', subprocess.Popen(command, env=sim_env,
                    cwd=self.simroot, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
            rc = self.wait(sim, required_processes=(('inference controller', controller), *self.service_processes))
            if rc != 0:
                raise RuntimeError(f'{key} exited {rc}; exact-state checkpoint preserved')
            result = full.validate_group(group_out / 'group_report.json', group, self.simroot,
                                         expected_campaign=self.arm_dirs[(arm, uid)].name)
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

    def save_progress(self):
        with self.lock:
            rows = sorted(self.completed.values(), key=lambda row: row['comparison_group_id'])
            full.write_json(self.out / 'progress.json', {'campaign': self.campaign,
                'updated_utc': full.utc_now(), 'complete': len(rows) == 12,
                'groups_completed': len(rows), 'groups_target': 12,
                'scored': sum(row['scored'] for row in rows), 'target': 12,
                'success': sum(row['success'] for row in rows), 'groups': rows})

    def collect_pending(self):
        pending = []
        for unit in self.manifest['units']:
            uid, group = unit['unit_id'], unit['group']
            for arm in unit['arm_order']:
                result = pending_group(self.arm_dirs[(arm, uid)], group, self.simroot)
                key = arm + '__' + uid
                if result is None:
                    pending.append((unit, arm))
                else:
                    self.completed[key] = dict(result, arm=arm, comparison_group_id=key)
        return pending

    def run(self):
        pending = self.collect_pending()
        self.save_progress()
        if not pending:
            self.report['complete'] = True
            return
        self.start_server()
        self.providers, self.service_processes = start_services(self)
        for unit, arm in pending:
            if self.stop.is_set():
                break
            try:
                self.run_group(unit, arm)
            except BaseException as exc:
                failure = {'unit_id': unit['unit_id'], 'arm': arm,
                           'type': type(exc).__name__, 'message': str(exc)}
                self.failed.append(failure)
                self.stop.set()
                self.record('worker_interrupted', **failure)
                break
        self.report['failures'] = self.failed
        self.report['complete'] = len(self.completed) == 12
        if not self.report['complete']:
            raise RuntimeError('perception comparison incomplete; inspect exact-state checkpoint before resume')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=full.DEFAULT_ROOT)
    p.add_argument('--campaign', required=True)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--mode', choices=['full', 'resume'], default='full')
    args = p.parse_args()
    os.umask(0o077)
    campaign = PerceptionCampaign(args)
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
