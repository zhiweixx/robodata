"""Up to twenty unique states per family, reusing the full-budget policy.

This is a custom subset with explicit unique-state exceptions, not the full sweep.
All GPU and model/action execution stays inside the four-L40S Slurm allocation.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import subprocess
import time

import full_supervisor as full
from build_subset_manifest import validate_subset_manifest, verify_group_layout_sources


def subset_experiment_config(campaign, manifest, pinned):
    family_targets = {}
    for group in manifest['groups']:
        family_targets[group['family']] = family_targets.get(group['family'], 0) + group['target_episodes']
    return {
        'campaign': campaign, 'model': full.MODEL, 'model_revision': pinned['revision'],
        'protocol': manifest['protocol'], 'is_official_full_evaluation': False,
        'manifest_protocol': '20 distinct initial states per family, balanced across configurations; user-authorized smaller targets when native unique states are exhausted',
        'task_families': manifest['family_count'], 'task_configurations': manifest['task_count'],
        'eval_layout_seeds': manifest['eval_seeds'],
        'target_scored_episodes': manifest['total_target_episodes'],
        'per_family_target': manifest['per_family_target'],
        'family_targets': family_targets,
        'family_target_exceptions': {family: target for family, target in family_targets.items()
                                     if target != manifest['per_family_target']},
        'excluded_families': manifest['excluded_families'],
        'source_full_manifest_sha256': manifest['source_full_manifest_sha256'],
        'invalid_layouts': 'native skip to next manifest-selected unique initial state; exhaustion remains incomplete rather than repeating states',
        'harness': 'ASPIRE public API/program executor with custom ARX X5 bridge and native CuRobo',
        'sam3': False, 'contact_graspnet': False, 'offline_skill_search': False,
        'observations': 'public three-camera RGB, metric depth API, calibration, proprioception and task language',
        'history': 'all public observations and final assistant messages; no rolling truncation',
        'model_turn_cap': None, 'independent_output_token_cap': None,
        'native_context_tokens': full.CONTEXT, 'native_task_action_budgets': True,
        'evaluation_seed_note': 'seed0; manifest physical-state deduplication, original native layout IDs preserved; no scored failure replay',
        'gpu_roles': {'model': 'two L40S, BF16 tensor parallel', 'simulation': 'two independent one-GPU workers'},
        'inference_phase_policy': 'Settings apply to each allocation; prior scores retain original provenance.',
        'api_cost_usd': 0, 'leaderboard_submission': False,
    }


class SubsetCampaign(full.Campaign):
    def __init__(self, args):
        if not args.campaign.startswith('qwen38_subset20_'):
            raise ValueError('subset campaign must use its isolated qwen38_subset20_ namespace')
        manifest = json.loads(args.manifest.read_text())
        validate_subset_manifest(manifest)
        if args.mode not in ('full', 'resume'):
            raise ValueError('subset supports full or resume; no evaluation smoke groups')
        super().__init__(args)
        self.report['dataset_protocol'] = self.manifest['protocol']
        self.report['is_official_full_evaluation'] = False

    def prepare(self):
        # Verify the saved native asset bytes and semantic state hashes before
        # starting inference. Paths are host paths and must not be resolved in
        # the container or exposed to the model.
        layout_verification = [verify_group_layout_sources(group)
                               for group in self.manifest['groups']]
        verification_path = self.runtime / 'layout_source_verification.json'
        full.write_json(verification_path, {'verified_utc': full.utc_now(),
                                          'groups': layout_verification})
        self.report['layout_source_verification'] = {
            'path': str(verification_path), 'sha256': full.sha_file(verification_path),
            'groups': len(layout_verification)}
        campaign_manifest = self.out / 'manifest.json'
        if campaign_manifest.exists():
            if json.loads(campaign_manifest.read_text()) != self.manifest:
                raise RuntimeError('subset manifest changed; never merge unlike protocols')
        else:
            full.write_json(campaign_manifest, self.manifest)
        self.report['manifest_sha256'] = full.sha_file(campaign_manifest)
        pinned = json.loads((self.root / 'qwen/model_manifest.json').read_text())
        if pinned['repo'] != full.MODEL:
            raise RuntimeError('unexpected model snapshot')
        self.snapshot = Path(pinned['snapshot_path']).resolve(strict=True)
        cfg = json.loads((self.snapshot / 'config.json').read_text())
        if cfg['text_config']['max_position_embeddings'] != full.CONTEXT:
            raise RuntimeError('native model context changed')
        self.report['model_manifest'] = pinned
        full.write_json(self.out / 'experiment_config.json',
                        subset_experiment_config(self.campaign, self.manifest, pinned))
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
        full.write_json(self.runtime / 'allocation.json', self.report)
        self.update_inference_phases()

    def run_group(self, group, gpu_index, smoke=False):
        if smoke:
            raise ValueError('subset simulator smoke is not authorized')
        gid = group['group_id']
        group_out = self.out / 'groups' / gid
        group_out.mkdir(parents=True, exist_ok=True)
        attempt_dir = self.runtime / 'scored' / f'worker{gpu_index}' / gid
        attempt_dir.mkdir(parents=True, exist_ok=True)
        relay = full.create_private_relay(self.job, gid)
        full.write_json(attempt_dir / 'relay_location.json', {
            'live_node_local_path': str(relay), 'host': socket.gethostname(),
            'archive_path': str(attempt_dir / 'relay')})
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
                'episode_seeds': {eid: group['eval_seed'] for eid in group['episode_ids']},
                'base_url': self.url, 'expected_context_length': full.CONTEXT,
                'socket_idle_timeout_s': None}
            full.write_json(attempt_dir / 'controller_config.json', config)
            with (attempt_dir / 'controller.log').open('w') as f:
                controller = self.track(gid + '_controller', subprocess.Popen([
                    '/usr/bin/python3.12', str(self.code / 'full_model_driver.py'),
                    '--config', str(attempt_dir / 'controller_config.json'),
                    '--relay-dir', str(relay)], env=self.env, cwd=self.root,
                    stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
            ready_until = time.monotonic() + 60
            while not (relay / 'ready.json').exists():
                if controller.poll() is not None or time.monotonic() > ready_until:
                    raise RuntimeError('local full controller failed readiness')
                time.sleep(.2)
            command = ['bash', str(self.simroot / 'scripts/container_exec.sh'), '/bin/bash',
                '/work/aspire_validate/qwen_full/subset_inside.sh', group['task'], self.campaign,
                str(group['eval_seed']), str(self.gpus[gpu_index]['index']),
                self.gpus[gpu_index]['uuid'], str(group['target_episodes']),
                ','.join(str(value) for value in group['candidate_layout_ids'])]
            self.record('group_started', group_id=gid, gpu=gpu_index, smoke=False,
                        target_episodes=group['target_episodes'])
            with (attempt_dir / 'simulator.log').open('w') as f:
                sim = self.track(gid + '_sim', subprocess.Popen(command, env=sim_env,
                    cwd=self.simroot, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
            rc = self.wait(sim, required_processes=(('inference controller', controller),))
            if rc != 0:
                raise RuntimeError(f'simulator group {gid} exited {rc}; checkpoint preserved')
            result = full.validate_group(group_out / 'group_report.json', group, self.simroot)
            with self.lock:
                self.completed[gid] = result
            self.save_progress()
            self.record('group_finished', group_id=gid, gpu=gpu_index, smoke=False,
                        wall_seconds=time.monotonic()-started, result=result)
        finally:
            if sim is not None:
                self.cleanup_child(gid + '_sim', sim, grace=45)
            if controller is not None:
                self.cleanup_child(gid + '_controller', controller)
            shutil.copytree(relay, attempt_dir / 'relay')
            self.record('relay_archived', group_id=gid, path=str(attempt_dir / 'relay'))
            shutil.rmtree(relay)

    def collect_pending_groups(self):
        pending = queue.Queue()
        for group in self.manifest['groups']:
            report = self.out / 'groups' / group['group_id'] / 'group_report.json'
            prior = json.loads(report.read_text()) if report.exists() else None
            if prior and prior.get('complete'):
                self.completed[group['group_id']] = full.validate_group(report, group, self.simroot)
                continue
            if prior and prior.get('scored', 0) > 0:
                native_result = full.path_host(prior['native_result_path'], self.simroot)
                checkpoint = native_result.parent.parent / ('_resume_' + prior['native_run_id'] + '.json')
                if not checkpoint.exists():
                    raise RuntimeError(f"refusing to replay scored layouts without checkpoint: {group['group_id']}")
                if prior.get('scores_complete') and not prior.get('videos_validated'):
                    raise RuntimeError('scoring complete but video evidence needs repair')
            pending.put(group)
        return pending

    def run(self):
        pending = self.collect_pending_groups()
        self.save_progress()
        if pending.empty():
            self.report['complete'] = True
            return
        # Only the inherited non-scored multimodal model smoke runs. No skipped
        # task is instantiated and no simulator smoke is mixed into this subset.
        self.start_server()
        def worker(index):
            while not self.stop.is_set():
                try:
                    group = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    self.run_group(group, index)
                except BaseException as exc:
                    self.failed.append({'group_id': group['group_id'],
                                        'type': type(exc).__name__, 'message': str(exc)})
                    self.stop.set()
                    self.record('worker_interrupted', **self.failed[-1])
                    return
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, index) for index in (0, 3)]
            for future in futures:
                future.result()
        self.report['failures'] = self.failed
        self.report['complete'] = len(self.completed) == self.manifest['group_count']
        if not self.report['complete']:
            raise RuntimeError('subset allocation incomplete; resume native checkpoints after inspection')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=full.DEFAULT_ROOT)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--mode', choices=['full', 'resume'], default='full')
    parser.add_argument('--mtp-tokens', type=int, choices=[0, 1, 3], default=3)
    args = parser.parse_args()
    if not args.campaign.replace('_', '').isalnum():
        raise ValueError('invalid campaign ID')
    os.umask(0o077)
    campaign = SubsetCampaign(args)
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
        raise SystemExit(rc)


if __name__ == '__main__':
    main()
