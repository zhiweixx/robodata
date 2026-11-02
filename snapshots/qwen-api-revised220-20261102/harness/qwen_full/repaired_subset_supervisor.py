"""Frozen repaired-harness rerun of the unchanged 767-episode subset."""
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
from types import SimpleNamespace

import full_supervisor as full
import subset_supervisor
import perception_services
import repaired_subset_contract
from audit_full_results import expected_result_path, native_details
from paired_manifest import verify_benchmark_sources


def server_command(python, snapshot, port, mtp_tokens):
    # The prior runtime used vLLM's seed=0 default. State it explicitly so the
    # retained launch command independently proves the identical seed choice.
    return full.server_command(python, snapshot, port, mtp_tokens) + ['--seed', '0']


def pending_group(campaign_dir, group, simroot):
    """Resume only the native unscored suffix; invalid candidates may be skipped."""
    directory = Path(campaign_dir) / 'groups' / group['group_id']
    reports = [p for p in (directory / 'group_report.json', directory / 'group_progress.json')
               if p.exists()]
    reports.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    native = expected_result_path(Path(simroot), Path(campaign_dir).name, group)
    details = native_details(json.loads(native.read_text()), group) if native.exists() else {}
    for report in reports:
        prior = json.loads(report.read_text())
        if prior.get('complete'):
            return full.validate_group(report, group, simroot,
                                       expected_campaign=Path(campaign_dir).name)
        if prior.get('scored', 0) > len(details):
            raise RuntimeError('group report has scores missing from native evidence; never replay')
    if len(details) == group['target_episodes']:
        raise RuntimeError('native scores complete but completion/video evidence needs repair; never replay')
    checkpoint = native.parent.parent / ('_resume_' + native.parent.name + '.json')
    if details and not checkpoint.exists():
        raise RuntimeError('refusing to replay scored layouts without native checkpoint')
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        candidates = set(group['candidate_layout_ids'])
        completed = {row['layout_id'] for row in details.values()}
        saved_completed = saved.get('completed_layout_ids', [])
        abandoned = saved.get('abandoned_layout_ids', [])
        for values in (saved_completed, abandoned):
            if (not isinstance(values, list) or any(type(i) is not int for i in values)
                    or len(values) != len(set(values)) or not set(values) <= candidates):
                raise RuntimeError('native checkpoint contains invalid or unauthorized layout IDs')
        if set(saved_completed) != completed or completed & set(abandoned):
            raise RuntimeError('native checkpoint and scored layout IDs disagree')
        if not candidates - completed - set(abandoned):
            raise RuntimeError('unique candidate pool exhausted before target; no replacement/restart authorized')
    return None


def fresh_campaign_guard(root, campaign, manifest):
    out = Path(root) / 'full_results' / campaign
    if any((out / 'groups').glob('*')):
        raise ValueError('fresh repaired run cannot reuse existing groups; use inspected resume')
    for group in manifest['groups']:
        native = expected_result_path(Path(root).parent, campaign, group)
        checkpoint = native.parent.parent / ('_resume_' + native.parent.name + '.json')
        if native.exists() or checkpoint.exists():
            raise ValueError('fresh repaired run cannot reuse native scoring/checkpoints')


class RepairedSubsetCampaign(subset_supervisor.SubsetCampaign):
    def __init__(self, args):
        root = args.root.resolve(strict=True)
        contract = repaired_subset_contract.load_contract(root, args.campaign)
        verification = repaired_subset_contract.verify_profile(root, contract)
        profile = root / contract['source_profile']['relative_root']
        bootstrap = repaired_subset_contract.verify_executed_sources(root, contract, {
            'qwen_full/repaired_subset_supervisor.py': __file__,
            'qwen_full/repaired_subset_contract.py': repaired_subset_contract.__file__,
            'qwen_full/subset_supervisor.py': subset_supervisor.__file__,
            'qwen_full/full_supervisor.py': full.__file__,
            'qwen_full/perception_services.py': perception_services.__file__})
        batch_hash = os.environ.get('QWEN_REPAIRED_BATCH_SHA256')
        if batch_hash != contract['source_profile']['files_sha256']['qwen_full/repaired_subset_eval.sbatch']:
            raise ValueError('submitted batch script differs from frozen repaired profile')
        args.manifest = root / 'full_results' / args.campaign / 'manifest.json'
        if args.mode == 'full':
            fresh_campaign_guard(root, args.campaign, json.loads(args.manifest.read_text()))
        args.mtp_tokens = contract['inference']['mtp_tokens']
        self.execution_contract = contract
        super().__init__(args)
        self.source_profile = profile
        self.code = profile / 'qwen_full'
        self.service_processes = []
        self.env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            PYTHONPATH=os.pathsep.join(str(profile / rel) for rel in
                                      ('qwen_full', 'qwen_integration', 'ASPIRE/src')))
        self.report.update(source_profile_verification=verification,
            execution_contract_sha256=full.sha_file(self.out / 'execution_manifest.json'),
            bootstrap_verification=bootstrap, batch_script_sha256=batch_hash,
            execution_hardware=contract['hardware'], providers=contract['providers'],
            source_sha256={p.name: full.sha_file(p) for p in self.code.iterdir()
                          if p.is_file() and p.suffix in {'.py', '.sh', '.sbatch'}})

    def prepare(self):
        self.report['verified_benchmark_source_files'] = verify_benchmark_sources(self.manifest)
        pinned = json.loads((self.root / 'qwen/model_manifest.json').read_text())
        if {key: pinned.get(key) for key in ('repo', 'revision')} != self.execution_contract['model']:
            raise RuntimeError('model snapshot differs from frozen repaired contract')
        super().prepare()
        config_path = self.out / 'experiment_config.json'
        config = json.loads(config_path.read_text())
        enabled = self.execution_contract['providers']['sam3']
        config.update(execution_contract_sha256=self.report['execution_contract_sha256'],
            source_profile=self.execution_contract['source_profile'], sam3=enabled,
            contact_graspnet=enabled, offline_skill_search=False,
            execution_hardware=self.execution_contract['hardware'],
            gpu_roles=self.execution_contract['hardware'],
            rerun_note='Fresh repaired-harness campaign; original subset candidate IDs and native budgets retained; no historical scores included.')
        full.write_json(config_path, config)
        if not enabled:
            full.write_json(self.runtime / 'providers.json', {
                'providers': {'sam3': False, 'contact_graspnet': False},
                'provenance': {'offline_skill_discovery': False}, 'grasp_calibration': None})

    def start_server(self):
        command = server_command(self.python, self.snapshot, self.port, self.mtp_tokens)
        self.report['server_command'] = command
        env = dict(self.env, CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in self.gpus[1:3]))
        with (self.runtime / 'server.log').open('w') as f:
            self.server = self.track('server', subprocess.Popen(command, env=env,
                cwd=self.root, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
        started = time.monotonic()
        self.record('server_started', pid=self.server.pid)
        self.report['advertised_models'] = full.wait_ready(self.server, self.url,
            started + 2400, self.record)
        verified = full.verify_inference_variant(self.runtime / 'server.log', self.mtp_tokens)
        self.report['inference_variant_evidence'] = verified
        self.report['actual_inference_variant'] = verified['variant']
        self.report['server_startup_seconds'] = time.monotonic() - started
        full.write_json(self.runtime / 'allocation.json', self.report)
        self.update_inference_phases()
        self.record('server_ready', seconds=self.report['server_startup_seconds'])
        smoke = self.runtime / 'model_smoke'
        command = [str(self.python), str(self.code / 'full_model_smoke.py'),
            '--base-url', self.url, '--output', str(smoke / 'full_model_smoke.json')]
        with (self.runtime / 'model_smoke.log').open('w') as f:
            process = self.track('model_smoke', subprocess.Popen(command, env=self.env,
                cwd=self.root, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
        try:
            if self.wait(process) != 0:
                raise RuntimeError('native-context multimodal interface smoke failed')
            result = json.loads((smoke / 'full_model_smoke.json').read_text())
            if result.get('ok') is not True:
                raise RuntimeError('model smoke did not validate colors and schema')
            self.record('model_smoke_passed', report_path=str(smoke / 'full_model_smoke.json'))
        finally:
            self.cleanup_child('model_smoke', process)

    def run_group(self, group, gpu_index, smoke=False):
        if smoke or gpu_index not in self.execution_contract['hardware']['simulator_gpus']:
            raise ValueError('unauthorized repaired subset simulator work')
        repaired_subset_contract.verify_profile(self.root, self.execution_contract)
        gid = group['group_id']
        group_out = self.out / 'groups' / gid
        group_out.mkdir(parents=True, exist_ok=True)
        attempt_dir = self.runtime / 'scored' / f'worker{gpu_index}' / gid
        attempt_dir.mkdir(parents=True, exist_ok=False)
        relay = full.create_private_relay(self.job, gid)
        full.write_json(attempt_dir / 'relay_location.json', {'live_node_local_path': str(relay),
            'host': socket.gethostname(), 'archive_path': str(attempt_dir / 'relay')})
        sim_env = full.restore_simulator_proxies(self.env, os.environ)
        sim_env.update(CUDA_VISIBLE_DEVICES=self.gpus[gpu_index]['uuid'],
            ROBODOJO_STATE_DIR=str(self.root / 'full_runtime' / f'worker{gpu_index}'),
            SINGULARITY_BINDPATH=str(relay) + ':/aspire-relay',
            SINGULARITYENV_ASPIRE_SIM_LOG=str(Path('/work') / (attempt_dir / 'simulator.log').relative_to(self.simroot)),
            SINGULARITYENV_ASPIRE_PERCEPTION_CONFIG=str(Path('/work') / (self.runtime / 'providers.json').relative_to(self.simroot)))
        controller = sim = None
        started = time.monotonic()
        try:
            config = {'usage_jsonl': str(attempt_dir / 'usage.jsonl'),
                'episode_ids': group['episode_ids'],
                'episode_seeds': {eid: group['eval_seed'] for eid in group['episode_ids']},
                'base_url': self.url, 'expected_context_length': full.CONTEXT, 'socket_idle_timeout_s': None}
            full.write_json(attempt_dir / 'controller_config.json', config)
            with (attempt_dir / 'controller.log').open('w') as handle:
                controller = self.track(gid + '_controller', subprocess.Popen([
                    '/usr/bin/python3.12', '-B', str(self.code / 'full_model_driver.py'),
                    '--config', str(attempt_dir / 'controller_config.json'), '--relay-dir', str(relay)],
                    env=self.env, cwd=self.source_profile, stdout=handle,
                    stderr=subprocess.STDOUT, start_new_session=True))
            deadline = time.monotonic() + 60
            while not (relay / 'ready.json').exists():
                if controller.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('repaired subset controller failed readiness')
                time.sleep(.2)
            inside = Path('/work/aspire_validate') / self.execution_contract['source_profile']['relative_root'] / 'qwen_full/repaired_subset_inside.sh'
            command = ['bash', str(self.simroot / 'scripts/container_exec.sh'), '/bin/bash',
                str(inside), group['task'], self.campaign, str(group['eval_seed']),
                str(self.gpus[gpu_index]['index']), self.gpus[gpu_index]['uuid'],
                str(group['target_episodes']), ','.join(map(str, group['candidate_layout_ids']))]
            self.record('group_started', group_id=gid, gpu=gpu_index, smoke=False,
                        target_episodes=group['target_episodes'], source_profile=str(self.source_profile))
            with (attempt_dir / 'simulator.log').open('w') as handle:
                sim = self.track(gid + '_sim', subprocess.Popen(command, env=sim_env,
                    cwd=self.simroot, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
            rc = self.wait(sim, required_processes=(('inference controller', controller), *self.service_processes))
            if rc != 0:
                raise RuntimeError(f'{gid} exited {rc}; native checkpoint retained')
            result = full.validate_group(group_out / 'group_report.json', group, self.simroot,
                                         expected_campaign=self.campaign)
            with self.lock:
                self.completed[gid] = result
            self.save_progress()
            self.record('group_finished', group_id=gid, gpu=gpu_index,
                        wall_seconds=time.monotonic() - started, result=result)
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
            result = pending_group(self.out, group, self.simroot)
            if result is None:
                pending.put(group)
            else:
                self.completed[group['group_id']] = result
        return pending

    def run(self):
        pending = self.collect_pending_groups()
        self.save_progress()
        if pending.empty():
            self.report['complete'] = True
            return
        self.start_server()  # Existing non-scored three-image model interface check only.
        if self.execution_contract['providers']['sam3']:
            proxy = SimpleNamespace(root=self.root, runtime=self.runtime, source_profile=self.source_profile,
                manifest={'profiles': {'perception': self.execution_contract['source_profile']},
                          'perception_artifacts': self.execution_contract['perception_artifacts']},
                env=self.env, gpus=self.gpus, stop=self.stop, report=self.report,
                track=self.track, record=self.record)
            self.providers, self.service_processes = perception_services.start_services(proxy)
        def worker(index):
            while not self.stop.is_set():
                try:
                    group = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    self.run_group(group, index)
                except BaseException as exc:
                    failure = {'group_id': group['group_id'], 'type': type(exc).__name__, 'message': str(exc)}
                    with self.lock:
                        self.failed.append(failure)
                    self.stop.set()
                    self.record('worker_interrupted', **failure)
                    return
        indices = self.execution_contract['hardware']['simulator_gpus']
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices)) as pool:
            futures = [pool.submit(worker, index) for index in indices]
            for future in futures:
                future.result()
        self.report.update(failures=self.failed, complete=len(self.completed) == self.manifest['group_count'])
        if not self.report['complete']:
            raise RuntimeError('repaired subset incomplete; inspect native checkpoint before resume')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=full.DEFAULT_ROOT)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--mode', choices=('full', 'resume'), default='full')
    args = parser.parse_args()
    os.umask(0o077)
    campaign = RepairedSubsetCampaign(args)
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
