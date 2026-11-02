"""Resumable official-size RoboDojo campaign, two Isaac workers and one TP2 Qwen.

Infrastructure interruption never becomes a policy failure. Native task×seed
checkpoints own scored layout accounting. No episode/model-turn walltime limit.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time

SHARED = Path(__file__).resolve().parents[1] / 'qwen_integration'
sys.path.insert(0, str(SHARED))
from qwen_server_suite import (MODEL, DEFAULT_ROOT, allocated_devices, resolve_gpu_uuids,
    sanitized_environment, restore_simulator_proxies, simulator_proxy_names,
    unused_loopback_port, wait_ready, stop_process_group, utc_now, write_json)

CONTEXT = 262144


def create_private_relay(job, gid):
    # Lighthouse Turbo NFS forces project-group rwx even after chmod(0700).
    # Keep the live control channel on node-local storage, then archive it with
    # the experiment after both endpoint processes have stopped.
    parent = Path(os.environ.get('SLURM_TMPDIR') or '/tmp')
    relay = Path(tempfile.mkdtemp(prefix=f'robodojo-{job}-{gid}-', dir=parent))
    info = relay.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        relay.rmdir()
        raise RuntimeError('node-local relay directory must be private and user-owned')
    return relay


def inference_variant(mtp_tokens):
    if type(mtp_tokens) is not int or mtp_tokens not in (0, 1, 3):
        raise ValueError('mtp_tokens must be exactly 0, 1, or 3')
    return 'bf16' if mtp_tokens == 0 else f'bf16_mtp{mtp_tokens}'


def server_command(python, snapshot, port, mtp_tokens=0):
    # Image count cannot be the first bound: each image uses at least one token.
    # Skip synthetic MM profiling because this deliberately generous item count
    # must never be materialized as a dummy batch. 15% VRAM remains for encoder.
    inference_variant(mtp_tokens)
    command = [str(python), '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(snapshot), '--served-model-name', MODEL,
        '--dtype', 'bfloat16', '--tensor-parallel-size', '2',
        '--max-num-seqs', '2', '--max-model-len', str(CONTEXT),
        '--max-num-batched-tokens', '8192', '--enable-chunked-prefill',
        '--limit-mm-per-prompt', json.dumps({'image': CONTEXT, 'video': 0}),
        '--skip-mm-profiling', '--gpu-memory-utilization', '0.85',
        '--reasoning-parser', 'qwen3', '--host', '127.0.0.1', '--port', str(port)]
    if mtp_tokens:
        command += ['--speculative-config', json.dumps({
            'method': 'mtp', 'num_speculative_tokens': mtp_tokens})]
    return command


def verify_inference_variant(log_path, mtp_tokens):
    expected = inference_variant(mtp_tokens)
    engine = None
    with Path(log_path).open() as handle:
        for line in handle:
            if 'Initializing a V1 LLM engine' in line:
                engine = line.strip()
    if engine is None or 'dtype=torch.bfloat16' not in engine or 'quantization=None' not in engine:
        raise RuntimeError('missing unquantized BF16 engine startup evidence')
    match = re.search(r'speculative_config=(.*?), tokenizer=', engine)
    if match is None:
        raise RuntimeError('missing engine speculative configuration')
    spec = match.group(1)
    if mtp_tokens == 0:
        valid = spec == 'None'
    else:
        valid = (re.search(r"method\s*=\s*['\"]mtp['\"]", spec)
                 and re.search(r'num_spec(?:ulative)?_tokens\s*=\s*' + str(mtp_tokens) + r'\b', spec))
    if not valid:
        raise RuntimeError('actual engine speculative configuration differs from requested variant')
    return {'variant': expected, 'engine_config': engine, 'startup_log': str(log_path),
            'startup_log_sha256_at_verification': sha_file(log_path)}


def allocation_inference_phase(report, report_path):
    """Index evidence without assigning a new allocation's variant to old work."""
    actual = report.get('actual_inference_variant')
    has_actual_field = 'actual_inference_variant' in report
    variant, evidence = actual, ('verified_engine_startup' if actual else
        'engine_not_verified' if has_actual_field else 'server_not_recorded')
    command = report.get('server_command')
    if not has_actual_field and isinstance(command, list):
        # Historical reports predate explicit variant fields. Retain their
        # own command as weaker evidence, without claiming engine verification.
        try:
            if command[command.index('--dtype')+1] == 'bfloat16':
                if '--speculative-config' not in command:
                    variant = 'bf16'
                else:
                    spec = json.loads(command[command.index('--speculative-config')+1])
                    if spec.get('method') == 'mtp':
                        variant = inference_variant(spec['num_speculative_tokens'])
                evidence = 'historical_server_command' if variant else 'unrecognized_server_command'
        except (ValueError, IndexError, KeyError, TypeError):
            variant, evidence = None, 'unrecognized_server_command'
    return {'allocation': report['allocation'], 'mode': report.get('mode'),
            'variant': variant, 'variant_evidence': evidence,
            'requested_variant': report.get('requested_inference_variant'),
            'allocation_report': str(report_path), 'started_utc': report.get('started_utc'),
            'finished_utc': report.get('finished_utc')}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path_host(path, simroot):
    value = Path(path)
    if value.is_relative_to('/work'):
        return simroot / value.relative_to('/work')
    if not value.is_absolute():
        return simroot / 'RoboDojo' / value
    return value


def validate_group(report_path, group, simroot, expected_campaign=None):
    import math
    import re

    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    expected = group['target_episodes']
    campaign = expected_campaign or report_path.parents[2].name
    expected_run_id = f"{campaign}_{group['group_id']}"
    if (report.get('mode') != 'evaluation' or report.get('task') != group['task']
            or type(report.get('eval_seed')) is not int or report['eval_seed'] != group['eval_seed']
            or report.get('native_run_id') != expected_run_id):
        raise RuntimeError('native report task, seed, mode, or campaign identity mismatch')
    if (report.get('complete') is not True or report.get('target') != expected
            or report.get('scored') != expected):
        raise RuntimeError(f"incomplete native group {group['group_id']}")
    result_path = path_host(report['native_result_path'], simroot)
    if result_path.name != '_result.json' or result_path.parent.name != expected_run_id:
        raise RuntimeError('native result path does not belong to the expected campaign group')
    native = json.loads(result_path.read_text())
    detail_mapping = native['details']
    if (not isinstance(detail_mapping, dict)
            or set(detail_mapping) != {str(index) for index in range(expected)}):
        raise RuntimeError('native scored episode indices are incomplete or noncanonical')
    details = list(detail_mapping.values())
    ids = [row['layout_id'] for row in details]
    if (len(details) != expected or native['eval_time'] != expected
            or len(set(ids)) != expected
            or any(type(i) is not int or i not in group['candidate_layout_ids'] for i in ids)):
        raise RuntimeError('native scored count, layout uniqueness, or candidate membership mismatch')
    if any(type(d['success']) is not bool or type(d['score']) not in (int, float)
           or not math.isfinite(d['score']) or not 0 <= d['score'] <= 1 for d in details):
        raise RuntimeError('invalid native scoring detail')
    # Native episode indices, rather than layout IDs, name the camera videos.
    # Count alone can accept duplicate head views while wrist evidence is absent.
    videos = sorted(result_path.parent.glob('*.mp4'))
    pattern = re.compile(r'episode_([0-9]{7})_cam_(head|left_wrist|right_wrist)_(success|fail)\.mp4')
    expected_views = {(index, camera) for index in range(expected)
                      for camera in ('head', 'left_wrist', 'right_wrist')}
    seen = set()
    for path in videos:
        match = pattern.fullmatch(path.name)
        if match is None or path.stat().st_size == 0:
            raise RuntimeError('invalid or empty native camera video')
        index, camera, outcome = int(match[1]), match[2], match[3]
        view = (index, camera)
        if view not in expected_views or view in seen:
            raise RuntimeError('duplicate or unexpected native camera video')
        if outcome != ('success' if detail_mapping[str(index)]['success'] else 'fail'):
            raise RuntimeError('native camera video outcome disagrees with score')
        seen.add(view)
    if seen != expected_views:
        raise RuntimeError(f'native three-camera video evidence incomplete: {len(videos)} vs {expected*3}')
    return {'group_id': group['group_id'], 'task': group['task'],
            'eval_seed': group['eval_seed'], 'target': expected, 'scored': len(details),
            'success': sum(d['success'] for d in details),
            'score_sum': sum(d['score'] for d in details),
            'native_result_path': str(result_path), 'native_result_sha256': sha_file(result_path),
            'group_report_sha256': sha_file(report_path), 'video_count': len(videos)}


class Campaign:
    def __init__(self, args):
        self.mtp_tokens = getattr(args, 'mtp_tokens', 0)
        requested_variant = inference_variant(self.mtp_tokens)
        self.root = args.root.resolve(strict=True)
        self.simroot = self.root.parent
        self.code = self.root / 'qwen_full'
        self.campaign = args.campaign
        self.out = self.root / 'full_results' / self.campaign
        self.out.mkdir(parents=True, exist_ok=True)
        self.job = os.environ.get('SLURM_JOB_ID', 'local')
        self.runtime = self.out / 'allocations' / self.job
        self.runtime.mkdir(parents=True, exist_ok=False)
        self.manifest_path = args.manifest.resolve(strict=True)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.env = sanitized_environment(os.environ, self.root)
        self.python = self.root / 'qwen/.venv-runtime/bin/python'
        self.env['PATH'] = str(self.python.parent) + os.pathsep + self.env.get('PATH', '')
        self.devices = allocated_devices(os.environ.get('CUDA_VISIBLE_DEVICES'))
        self.server = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.children = {}
        self.failed = []
        self.completed = {}
        self.mode = args.mode
        self.report = {'campaign': self.campaign, 'allocation': self.job,
            'started_utc': utc_now(), 'complete': False, 'host': socket.gethostname(),
            'mode': self.mode, 'api_cost_usd': 0, 'context_length': CONTEXT,
            'requested_inference_variant': requested_variant, 'mtp_tokens': self.mtp_tokens,
            'actual_inference_variant': None,
            'model_turn_limit': None, 'output_token_limit': 'remaining native context only',
            'episode_walltime_limit': None, 'source_sha256': {
                p.name: sha_file(p) for p in self.code.iterdir()
                if p.is_file() and p.suffix in {'.py', '.sh', '.sbatch'}},
            'manifest_sha256': sha_file(self.manifest_path),
            'simulator_proxy_variable_names': simulator_proxy_names(os.environ)}

    def record(self, event, **values):
        row = {'utc': utc_now(), 'event': event, **values}
        with self.lock:
            with (self.runtime / 'events.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n'); f.flush(); os.fsync(f.fileno())
            print(json.dumps(row), flush=True)

    def track(self, name, process):
        with self.lock:
            self.children[name] = process
        return process

    def cleanup_child(self, name, process, grace=30):
        detail = stop_process_group(process, grace=grace)
        with self.lock:
            self.children.pop(name, None)
        self.record('process_cleaned', name=name, **detail)

    def prepare(self):
        campaign_manifest = self.out / 'manifest.json'
        if campaign_manifest.exists():
            if json.loads(campaign_manifest.read_text()) != self.manifest:
                raise RuntimeError('campaign manifest changed; never merge unlike protocols')
        else:
            write_json(campaign_manifest, self.manifest)
            # Compare normalized serialized content, since original formatting can vary.
        self.report['manifest_sha256'] = sha_file(campaign_manifest)
        pinned = json.loads((self.root / 'qwen/model_manifest.json').read_text())
        if pinned['repo'] != MODEL:
            raise RuntimeError('unexpected model snapshot')
        self.snapshot = Path(pinned['snapshot_path']).resolve(strict=True)
        cfg = json.loads((self.snapshot / 'config.json').read_text())
        if cfg['text_config']['max_position_embeddings'] != CONTEXT:
            raise RuntimeError('native model context changed')
        self.report['model_manifest'] = pinned
        write_json(self.out / 'experiment_config.json', {
            'campaign': self.campaign, 'model': MODEL, 'model_revision': pinned['revision'],
            'protocol': 'official RoboDojo simulation task/layout/scoring protocol',
            'task_families': 42, 'task_configurations': 54, 'eval_layout_seeds': [0,1,2],
            'target_scored_episodes': 6300, 'invalid_layouts': 'native skip and replacement',
            'harness': 'ASPIRE public API/program executor with custom ARX X5 bridge and native CuRobo',
            'sam3': False, 'contact_graspnet': False, 'offline_skill_search': False,
            'observations': 'public three-camera RGB, metric depth API, calibration, proprioception and task language',
            'history': 'all public observations and final assistant messages; no rolling truncation',
            'model_turn_cap': None, 'independent_output_token_cap': None,
            'native_context_tokens': CONTEXT, 'native_task_action_budgets': True,
            'evaluation_seed_note': 'each seed selects a distinct layout set; model sampler uses matching seed',
            'gpu_roles': {'model': 'two L40S, BF16 tensor parallel', 'simulation': 'two independent one-GPU workers'},
            'inference_phase_policy': 'Inference settings apply per allocation to newly generated outputs; earlier scored episodes retain their original allocation provenance and are not relabeled or rerun.',
            'inference_phase_attribution': 'Use allocation request ledgers and native checkpoints; allocation reports record requested and verified engine variants.',
            'api_cost_usd': 0, 'leaderboard_submission': False})
        gpu_text = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv'], text=True)
        (self.runtime / 'gpus.csv').write_text(gpu_text)
        self.gpus = resolve_gpu_uuids(gpu_text, self.devices)
        self.report['gpus'] = self.gpus
        for index in (0, 3):
            state = self.root / 'full_runtime' / f'worker{index}'
            for child in ('runtime-home', 'cache/kit', 'cache/main', 'cache/computecache',
                          'logs', 'config', 'data/documents', 'data/kit-user', 'pkg'):
                (state / child).mkdir(parents=True, exist_ok=True)
        self.port = unused_loopback_port()
        self.url = f'http://127.0.0.1:{self.port}/v1'
        self.env['ASPIRE_LOCAL_BASE_URL'] = self.url
        write_json(self.runtime / 'allocation.json', self.report)
        self.update_inference_phases()

    def update_inference_phases(self):
        config_path = self.out / 'experiment_config.json'
        if not config_path.exists():
            return
        experiment = json.loads(config_path.read_text())
        phases = []
        for path in sorted((self.out / 'allocations').glob('*/allocation.json')):
            report = json.loads(path.read_text())
            phases.append(allocation_inference_phase(report, path))
        experiment['inference_phases'] = phases
        write_json(config_path, experiment)

    def start_server(self):
        command = server_command(self.python, self.snapshot, self.port, self.mtp_tokens)
        self.report['server_command'] = command
        env = dict(self.env, CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in self.gpus[1:3]))
        with (self.runtime / 'server.log').open('w') as f:
            self.server = self.track('server', subprocess.Popen(command, env=env,
                cwd=self.root, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
        started = time.monotonic()
        self.record('server_started', pid=self.server.pid)
        self.report['advertised_models'] = wait_ready(self.server, self.url,
            started + 2400, self.record)
        verified = verify_inference_variant(self.runtime / 'server.log', self.mtp_tokens)
        self.report['inference_variant_evidence'] = verified
        self.report['actual_inference_variant'] = verified['variant']
        self.report['server_startup_seconds'] = time.monotonic() - started
        write_json(self.runtime / 'allocation.json', self.report)
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

    def wait(self, process, required_processes=()):
        while process.poll() is None:
            if self.stop.is_set():
                raise InterruptedError('campaign allocation stopping; current episode remains unscored')
            if self.server is not None and self.server.poll() is not None:
                raise RuntimeError('inference server exited; current episode remains unscored')
            for name, required in required_processes:
                if required.poll() is not None:
                    raise RuntimeError(f'{name} exited; current episode remains unscored')
            time.sleep(1)
        return process.returncode

    def run_group(self, group, gpu_index, smoke=False):
        gid = group['group_id']
        campaign = self.campaign + f'_smoke{gpu_index}_{self.job}' if smoke else self.campaign
        group_out = self.root / 'full_results' / campaign / 'groups' / gid
        group_out.mkdir(parents=True, exist_ok=True)
        attempt_dir = self.runtime / ('smoke' if smoke else 'scored') / f'worker{gpu_index}' / gid
        attempt_dir.mkdir(parents=True, exist_ok=True)
        # Every allocation keeps its request/response ledger, even for interrupted work.
        relay = create_private_relay(self.job, gid)
        write_json(attempt_dir / 'relay_location.json', {
            'live_node_local_path': str(relay), 'host': socket.gethostname(),
            'archive_path': str(attempt_dir / 'relay')})
        controller = sim = None
        sim_env = restore_simulator_proxies(self.env, os.environ)
        sim_env.update(CUDA_VISIBLE_DEVICES=self.gpus[gpu_index]['uuid'],
            ROBODOJO_STATE_DIR=str(self.root / 'full_runtime' / f'worker{gpu_index}'),
            SINGULARITY_BINDPATH=str(relay) + ':/aspire-relay',
            SINGULARITYENV_ASPIRE_SIM_LOG=str(Path('/work') /
                (attempt_dir / 'simulator.log').relative_to(self.simroot)))
        if smoke:
            sim_env['SINGULARITYENV_ASPIRE_FULL_MODE'] = 'interface_smoke'
        started = time.monotonic()
        try:
            if not smoke:
                config = {'usage_jsonl': str(attempt_dir / 'usage.jsonl'),
                    'episode_ids': group['episode_ids'],
                    'episode_seeds': {eid: group['eval_seed'] for eid in group['episode_ids']},
                    'base_url': self.url, 'expected_context_length': CONTEXT,
                    'socket_idle_timeout_s': None}
                write_json(attempt_dir / 'controller_config.json', config)
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
                '/work/aspire_validate/qwen_full/full_inside.sh', group['task'], campaign,
                str(group['eval_seed']), str(self.gpus[gpu_index]['index']), self.gpus[gpu_index]['uuid']]
            self.record('group_started', group_id=gid, gpu=gpu_index, smoke=smoke)
            with (attempt_dir / 'simulator.log').open('w') as f:
                sim = self.track(gid + '_sim', subprocess.Popen(command, env=sim_env,
                    cwd=self.simroot, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
            rc = self.wait(sim, required_processes=(('inference controller', controller),)
                           if controller is not None else ())
            if rc != 0:
                raise RuntimeError(f'simulator group {gid} exited {rc}; checkpoint preserved')
            if smoke:
                result = json.loads((group_out / 'group_report.json').read_text())
                if result.get('complete') is not True:
                    raise RuntimeError('interface smoke incomplete')
            else:
                result = validate_group(group_out / 'group_report.json', group, self.simroot)
                with self.lock:
                    self.completed[gid] = result
                self.save_progress()
            self.record('group_finished', group_id=gid, gpu=gpu_index, smoke=smoke,
                wall_seconds=time.monotonic()-started, result=result)
        finally:
            if sim is not None:
                self.cleanup_child(gid + '_sim', sim, grace=45)
            if controller is not None:
                self.cleanup_child(gid + '_controller', controller)
            shutil.copytree(relay, attempt_dir / 'relay')
            self.record('relay_archived', group_id=gid, path=str(attempt_dir / 'relay'))
            shutil.rmtree(relay)

    def save_progress(self):
        with self.lock:
            groups = sorted(self.completed.values(), key=lambda d: d['group_id'])
            write_json(self.out / 'progress.json', {'campaign': self.campaign,
                'updated_utc': utc_now(), 'complete': len(groups) == self.manifest['group_count'],
                'groups_completed': len(groups), 'groups_target': self.manifest['group_count'],
                'scored': sum(g['scored'] for g in groups),
                'target': self.manifest['total_target_episodes'],
                'success': sum(g['success'] for g in groups), 'groups': groups})

    def run(self):
        # Acceptance checks are sequential, separate from scored campaign data.
        smoke_group = next(g for g in self.manifest['groups']
                           if g['task']=='general_pickup' and g['eval_seed']==0)
        if self.mode in ('preflight', 'full'):
            for gpu_index in (0, 3):
                self.run_group(smoke_group, gpu_index, smoke=True)
        if self.mode == 'preflight':
            self.report['complete'] = True
            return
        self.start_server()
        pending = queue.Queue()
        for group in self.manifest['groups']:
            report = self.out / 'groups' / group['group_id'] / 'group_report.json'
            prior = json.loads(report.read_text()) if report.exists() else None
            if prior and prior.get('complete'):
                self.completed[group['group_id']] = validate_group(report, group, self.simroot)
                continue
            if prior and prior.get('scored', 0) > 0:
                native_result = path_host(prior['native_result_path'], self.simroot)
                checkpoint = native_result.parent.parent / ('_resume_' + prior['native_run_id'] + '.json')
                if not checkpoint.exists():
                    raise RuntimeError(f"refusing to replay scored layouts without checkpoint: {group['group_id']}")
                if prior.get('scores_complete') and not prior.get('videos_validated'):
                    raise RuntimeError(f"scoring is complete but video evidence needs repair: {group['group_id']}")
            pending.put(group)
        self.save_progress()
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
            futures = [pool.submit(worker, index) for index in (0,3)]
            for f in futures:
                f.result()
        self.report['failures'] = self.failed
        self.report['complete'] = len(self.completed) == self.manifest['group_count']
        if not self.report['complete']:
            raise RuntimeError('campaign allocation incomplete; resume native checkpoints after inspection')

    def finish(self):
        self.stop.set()
        for name, process in list(self.children.items()):
            try:
                self.cleanup_child(name, process)
            except Exception as exc:
                self.record('cleanup_error', name=name, error_type=type(exc).__name__)
        self.report['finished_utc'] = utc_now()
        write_json(self.runtime / 'allocation.json', self.report)
        self.update_inference_phases()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--mode', choices=['preflight','full','resume'], default='full')
    parser.add_argument('--mtp-tokens', type=int, choices=[0, 1, 3], default=0,
                        help='Opt-in MTP speculative depth; 0 retains baseline BF16 decoding')
    args = parser.parse_args()
    if not args.campaign.replace('_','').isalnum():
        raise ValueError('invalid campaign ID')
    os.umask(0o077)
    campaign = Campaign(args)
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
            campaign.report['error'] = {'type':type(exc).__name__, 'message':str(exc)}
            campaign.record('allocation_incomplete', **campaign.report['error'])
        finally:
            campaign.finish()
        return rc


if __name__ == '__main__':
    sys.exit(main())
