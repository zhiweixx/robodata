"""Run isolated BF16/MTP request replays only inside a four-L40S allocation.

This owns temporary servers, not simulators or scored episodes. The campaign
lock prevents the scored evaluation from restarting during the comparison.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from mtp_replay_benchmark import benchmark, server_command, write_new
from full_model_smoke import smoke
from qwen_server_suite import (allocated_devices, resolve_gpu_uuids,
    sanitized_environment, unused_loopback_port, wait_ready, stop_process_group,
    utc_now, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--trial', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Inference trials require a Slurm allocation')
    if not args.trial.replace('_', '').isalnum():
        raise ValueError('invalid trial ID')
    root = args.root.resolve(strict=True)
    out = root / 'mtp_trials' / args.trial / os.environ['SLURM_JOB_ID']
    out.mkdir(parents=True, exist_ok=False)
    lock = (root / 'full_results' / args.campaign / 'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    devices = allocated_devices(os.environ.get('CUDA_VISIBLE_DEVICES'))
    gpu_csv = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv'], text=True)
    gpus = resolve_gpu_uuids(gpu_csv, devices)
    gpu_ids = [g['uuid'] for g in gpus[1:3]]
    (out / 'gpus.csv').write_text(gpu_csv)
    model = json.loads((root / 'qwen/model_manifest.json').read_text())
    snapshot = Path(model['snapshot_path']).resolve(strict=True)
    env = sanitized_environment(os.environ, root)
    for name in ('CUDA_HOME', 'TRITON_PTXAS_PATH'):
        env.pop(name, None)
    python = root / 'qwen/.venv-runtime/bin/python'
    env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
    env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
    cases = [args.cases / f'{name}.json' for name in ('short', 'medium', 'long')]
    for path in cases:
        if not path.is_file():
            raise FileNotFoundError(path)
    report = {'purpose': 'isolated_performance_test_not_robot_evaluation',
              'started_utc': utc_now(), 'job_id': os.environ['SLURM_JOB_ID'],
              'campaign_protected': args.campaign, 'model_manifest': model,
              'gpus': gpus, 'model_gpu_uuids': gpu_ids, 'variants': {},
              'native_context': 262144, 'target_dtype': 'bfloat16',
              'server_runtime_versions': json.loads(subprocess.check_output([
                  str(python), '-c', 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ("vllm","torch","transformers")}))'
              ], env=env, text=True)),
              'source_sha256': {name: hashlib.sha256((root / 'qwen_full' / name).read_bytes()).hexdigest()
                  for name in ('mtp_trial_supervisor.py', 'mtp_replay_benchmark.py', 'full_supervisor.py')},
              'test_only_output_cap': 1024, 'changes_to_evaluation_budgets': False,
              'simulation_running': False, 'api_cost_usd': 0}
    write_json(out / 'trial.json', report)
    server = None
    interrupting = False

    def record(event, **fields):
        row = {'utc': utc_now(), 'event': event, **fields}
        with (out / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n'); handle.flush(); os.fsync(handle.fileno())
        print(json.dumps(row), flush=True)

    def interrupted(signum, frame):
        nonlocal interrupting
        if interrupting:
            return
        interrupting = True
        # Unblock streaming threads before their executor waits during unwind.
        # This signal targets only the fresh trial server's owned process group.
        if server is not None:
            stop_process_group(server, grace=8)
        raise InterruptedError(f'trial interrupted by signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, interrupted)
    try:
        for variant in ('bf16', 'mtp1', 'mtp3'):
            variant_out = out / variant
            variant_out.mkdir()
            port = unused_loopback_port()
            base_url = f'http://127.0.0.1:{port}/v1'
            command = server_command(python, snapshot, port, variant)
            result = {'command': command, 'started_utc': utc_now(), 'ok': False}
            report['variants'][variant] = result
            write_json(out / 'trial.json', report)
            server_log = variant_out / 'server.log'
            started = time.monotonic()
            try:
                with server_log.open('x') as handle:
                    server = subprocess.Popen(command, cwd=root, env=env,
                        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                record('variant_starting', variant=variant, pid=server.pid)
                wait_ready(server, base_url, time.monotonic() + 2400, record)
                result['startup_s'] = time.monotonic() - started
                record('variant_ready', variant=variant, startup_s=result['startup_s'])
                result['smoke'] = smoke(variant_out / 'vision_schema_smoke.json', base_url)
                record('vision_schema_passed', variant=variant)
                result['single'] = benchmark(cases, base_url, server_log, variant,
                    variant_out / 'concurrency1', 1, 1024, 2, 1, gpu_ids)
                record('single_request_tests_finished', variant=variant,
                    result=result['single'])
                result['dual'] = benchmark(cases[:2], base_url, server_log, variant,
                    variant_out / 'concurrency2', 2, 1024, 2, 1, gpu_ids)
                result['dual_long'] = benchmark(cases[1:], base_url, server_log, variant,
                    variant_out / 'concurrency2_long', 2, 1024, 2, 1, gpu_ids)
                result['ok'] = True
                record('variant_finished', variant=variant, result=result['dual'])
            except Exception as exc:
                result['error_type'] = type(exc).__name__
                result['error'] = str(exc)
                record('variant_failed', variant=variant, error_type=type(exc).__name__, error=str(exc))
                if isinstance(exc, InterruptedError):
                    raise
            finally:
                if server is not None:
                    result['server_cleanup'] = stop_process_group(server, grace=30)
                    server = None
                result['finished_utc'] = utc_now()
                result['wall_s'] = time.monotonic() - started
                write_json(out / 'trial.json', report)
    finally:
        if server is not None:
            stop_process_group(server, grace=30)
        report['finished_utc'] = utc_now()
        write_json(out / 'trial.json', report)
        lock.close()
    eligible = [name for name in ('mtp1', 'mtp3')
                if report['variants'].get(name, {}).get('ok') is True]
    report['eligible_mtp_variants'] = eligible
    report['comparison_available'] = bool(eligible) and report['variants'].get('bf16', {}).get('ok') is True
    write_json(out / 'trial.json', report)
    record('trial_finished', path=str(out / 'trial.json'),
           comparison_available=report['comparison_available'], eligible_mtp_variants=eligible)
    return 0 if report['comparison_available'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
