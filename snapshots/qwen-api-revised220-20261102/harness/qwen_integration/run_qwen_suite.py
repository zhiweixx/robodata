"""Exactly three local Qwen episodes; same ASPIRE observations and action budgets."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

TASKS = ('general_pickup', 'stack_bowls', 'align_blocks')
LIMITS = dict(zip(TASKS, (200, 800, 200)))
MODEL = 'Qwen/Qwen3.8-27B'


def clean_environment(source):
    result = dict(source)
    for name in list(result):
        if any(word in name.upper() for word in ('TOKEN', 'API_KEY', 'KEY_FILE')):
            result.pop(name)
    return result


def terminate(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def validate(report, task):
    if not (report.get('ok') is True and report.get('episode_id') == task+'_s0_l0'
            and report.get('task') == task and report.get('seed') == 0
            and report.get('layout_id') == 0 and report.get('model') == MODEL):
        raise RuntimeError('incomplete or mismatched episode report')
    iterations = report.get('iterations')
    if not isinstance(iterations, list) or not 1 <= len(iterations) <= 8:
        raise RuntimeError('invalid model call count')
    terminal_field = "give_up" if report.get("proposal_schema") == "aspire_program_give_up_v2" else "done"
    for index, entry in enumerate(iterations, 1):
        if entry.get('iteration') != index or type(entry.get(terminal_field)) is not bool:
            raise RuntimeError('invalid iteration record')
        if entry[terminal_field] and index != len(iterations):
            raise RuntimeError('model continued after give_up')
    if report.get('total_programs_executed') != sum(not r[terminal_field] for r in iterations):
        raise RuntimeError('program count mismatch')
    official = report['official']
    if (official['step_limit'] != LIMITS[task]
            or type(official['environment_steps']) is not int
            or not 0 <= official['environment_steps'] <= LIMITS[task]
            or type(official['official_success']) is not bool):
        raise RuntimeError('native action budget/scorer mismatch')
    return official


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum():
        raise ValueError('invalid run ID')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    integration = Path(__file__).resolve().parent
    root, simroot = integration.parent, integration.parent.parent
    output = root/'results'/args.run_id
    output.mkdir(parents=True, exist_ok=False)
    private = Path('/home/ROBODATA_USER/.config/qwen-pilot')
    private.mkdir(mode=0o700, parents=True, exist_ok=True)
    relay = private/('relay-'+args.run_id)
    relay.mkdir(mode=0o700, exist_ok=False)
    config = {'usage_jsonl': str(output/'usage.jsonl'),
        'episode_ids': [t+'_s0_l0' for t in TASKS], 'model': MODEL,
        'base_url': os.environ['ASPIRE_LOCAL_BASE_URL'], 'max_iterations': 8,
        'max_input_tokens': 128000, 'max_output_tokens': 8192, 'request_timeout_s': 400}
    (output/'controller_config.json').write_text(json.dumps(config, indent=2))
    snapshot = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in integration.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.sbatch')}
    (output/'source_snapshot.json').write_text(json.dumps(snapshot, indent=2))
    (output/'experiment_config.json').write_text(json.dumps({
        'model': MODEL, 'tasks': list(TASKS), 'seed': 0, 'layout': 0,
        'native_action_budgets': LIMITS, 'max_model_calls_per_episode': 8,
        'sam3': False, 'contact_graspnet': False, 'history': 'all public messages and images; final JSON assistants',
        'simulator_gpu': os.environ['ASPIRE_SIM_GPU'], 'simulator_gpu_uuid': os.environ['ASPIRE_SIM_GPU_UUID'],
        'simulator_cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'gpu_visibility_note': 'Preserve Astra full allocation visibility for Isaac; device_id=0 and renderer/activeGpu=0, verified by UUID.',
        'process_timeout_s': 3600,
        'astra_process_timeout_s': 1200, 'request_timeout_s': 400, 'astra_request_timeout_s': 180,
        'timeout_note': 'Local inference receives longer wall-clock allowance; same action/turn/token budgets.',
        'astra_general_pickup_comparable': False,
        'model_manifest': json.loads((root/'qwen/model_manifest.json').read_text())}, indent=2))
    env = clean_environment(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    attempts, complete, final_budget, controller, sim = [], False, None, None, None
    began = time.monotonic()

    def record(event, **fields):
        row = {'event':event, 'utc':dt.datetime.now(dt.timezone.utc).isoformat(), **fields}
        with (output/'attempts.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
        print(json.dumps(row), flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (output/'controller.log').open('w') as log:
            controller = subprocess.Popen([sys.executable, str(integration/'local_model_driver.py'),
                '--config', str(output/'controller_config.json'), '--relay-dir', str(relay)],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic()+60
        while not (relay/'ready.json').exists():
            if controller.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('local controller not ready; no episode started')
            time.sleep(.2)
        record('controller_ready', api_cost_usd=0)
        for task in TASKS:
            if controller.poll() is not None or (relay/'stopped.json').exists():
                raise RuntimeError('controller stopped; no automatic retry')
            taskout = output/task
            taskout.mkdir()
            simenv = clean_environment(env)
            simenv['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
            simenv['SINGULARITY_BINDPATH'] = str(relay)+':/aspire-relay'
            record('episode_process_start', task=task, seed=0, layout_id=0)
            started = time.monotonic()
            with (taskout/'simulator.log').open('w') as log:
                sim = subprocess.Popen(['bash', str(simroot/'scripts/container_exec.sh'), '/bin/bash',
                    '/work/aspire_validate/qwen_integration/model_inside.sh', task,args.run_id,task+'_s0_l0',
                    os.environ['ASPIRE_SIM_GPU_UUID']],
                    cwd=simroot, env=simenv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                rc = sim.wait(timeout=3600)
            sim = None
            attempt = {'task':task,'exit_code':rc,'wall_seconds':time.monotonic()-started}
            attempts.append(attempt)
            record('episode_process_end', **attempt)
            if rc:
                raise RuntimeError('simulator process failed; preserve evidence without retry')
            report = json.loads((taskout/'episode_report.json').read_text())
            official = validate(report, task)
            attempt.update(validated_episode=True, model_calls=len(report['iterations']),
                programs_executed=report['total_programs_executed'], **official)
            record('episode_validated', **attempt)
        from model_relay import RelayModelClient
        final_budget = RelayModelClient(relay).stop()['budget']
        if controller.wait(timeout=10) != 0 or final_budget['aborted']:
            raise RuntimeError('controller shutdown was not clean')
        if any(final_budget['calls'][t+'_s0_l0'] != a['model_calls'] for t,a in zip(TASKS,attempts)):
            raise RuntimeError('model call ledger/report mismatch')
        complete = True
        record('three_episodes_complete')
    finally:
        terminate(sim)
        terminate(controller)
        shutil.copytree(relay, output/'relay_audit')
        shutil.rmtree(relay)
        (output/'launcher_summary.json').write_text(json.dumps({
            'complete':complete,'attempts':attempts,'wall_seconds':time.monotonic()-began,
            'budget':final_budget,'api_cost_usd':0,'no_automatic_retry':True},indent=2))


if __name__ == '__main__':
    main()
