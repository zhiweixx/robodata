"""Three authorized episodes; a continuation may run only previously uncalled tasks."""
from __future__ import annotations
import argparse
import datetime as dt
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import subprocess
import time


AUTHORIZED_TASKS = ('general_pickup', 'stack_bowls', 'align_blocks')
GLOBAL_BUDGET = Decimal('50')


def _money(value):
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise RuntimeError('invalid nonnegative budget amount')
    return amount


def lock_prior_ledger(path):
    """Hold the existing lock without modifying its contents or the ledger.

    Linux NFS requires a writable descriptor for an exclusive flock. Opening
    the existing lock with O_RDWR performs no write and never truncates it.
    """
    fd = os.open(str(path) + '.lock', os.O_RDWR | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError('prior budget lock is not a regular file owned by this user')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


def claim_continuation(audit, output, tasks):
    """Consume the remaining-task authorization once, before any API process starts."""
    path = Path(audit['prior_ledger']).parent / 'continuation_claim.json'
    claim = {'schema_version': 1, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'selected_tasks': list(tasks), 'continuation_output': str(Path(output).resolve()),
             'continuation_run_id': Path(output).name,
             'prior_ledger_sha256': audit['prior_ledger_sha256'],
             'prior_charged_usd': audit['prior_charged_usd'],
             'remaining_budget_usd': audit['remaining_budget_usd'],
             'no_automatic_resume': True}
    encoded = (json.dumps(claim, sort_keys=True, indent=2) + '\n').encode()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        raise RuntimeError('remaining-task authorization was already claimed; no new run or automatic resume') from None
    # On interruption the claim deliberately remains, even if partially written.
    # A failed launch must be reviewed rather than silently getting a fresh budget.
    with os.fdopen(fd, 'wb') as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {'path': str(path), 'sha256': hashlib.sha256(encoded).hexdigest()}


def audit_prior_run(path, tasks):
    """Read immutable evidence; reject any attempted task or ambiguous old charge."""
    from reviewed_astra_client import price_usage
    path = Path(path).resolve(strict=True)
    expected_ids = [task + '_s0_l0' for task in AUTHORIZED_TASKS]
    starts, settlements, received, counts_open = {}, {}, {}, set()
    calls = {episode: 0 for episode in expected_ids}
    costs = {episode: Decimal(0) for episode in expected_ids}
    digest = hashlib.sha256()
    config = None
    with path.open('rb') as handle:
        for number, raw in enumerate(handle):
            digest.update(raw)
            if not raw.endswith(b'\n'):
                raise RuntimeError('prior ledger has a partial record')
            row = json.loads(raw)
            event, attempt = row.get('event'), row.get('attempt_id')
            if number == 0:
                config = row.get('config', {})
                if (event != 'suite_initialized' or config.get('model') != 'gpt-6-astra'
                        or config.get('purpose') != 'aspire_robodojo_pilot'
                        or config.get('episode_ids') != expected_ids
                        or _money(config.get('suite_budget_usd')) != GLOBAL_BUDGET
                        or config.get('max_calls_per_episode') != 8):
                    raise RuntimeError('prior ledger is not the original authorized three-task suite')
                continue
            if event in ('suite_aborted', 'request_error', 'token_count_error', 'driver_stopped'):
                raise RuntimeError('prior suite contains an abort or unresolved error; continuation prohibited')
            if event == 'token_count_started':
                if attempt in counts_open:
                    raise RuntimeError('duplicate open count in prior ledger')
                counts_open.add(attempt)
            elif event == 'token_count_received':
                if attempt not in counts_open:
                    raise RuntimeError('orphan token count in prior ledger')
                counts_open.remove(attempt)
            elif event == 'request_started':
                episode = row.get('episode_id')
                if not isinstance(attempt, str) or attempt in starts or episode not in calls:
                    raise RuntimeError('invalid request in prior ledger')
                if _money(row['reserved_usd']) != _money(config['request_reservation_usd']):
                    raise RuntimeError('prior reservation does not match suite configuration')
                starts[attempt] = row
                calls[episode] += 1
            elif event == 'response_received':
                if attempt not in starts or attempt in received:
                    raise RuntimeError('invalid response in prior ledger')
                if row.get('model') != 'gpt-6-astra' or row.get('service_tier') != 'default':
                    raise RuntimeError('prior response used an unexpected model or pricing tier')
                received[attempt] = row
            elif event == 'request_settled':
                if attempt not in starts or attempt not in received or attempt in settlements:
                    raise RuntimeError('invalid settlement in prior ledger')
                episode = starts[attempt]['episode_id']
                amount = _money(row['charged_usd'])
                repriced, _ = price_usage(received[attempt]['usage_raw'])
                if (row.get('episode_id') != episode or amount != repriced
                        or amount > _money(starts[attempt]['reserved_usd'])):
                    raise RuntimeError('prior usage and settlement do not reconcile')
                settlements[attempt] = amount
                costs[episode] += amount
                if (_money(row['episode_charged_usd']) != costs[episode]
                        or _money(row['suite_charged_usd']) != sum(costs.values(), Decimal(0))):
                    raise RuntimeError('prior cumulative accounting is inconsistent')
    if config is None or not starts:
        raise RuntimeError('a continuation requires a nonempty original run')
    if counts_open or set(starts) != set(settlements):
        raise RuntimeError('prior ledger contains unknown billing; no continuation allowed')
    if any(count > 8 for count in calls.values()):
        raise RuntimeError('prior run exceeded an episode model-call cap')
    remaining = tuple(task for task in AUTHORIZED_TASKS if calls[task + '_s0_l0'] == 0)
    if tuple(tasks) != remaining:
        raise RuntimeError('continuation must run exactly the original uncalled tasks, in order')
    total = sum(costs.values(), Decimal(0))
    if total >= GLOBAL_BUDGET:
        raise RuntimeError('original global API budget is exhausted')
    diagnosis_path = path.parent / 'integration_diagnosis.json'
    diagnosis, diagnosis_evidence = {}, None
    if diagnosis_path.exists():
        diagnosis_bytes = diagnosis_path.read_bytes()
        diagnosis = json.loads(diagnosis_bytes)
        diagnosis_evidence = {'path': str(diagnosis_path),
                              'sha256': hashlib.sha256(diagnosis_bytes).hexdigest()}
    reports = []
    for task in AUTHORIZED_TASKS:
        episode = task + '_s0_l0'
        if not calls[episode]:
            continue
        report_path = path.parent / task / 'episode_report.json'
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
        official = validate_episode_report(report, task, episode)
        if len(report['iterations']) != calls[episode]:
            raise RuntimeError('prior measured episode and API ledger call counts differ')
        reports.append({'task': task, 'episode_id': episode, 'validated_episode': True,
            'schema_ledger_validated': True,
            'validation_scope': 'Report schema and ledger consistency only; not capability measurement validity',
            'measurement_valid': diagnosis.get(task + '_measurement_valid'),
            'measurement_diagnosis': diagnosis.get('reason'),
            'model_calls': calls[episode], 'programs_executed': report['total_programs_executed'],
            'official_success': official['official_success'], 'prior_report': str(report_path),
            'prior_report_sha256': hashlib.sha256(report_bytes).hexdigest()})
    return {'prior_ledger': str(path), 'prior_ledger_sha256': digest.hexdigest(),
            'prior_charged_usd': str(total), 'remaining_budget_usd': str(GLOBAL_BUDGET - total),
            'prior_calls': calls, 'prior_episodes': reports, 'remaining_tasks': list(remaining),
            'prior_diagnosis_evidence': diagnosis_evidence, 'unknown_billing': False}


def verify_prior_unchanged(audit):
    if audit is None:
        return
    with Path(audit['prior_ledger']).open('rb') as handle:
        current = hashlib.file_digest(handle, 'sha256').hexdigest()
    if current != audit['prior_ledger_sha256']:
        raise RuntimeError('original ledger changed after continuation audit')
    for item in audit['prior_episodes']:
        if hashlib.sha256(Path(item['prior_report']).read_bytes()).hexdigest() != item['prior_report_sha256']:
            raise RuntimeError('an original episode report changed after continuation audit')
    diagnosis = audit.get('prior_diagnosis_evidence')
    if diagnosis and hashlib.sha256(Path(diagnosis['path']).read_bytes()).hexdigest() != diagnosis['sha256']:
        raise RuntimeError('the original integration diagnosis changed after continuation audit')
    claim = audit.get('continuation_claim')
    if claim and hashlib.sha256(Path(claim['path']).read_bytes()).hexdigest() != claim['sha256']:
        raise RuntimeError('the continuation claim changed after authorization was consumed')


def check_private_key_metadata(private):
    """Check ownership/mode without opening or reading credential contents."""
    info = private.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('credential directory must be owned by this user and mode 700')
    info = (private / 'openai.key').lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('credential file must be private, regular, and owned by this user')


def simulator_environment(source):
    result = dict(source)
    credentials = ('OPENAI_API_KEY', 'OPENAI_API_KEY_FILE', 'HF_TOKEN',
                   'ARK_API_KEY', 'ARK_API_KEY_BACKUP', 'SAFEHARNESS_KEY_FILE')
    for prefix in ('', 'SINGULARITYENV_', 'APPTAINERENV_'):
        for name in credentials:
            result.pop(prefix + name, None)
    return result


def validate_episode_report(report, task, episode_id):
    if not (report.get('ok') is True and report.get('episode_id') == episode_id
            and report.get('task') == task and report.get('seed') == 0
            and report.get('layout_id') == 0 and report.get('model') == 'gpt-6-astra'):
        raise RuntimeError('episode report is missing, incomplete or mismatched')
    iterations = report.get('iterations')
    if not isinstance(iterations, list) or not 1 <= len(iterations) <= 8:
        raise RuntimeError('episode did not record 1..8 model calls')
    terminal_field = "give_up" if report.get("proposal_schema") == "aspire_program_give_up_v2" else "done"
    for index, entry in enumerate(iterations, 1):
        if entry.get('iteration') != index or type(entry.get(terminal_field)) is not bool:
            raise RuntimeError('model iteration records are incomplete or out of sequence')
        if entry[terminal_field] and index != len(iterations):
            raise RuntimeError('model calls continued after give_up')
    if report.get('total_programs_executed') != sum(not item[terminal_field] for item in iterations):
        raise RuntimeError('model call count and executed program count are inconsistent')
    official = report['official']
    steps, limit = official['environment_steps'], official['step_limit']
    if type(steps) is not int or type(limit) is not int or not 0 <= steps <= limit:
        raise RuntimeError('native episode action budget was not respected')
    if type(official['official_success']) is not bool:
        raise RuntimeError('official_success is not an evaluator boolean')
    return official


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--tasks', nargs='+', choices=AUTHORIZED_TASKS)
    parser.add_argument('--prior-ledger', type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if not args.run_id.replace('_', '').isalnum():
        raise ValueError('invalid run ID')
    integration = Path(__file__).resolve().parent
    root = integration.parent
    simroot = root.parent
    output = root / 'results' / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    private = Path('/home/ROBODATA_USER/.config/aspire-pilot')
    check_private_key_metadata(private)
    relay = private / ('relay-' + args.run_id)
    relay.mkdir(mode=0o700, exist_ok=False)
    tasks = tuple(args.tasks or AUTHORIZED_TASKS)
    if len(set(tasks)) != len(tasks):
        raise ValueError('duplicate tasks are forbidden')
    if args.prior_ledger is None and tasks != AUTHORIZED_TASKS:
        raise ValueError('a task subset requires the original immutable budget ledger')
    prior_lock = None
    prior_audit = None
    if args.prior_ledger is not None:
        prior_path = args.prior_ledger.resolve(strict=True)
        prior_lock = lock_prior_ledger(prior_path)
        try:
            prior_audit = audit_prior_run(prior_path, tasks)
            prior_audit['continuation_claim'] = claim_continuation(prior_audit, output, tasks)
            (output / 'prior_run_audit.json').write_text(json.dumps(prior_audit, indent=2))
        except BaseException:
            os.close(prior_lock)
            raise
    episode_ids = [t + '_s0_l0' for t in tasks]
    config = {
        'usage_jsonl': str(output / 'usage.jsonl'),
        'episode_ids': episode_ids,
        'suite_budget_usd': '50' if prior_audit is None else prior_audit['remaining_budget_usd'],
        'max_iterations': 8,
        'client_source': str(integration / 'reviewed_astra_client.py'),
    }
    (output / 'controller_config.json').write_text(json.dumps(config, indent=2))
    py = root / '.venv-controller/bin/python'
    api_env = os.environ.copy()
    api_env.pop('OPENAI_API_KEY', None)
    api_env['OPENAI_API_KEY_FILE'] = str(private / 'openai.key')
    api_env['PYTHONUNBUFFERED'] = '1'
    attempts = []
    suite_completed = False
    final_budget = None
    api = None
    active_sim = None
    started = time.monotonic()

    def record(event, **fields):
        row = {'event': event, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(), **fields}
        with (output / 'attempts.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps(row), flush=True)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        with (output / 'controller.log').open('w') as stream:
            api = subprocess.Popen([str(py), str(integration / 'model_driver.py'),
                '--config', str(output / 'controller_config.json'), '--relay-dir', str(relay)],
                env=api_env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 60
        while not (relay / 'ready.json').exists():
            if api.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('controller did not become ready; no episode launched')
            time.sleep(.2)
        record('controller_ready', authorized_episodes=len(tasks), budget_usd=config['suite_budget_usd'],
               global_authorized_budget_usd='50', prior_charged_usd='0' if prior_audit is None else prior_audit['prior_charged_usd'])
        for task, episode_id in zip(tasks, episode_ids):
            verify_prior_unchanged(prior_audit)
            if api.poll() is not None or (relay / 'stopped.json').exists():
                raise RuntimeError('controller stopped; no further episodes allowed')
            taskout = output / task
            taskout.mkdir()
            sim_env = simulator_environment(os.environ)
            sim_env['SINGULARITY_BINDPATH'] = str(relay) + ':/aspire-relay'
            record('episode_process_start', task=task, seed=0, layout_id=0)
            began = time.monotonic()
            with (taskout / 'simulator.log').open('w') as stream:
                active_sim = subprocess.Popen(['bash', str(simroot / 'scripts/container_exec.sh'),
                    '/bin/bash', '/work/aspire_validate/integration/model_inside.sh',
                    task, args.run_id, episode_id], cwd=simroot, env=sim_env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    rc = active_sim.wait(timeout=1200)
                except subprocess.TimeoutExpired:
                    os.killpg(active_sim.pid, signal.SIGTERM)
                    try:
                        active_sim.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(active_sim.pid, signal.SIGKILL)
                        active_sim.wait()
                    raise RuntimeError('simulator timeout; no automatic retry')
            active_sim = None
            attempt = {'task': task, 'exit_code': rc, 'wall_seconds': time.monotonic() - began}
            attempts.append(attempt)
            record('episode_process_end', **attempt)
            if rc:
                raise RuntimeError('simulator failed; inspect preserved evidence before any new run')
            # Native RoboDojo can swallow rollout exceptions and Kit can exit
            # before our post-main checks. A zero process status is insufficient.
            report = json.loads((taskout / 'episode_report.json').read_text())
            official = validate_episode_report(report, task, episode_id)
            attempt['validated_episode'] = True
            attempt['schema_ledger_validated'] = True
            attempt['validation_scope'] = 'Report schema and ledger consistency only; not capability measurement validity'
            attempt['measurement_valid'] = None
            attempt['episode_id'] = episode_id
            attempt['model_calls'] = len(report['iterations'])
            attempt['programs_executed'] = report['total_programs_executed']
            attempt['official_success'] = official['official_success']
            record('episode_report_validated', task=task,
                   official_success=official['official_success'],
                   environment_steps=official['environment_steps'])
        # Stop the trusted controller only after all three clients are finished.
        import sys
        sys.path.insert(0, str(integration))
        from model_relay import RelayModelClient
        final_budget = RelayModelClient(relay).stop()['budget']
        if api.wait(timeout=10) != 0:
            raise RuntimeError('controller did not shut down cleanly')
        if (final_budget['aborted'] or final_budget['unsettled_attempts']
                or Decimal(final_budget['suite_charged_usd']) > Decimal(config['suite_budget_usd'])
                or any(final_budget['calls'][item['episode_id']] != item['model_calls'] for item in attempts)):
            raise RuntimeError('final budget ledger and episode reports do not reconcile')
        verify_prior_unchanged(prior_audit)
        suite_completed = True
        record('authorized_episodes_completed', tasks=list(tasks))
    finally:
        if active_sim is not None and active_sim.poll() is None:
            os.killpg(active_sim.pid, signal.SIGTERM)
            try:
                active_sim.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(active_sim.pid, signal.SIGKILL)
                active_sim.wait()
        if api is not None and api.poll() is None:
            os.killpg(api.pid, signal.SIGTERM)
            try:
                api.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(api.pid, signal.SIGKILL)
                api.wait()
        shutil.copytree(relay, output / 'relay_audit', dirs_exist_ok=False)
        prior_episodes = [] if prior_audit is None else prior_audit['prior_episodes']
        combined_episodes = prior_episodes + attempts
        prior_cost = Decimal(0) if prior_audit is None else Decimal(prior_audit['prior_charged_usd'])
        combined_cost = None if final_budget is None else str(prior_cost + Decimal(final_budget['suite_charged_usd']))
        (output / 'launcher_summary.json').write_text(json.dumps({
            'attempts': attempts, 'wall_seconds': time.monotonic() - started,
            'complete': suite_completed, 'budget': final_budget,
            'global_authorized_budget_usd': '50', 'prior_run_audit': prior_audit,
            'combined_charged_usd': combined_cost, 'combined_episodes': combined_episodes,
            'combined_scope': 'Authorized attempt coverage and accounting; not a three-task capability score',
            'combined_measurement_valid': False if any(item.get('measurement_valid') is False
                                                      for item in combined_episodes) else None,
            'combined_complete': suite_completed and len(combined_episodes) == 3
                and all(item.get('validated_episode') for item in combined_episodes),
        }, indent=2))
        if prior_lock is not None:
            os.close(prior_lock)


if __name__ == '__main__':
    main()
