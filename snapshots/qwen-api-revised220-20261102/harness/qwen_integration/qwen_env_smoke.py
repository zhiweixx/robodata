"""Diagnostic: original deterministic smoke under Qwen's host environment, no model."""
import json
import os
from pathlib import Path
import subprocess
import time
from qwen_server_suite import sanitized_environment, DEFAULT_ROOT

root = DEFAULT_ROOT
run_id = 'qwen_env_smoke_' + os.environ['SLURM_JOB_ID']
out = root / 'qwen' / 'runs' / run_id
out.mkdir(parents=True, exist_ok=False)
env = sanitized_environment(os.environ, root)
restored_proxy_names = []
if os.environ.get('ASPIRE_RESTORE_PROXY') == '1':
    for name, value in os.environ.items():
        normalized = name.upper()
        for prefix in ('SINGULARITYENV_', 'APPTAINERENV_'):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        if normalized in {'HTTP_PROXY', 'HTTPS_PROXY', 'FTP_PROXY', 'ALL_PROXY', 'NO_PROXY'}:
            env[name] = value
            restored_proxy_names.append(name)
env['PATH'] = str(root / 'qwen/.venv-runtime/bin') + os.pathsep + env.get('PATH', '')
start = time.monotonic()
with (out / 'simulator.log').open('w') as log:
    process = subprocess.Popen(['bash', str(root.parent / 'scripts/container_exec.sh'),
        '/bin/bash', '/work/aspire_validate/integration/smoke_inside.sh', run_id],
        cwd=root.parent, env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    try:
        status = process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        import signal
        os.killpg(process.pid, signal.SIGKILL)
        status = process.wait()
report = {'run_id': run_id, 'model_requests': 0, 'exit_code': status,
          'wall_seconds': time.monotonic() - start,
          'restored_proxy_names': sorted(restored_proxy_names)}
report_path = root / 'results' / ('aspire_smoke_' + run_id) / 'smoke_report.json'
report['smoke_report'] = str(report_path)
report['ok'] = status == 0 and report_path.exists() and json.loads(report_path.read_text()).get('ok') is True
(out / 'diagnosis.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
raise SystemExit(0 if report['ok'] else 1)
