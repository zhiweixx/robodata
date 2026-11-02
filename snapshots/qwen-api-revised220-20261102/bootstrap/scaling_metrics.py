"""Stdlib-only, read-only resource and throughput measurements for scaling trials.

Sampling does not launch GPU compute, inspect process environments/command lines,
or change serving/robot behavior. VRAM/RSS peaks are sampled rather than exact.
"""
from __future__ import annotations

import collections
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    if value is None:
        return None
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timing requires explicit UTC offset')
    return parsed.timestamp()


def read_json(path):
    return json.loads(Path(path).read_text())


def numeric(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_gpu_csv(text):
    fields = ('index', 'uuid', 'utilization_gpu_percent', 'memory_used_mib',
              'memory_total_mib', 'power_draw_watts')
    rows = []
    for values in csv.reader(text.splitlines()):
        if not values:
            continue
        if len(values) != len(fields):
            raise ValueError('nvidia-smi returned unexpected column count')
        row = dict(zip(fields, (value.strip() for value in values)))
        for key in fields[2:]:
            row[key] = numeric(row[key])
        row['index'] = int(row['index'])
        total, used = row['memory_total_mib'], row['memory_used_mib']
        row['memory_used_fraction'] = used / total if total and used is not None else None
        row['memory_free_mib'] = total - used if total is not None and used is not None else None
        rows.append(row)
    return rows


def read_process_stat(path, page_size):
    """Parse proc stat without recording its comm field, including spaces/parentheses."""
    raw = Path(path).read_text()
    end = raw.rfind(')')
    if end < 0:
        raise ValueError('malformed proc stat')
    pid = int(raw.split('(', 1)[0].strip())
    rest = raw[end + 1:].split()
    return {'pid': pid, 'ppid': int(rest[1]), 'cpu_ticks': int(rest[11]) + int(rest[12]),
            'start_ticks': int(rest[19]), 'rss_bytes': int(rest[21]) * page_size}


def process_tree_sample(roots, *, previous=None, elapsed_seconds=None,
                        proc_root=Path('/proc'), clock_ticks=None, page_size=None):
    clock_ticks = clock_ticks or os.sysconf('SC_CLK_TCK')
    page_size = page_size or os.sysconf('SC_PAGE_SIZE')
    roots = {str(label): int(pid) for label, pid in roots.items() if isinstance(pid, int) and pid > 0}
    rows, children = {}, collections.defaultdict(list)
    for directory in Path(proc_root).iterdir():
        if not directory.name.isdigit():
            continue
        try:
            row = read_process_stat(directory / 'stat', page_size)
        except (OSError, ValueError, IndexError):
            continue  # Processes can exit between directory listing and read.
        rows[row['pid']] = row
        children[row['ppid']].append(row['pid'])

    def descendants(pid):
        found, pending = set(), [pid]
        while pending:
            current = pending.pop()
            if current in found or current not in rows:
                continue
            found.add(current)
            pending.extend(children[current])
        return found

    selected_by_root = {label: descendants(pid) for label, pid in roots.items()}
    selected = set().union(*selected_by_root.values()) if selected_by_root else set()
    previous = previous or {}
    for pid in selected:
        row, prior = rows[pid], previous.get(pid)
        row['cpu_cores'] = None
        if (elapsed_seconds and elapsed_seconds > 0 and prior is not None
                and prior['start_ticks'] == row['start_ticks']
                and row['cpu_ticks'] >= prior['cpu_ticks']):
            row['cpu_cores'] = (row['cpu_ticks'] - prior['cpu_ticks']) / clock_ticks / elapsed_seconds

    def aggregate(pids):
        values = [rows[pid]['cpu_cores'] for pid in pids if rows[pid]['cpu_cores'] is not None]
        return {'process_count': len(pids), 'rss_bytes': sum(rows[pid]['rss_bytes'] for pid in pids),
                'cpu_cores': sum(values) if values else None,
                'cpu_processes_with_delta': len(values)}

    result = {**aggregate(selected), 'roots': roots,
              'by_root': {label: aggregate(pids) for label, pids in selected_by_root.items()},
              'scope': 'Union of supplied host process roots and current descendants; no duplicate PIDs.',
              'rss_note': 'Sum of RSS counts shared pages multiple times; it is not unique physical memory.',
              'cpu_note': 'Delta for surviving PID/start-time pairs; short-lived exited processes can be missed.'}
    return result, {pid: rows[pid] for pid in selected}


def cgroup_memory_sample(proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    """Inspect this task's cgroup. Do not assume that it covers other Slurm steps."""
    lines = (Path(proc_root) / 'self/cgroup').read_text().splitlines()
    version, relative = None, None
    for line in lines:
        hierarchy, controllers, value = line.split(':', 2)
        if hierarchy == '0' and controllers == '':
            version, relative = 2, value
            break
        if 'memory' in controllers.split(','):
            version, relative = 1, value
    if relative is None:
        return {'available': False, 'reason': 'no memory cgroup listed'}
    if '..' in Path(relative).parts:
        return {'available': False, 'reason': 'unsupported cgroup path'}
    mount = Path(cgroup_root) if version == 2 else Path(cgroup_root) / 'memory'
    directory = mount / relative.lstrip('/')
    current_name = 'memory.current' if version == 2 else 'memory.usage_in_bytes'
    limit_name = 'memory.max' if version == 2 else 'memory.limit_in_bytes'
    try:
        current = int((directory / current_name).read_text().strip())
    except (OSError, ValueError):
        return {'available': False, 'reason': 'memory counter unavailable', 'version': version}
    try:
        peak_name = 'memory.peak' if version == 2 else 'memory.max_usage_in_bytes'
        kernel_peak = int((directory / peak_name).read_text().strip())
    except (OSError, ValueError):
        kernel_peak = None
    limits, parent = [], directory
    while parent.is_relative_to(mount):
        try:
            raw = (parent / limit_name).read_text().strip()
            value = int(raw)
            if 0 < value < 1 << 60:
                limits.append(value)
        except (OSError, ValueError):
            pass
        if parent == mount:
            break
        parent = parent.parent
    events = {}
    if version == 2:
        try:
            for line in (directory / 'memory.events').read_text().splitlines():
                key, value = line.split()
                events[key] = int(value)
        except (OSError, ValueError):
            pass
    else:
        try:
            events['limit_failcnt'] = int((directory / 'memory.failcnt').read_text().strip())
        except (OSError, ValueError):
            pass
    limit = min(limits) if limits else None
    return {'available': True, 'version': version, 'memory_current_bytes': current,
            'kernel_peak_bytes': kernel_peak,
            'memory_limit_bytes': limit, 'memory_fraction': current / limit if limit else None,
            'events': events, 'scope': 'Collector task cgroup; may exclude another Slurm step.'}


def host_memory_sample(proc_root=Path('/proc')):
    values = {}
    for line in (Path(proc_root) / 'meminfo').read_text().splitlines():
        key, raw = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable'):
            values[key + '_bytes'] = int(raw.strip().split()[0]) * 1024
    values['scope'] = 'Whole shared host, including other users; context only, not job memory use.'
    return values


class MetricsCollector:
    def __init__(self, output_path, gpu_uuids, *, root_pids=None, interval_seconds=3,
                 memory_limit_bytes=None):
        if not 2 <= interval_seconds <= 5:
            raise ValueError('sampling interval must be between 2 and 5 seconds')
        self.path = Path(output_path)
        self.gpu_uuids = list(gpu_uuids)
        if (not self.gpu_uuids or len(set(self.gpu_uuids)) != len(self.gpu_uuids)
                or any(not str(value).startswith('GPU-') or ',' in value for value in self.gpu_uuids)):
            raise ValueError('explicit distinct allocated GPU UUIDs required')
        self.root_pids = root_pids or (lambda: {'supervisor': os.getpid()})
        self.interval = float(interval_seconds)
        self.memory_limit_bytes = memory_limit_bytes
        self._stop = threading.Event()
        self._thread = None
        self._previous = {}
        self._previous_time = None

    def sample(self):
        start = time.monotonic()
        row = {'utc': utc_now(), 'monotonic_seconds': start, 'errors': [],
               'configured_sampling_interval_seconds': self.interval,
               'configured_memory_limit_bytes': self.memory_limit_bytes}
        try:
            completed = subprocess.run(['nvidia-smi', '--id=' + ','.join(self.gpu_uuids),
                '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4)
            if completed.returncode:
                raise RuntimeError('nvidia-smi exit ' + str(completed.returncode))
            row['gpus'] = parse_gpu_csv(completed.stdout)
            if {gpu['uuid'] for gpu in row['gpus']} != set(self.gpu_uuids):
                raise RuntimeError('GPU telemetry does not match assigned UUIDs')
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            row['errors'].append({'component': 'gpu', 'type': type(exc).__name__, 'message': str(exc)})
            row['gpus'] = []
        try:
            roots = dict(self.root_pids())
            row['processes'], self._previous = process_tree_sample(roots, previous=self._previous,
                elapsed_seconds=None if self._previous_time is None else start - self._previous_time)
            self._previous_time = start
        except (OSError, ValueError, RuntimeError) as exc:
            row['errors'].append({'component': 'processes', 'type': type(exc).__name__, 'message': str(exc)})
        try:
            row['cgroup'] = cgroup_memory_sample()
        except (OSError, ValueError) as exc:
            row['errors'].append({'component': 'cgroup', 'type': type(exc).__name__, 'message': str(exc)})
        try:
            row['host_memory'] = host_memory_sample()
        except (OSError, ValueError):
            row['host_memory'] = {'available': False}
        row['sample_wall_seconds'] = time.monotonic() - start
        return row

    def _run(self):
        with self.path.open('a') as handle:
            while not self._stop.is_set():
                started = time.monotonic()
                row = self.sample()
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                handle.flush()
                self._stop.wait(max(0, self.interval - (time.monotonic() - started)))

    def start(self):
        if self._thread is not None:
            raise RuntimeError('collector cannot be restarted')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        self._thread = threading.Thread(target=self._run, name='scaling-metrics', daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError('resource sampler did not stop; metrics file still active')
        return summarize_telemetry(self.path)


def _max(values):
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _min(values):
    present = [value for value in values if value is not None]
    return min(present) if present else None


def summarize_telemetry(path):
    rows, malformed = [], 0
    if path is not None and Path(path).exists():
        with Path(path).open() as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1
    result = {'sample_count': len(rows), 'malformed_sample_count': malformed,
              'sampling_error_count': sum(len(row.get('errors', [])) for row in rows) + malformed,
              'gpus': {}, 'processes': {}, 'cgroup': {}, 'host_memory': {},
              'sampled_peaks_are_lower_bounds': True,
              'perception_queue_and_compute_times_available': False}
    result['configured_sampling_interval_seconds'] = _max(row.get('configured_sampling_interval_seconds') for row in rows)
    points = [row['monotonic_seconds'] for row in rows if 'monotonic_seconds' in row]
    result['maximum_sample_gap_seconds'] = max((b - a for a, b in zip(points, points[1:])), default=None)
    result['maximum_sampler_wall_seconds'] = _max(row.get('sample_wall_seconds') for row in rows)
    gpu_rows = collections.defaultdict(list)
    for row in rows:
        for gpu in row.get('gpus', []):
            gpu_rows[gpu['uuid']].append(gpu)
    for uuid, samples in gpu_rows.items():
        util = [gpu['utilization_gpu_percent'] for gpu in samples if gpu['utilization_gpu_percent'] is not None]
        result['gpus'][uuid] = {
            'sample_count': len(samples), 'peak_memory_used_mib': _max(gpu.get('memory_used_mib') for gpu in samples),
            'memory_total_mib': _min(gpu.get('memory_total_mib') for gpu in samples),
            'peak_memory_used_fraction': _max(gpu.get('memory_used_fraction') for gpu in samples),
            'min_memory_free_mib': _min(gpu.get('memory_free_mib') for gpu in samples),
            'peak_gpu_utilization_percent': max(util) if util else None,
            'mean_gpu_utilization_percent': statistics.mean(util) if util else None,
            'peak_power_draw_watts': _max(gpu.get('power_draw_watts') for gpu in samples)}
    process_rows = [row['processes'] for row in rows if 'processes' in row]
    result['processes'] = {'peak_rss_bytes': _max(row.get('rss_bytes') for row in process_rows),
        'peak_cpu_cores': _max(row.get('cpu_cores') for row in process_rows),
        'peak_process_count': _max(row.get('process_count') for row in process_rows),
        'rss_note': 'Sum of process RSS double-counts shared pages; not unique resident memory.'}
    labels = {label for row in process_rows for label in row.get('by_root', {})}
    result['processes']['by_root'] = {
        label: {'peak_rss_bytes': _max(row.get('by_root', {}).get(label, {}).get('rss_bytes') for row in process_rows),
                'peak_cpu_cores': _max(row.get('by_root', {}).get(label, {}).get('cpu_cores') for row in process_rows)}
        for label in sorted(labels)}
    cg = [row['cgroup'] for row in rows if row.get('cgroup', {}).get('available')]
    limit = _min(row.get('memory_limit_bytes') for row in cg)
    configured = _min(row.get('configured_memory_limit_bytes') for row in rows)
    if configured and (limit is None or configured < limit):
        limit = configured
    peak = _max(row.get('memory_current_bytes') for row in cg)
    kernel_peak = _max(row.get('kernel_peak_bytes') for row in cg)
    conservative_peak = _max((peak, kernel_peak))
    result['cgroup'] = {'available': bool(cg), 'peak_memory_current_bytes': peak,
        'min_memory_current_bytes': _min(row.get('memory_current_bytes') for row in cg),
        'first_memory_current_bytes': cg[0].get('memory_current_bytes') if cg else None,
        'kernel_peak_bytes': kernel_peak,
        'memory_limit_bytes': limit,
        'peak_memory_fraction': conservative_peak / limit if conservative_peak is not None and limit else None,
        'scope': 'Collector task cgroup; kernel peak may include prior stages in the same allocation.'}
    result['host_memory'] = {
        'min_available_bytes': _min(row.get('host_memory', {}).get('MemAvailable_bytes') for row in rows),
        'total_bytes': _min(row.get('host_memory', {}).get('MemTotal_bytes') for row in rows),
        'scope': 'Whole shared host; not a per-job capacity guard.'}
    for name in ('oom', 'oom_kill', 'limit_failcnt'):
        counters = [row['events'][name] for row in cg if name in row.get('events', {})]
        result['cgroup'][name + '_delta'] = max(counters) - counters[0] if counters else None
    return result


def overlap_summary(intervals):
    """Interval peak, union busy time and weighted concurrency; endpoints may touch."""
    events = []
    for start, finish in intervals:
        if finish < start:
            raise ValueError('negative request duration')
        events.extend([(start, 1), (finish, -1)])
    if not events:
        return {'peak_concurrency': 0, 'busy_wall_seconds': 0.0,
                'request_seconds': 0.0, 'mean_concurrency_while_busy': None}
    grouped = collections.defaultdict(int)
    for point, delta in events:
        grouped[point] += delta
    active, peak, busy, integral = 0, 0, 0.0, 0.0
    prior = min(grouped)
    for point in sorted(grouped):
        elapsed = point - prior
        if active > 0:
            busy += elapsed
            integral += active * elapsed
        active += grouped[point]
        peak = max(peak, active)
        prior = point
    return {'peak_concurrency': peak, 'busy_wall_seconds': busy, 'request_seconds': integral,
            'mean_concurrency_while_busy': integral / busy if busy else None}


def summarize_stage(workers, *, stage_started_utc, stage_finished_utc, telemetry_path=None):
    """Summarize one burst; independent native/video audit remains runner's responsibility.

    Each worker mapping supplies worker_id, usage_path, episode_report_paths,
    expected_layout_ids, native_result_path, and optional simulator launch times.
    Missing artifacts are retained as infrastructure/incomplete evidence.
    """
    begin, end = timestamp(stage_started_utc), timestamp(stage_finished_utc)
    if begin is None or end is None or end <= begin:
        raise ValueError('valid positive stage timing required')
    ids = [worker['worker_id'] for worker in workers]
    if len(set(ids)) != len(ids):
        raise ValueError('distinct worker identities required')
    result = {'expected_workers': len(workers), 'complete_scored_workers': 0,
              'expected_episodes': 0, 'complete_scored_episodes': 0,
              'workers_without_native_score': [], 'request_errors': [], 'workers': [],
              'stage_wall_seconds': end - begin, 'stage_started_utc': stage_started_utc,
              'stage_finished_utc': stage_finished_utc, 'is_accuracy_estimate': False,
              'sustained_warm_throughput_measured': False,
              'server_batch_size_measured': False,
              'exact_time_to_first_token_available': False,
              'resources': summarize_telemetry(telemetry_path)}
    request_intervals, episode_intervals = [], []
    layouts = collections.defaultdict(lambda: {'expected': 0, 'scored': 0, 'timings': [], 'durations': [], 'requests': []})
    prompt_total = completion_total = image_total = 0
    generation_total = 0.0
    all_requests, completed_requests = 0, 0
    for worker in workers:
        wid = worker['worker_id']
        entry = {'worker_id': wid, 'issues': [], 'requests': [], 'native_score_committed': False}
        rows = []
        usage_path = worker.get('usage_path')
        if usage_path and Path(usage_path).is_file():
            with Path(usage_path).open() as handle:
                for line in handle:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        entry['issues'].append('Incomplete/malformed usage record retained.')
        else:
            entry['issues'].append('No usage ledger available.')
        starts = {(row.get('episode_id'), row.get('attempt', 1), row.get('iteration')): row
                  for row in rows if row.get('event') == 'generation_started'}
        finishes = {(row.get('episode_id'), row.get('attempt', 1), row.get('iteration')): row
                    for row in rows if row.get('event') in ('raw_response', 'generation_transport_failure')}
        all_requests += len(starts)
        for key, start_row in starts.items():
            start = timestamp(start_row['utc'])
            finish_row = finishes.get(key)
            finish = timestamp(finish_row['utc']) if finish_row else end
            if finish is not None and start is not None and finish >= start:
                request_intervals.append((start, finish))
            else:
                entry['issues'].append('Invalid request interval; omitted from concurrency.')
        seen = set()
        for row in rows:
            event = row.get('event')
            if event in ('generation_transport_failure', 'generation_usage_unavailable',
                         'infrastructure_error', 'infrastructure_stopped', 'worker_infrastructure_failure'):
                result['request_errors'].append({'worker_id': wid, 'event': event,
                    'iteration': row.get('iteration'), 'error_type': row.get('error_type'),
                    'code': row.get('code')})
            if event != 'generation_usage':
                continue
            key = (row.get('episode_id'), row.get('attempt', 1), row.get('iteration'))
            if key in seen:
                entry['issues'].append('Duplicate usage identity; duplicate not summed.')
                continue
            seen.add(key)
            usage = row.get('usage', {})
            latency = numeric(usage.get('generation_wall_s'))
            prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
            if (type(prompt) is not int or type(completion) is not int or min(prompt, completion) < 0
                    or latency is None or latency < 0):
                entry['issues'].append('Invalid provider usage counters.')
                continue
            completed_requests += 1
            prompt_total += prompt
            completion_total += completion
            generation_total += latency
            image = usage.get('prompt_tokens_details', {}).get('image_tokens')
            image_total += image if type(image) is int else 0
            started = starts.get(key)
            start = timestamp(started['utc']) if started else None
            if start is None:
                entry['issues'].append('Usage without request start timestamp.')
            entry['requests'].append({'episode_id': key[0], 'iteration': key[2], 'attempt': key[1],
                'prompt_tokens': prompt, 'completion_tokens': completion,
                'image_tokens_raw': image, 'request_wall_seconds': latency,
                'request_started_utc': started.get('utc') if started else None})
        unresolved = set(starts) - seen
        if unresolved:
            result['request_errors'].append({'worker_id': wid, 'event': 'requests_without_complete_usage',
                                            'count': len(unresolved)})
        report_paths = worker.get('episode_report_paths')
        if report_paths is None:
            report_paths = [worker['episode_report_path']] if worker.get('episode_report_path') else []
        reports = [read_json(path) for path in report_paths if Path(path).is_file()]
        native_path = worker.get('native_result_path')
        native = read_json(native_path) if native_path and Path(native_path).is_file() else None
        expected_ids = worker.get('expected_layout_ids')
        if expected_ids is None:
            expected_ids = list(range(worker.get('target_episodes', 1)))
        if (not expected_ids or len(set(expected_ids)) != len(expected_ids)
                or any(type(value) is not int for value in expected_ids)):
            raise ValueError('explicit distinct expected layout IDs required')
        result['expected_episodes'] += len(expected_ids)
        entry['expected_layout_ids'] = expected_ids
        entry['episodes'] = []
        entry['tool_calls'] = []
        native_by_layout = {}
        if native:
            for detail in native.get('details', {}).values():
                layout = detail.get('layout_id')
                if layout in native_by_layout:
                    entry['issues'].append('Duplicate native layout ID; not accepted as complete.')
                    native_by_layout = {}
                    break
                if layout in expected_ids and type(detail.get('success')) is bool:
                    native_by_layout[layout] = detail
        committed = set()
        for layout in expected_ids:
            layouts[layout]['expected'] += 1
        seen_reports = set()
        for report in reports:
            layout = report.get('layout_id')
            if layout in seen_reports or layout not in expected_ids:
                entry['issues'].append('Duplicate/unexpected episode report retained; no double counting.')
                continue
            seen_reports.add(layout)
            episode = {'episode_id': report.get('episode_id'), 'layout_id': layout,
                'episode_wall_seconds': report.get('wall_seconds'),
                'environment_ticks': report.get('environment_steps'), 'step_limit': report.get('step_limit'),
                'termination_reason': report.get('termination_reason'),
                'native_score_committed': False,
                'model_declared_done': report.get('termination_reason') == 'model_done'}
            timing = report.get('scaling_timing', {})
            start = timestamp(timing.get('episode_started_utc'))
            finish = timestamp(timing.get('episode_callback_finished_utc'))
            if start is not None and finish is not None and finish >= start:
                episode['episode_timing'] = timing
                episode_intervals.append((start, finish))
                layouts[layout]['timings'].append((start, finish))
            duration = numeric(report.get('wall_seconds'))
            if duration is not None:
                layouts[layout]['durations'].append(duration)
            layouts[layout]['requests'] += [request for request in entry['requests']
                                           if request['episode_id'] == report.get('episode_id')]
            if report.get('native_score_committed') is True and layout in native_by_layout:
                detail = native_by_layout[layout]
                episode.update(native_score_committed=True, success=detail['success'], score=detail.get('score'))
                committed.add(layout)
                result['complete_scored_episodes'] += 1
                layouts[layout]['scored'] += 1
            for iteration in report.get('iterations', []):
                for call in (iteration.get('execution') or {}).get('tool_calls', []):
                    entry['tool_calls'].append({'layout_id': layout, 'iteration': iteration.get('iteration'),
                        **{key: call.get(key) for key in ('tool', 'ok', 'failure_code', 'duration_s')}})
            entry['episodes'].append(episode)
        if committed == set(expected_ids):
            entry['native_score_committed'] = True
            result['complete_scored_workers'] += 1
        else:
            result['workers_without_native_score'].append(wid)
        if entry['issues']:
            result['request_errors'].append({'worker_id': wid, 'event': 'worker_measurement_incomplete',
                                            'issues': entry['issues']})
        result['workers'].append(entry)
    concurrency = overlap_summary(request_intervals)
    episode_span = (max(end for _, end in episode_intervals) - min(start for start, _ in episode_intervals)
                    if len(episode_intervals) == result['expected_episodes'] and episode_intervals else None)
    latency_values = [request['request_wall_seconds'] for worker in result['workers'] for request in worker['requests']]
    result.update(model_request_count=all_requests, requests_with_usage=completed_requests,
        provider_reported_prompt_tokens=prompt_total, provider_reported_completion_tokens=completion_total,
        provider_reported_image_tokens=image_total, summed_request_wall_seconds=generation_total,
        api_concurrency=concurrency, mean_request_wall_seconds=statistics.mean(latency_values) if latency_values else None,
        p95_request_wall_seconds=(sorted(latency_values)[max(0, math.ceil(len(latency_values) * .95) - 1)] if latency_values else None),
        completion_tokens_per_api_busy_second=completion_total / concurrency['busy_wall_seconds'] if concurrency['busy_wall_seconds'] else None,
        completion_tokens_per_summed_request_second=completion_total / generation_total if generation_total else None,
        cold_start_inclusive_scored_episodes_per_hour=result['complete_scored_episodes'] * 3600 / (end - begin),
        episode_window_seconds=episode_span,
        startup_excluded_burst_episodes_per_hour=(result['complete_scored_episodes'] * 3600 / episode_span if episode_span else None),
        token_semantics='Provider counters; completion-token reasoning inclusion is unverified.',
        latency_semantics='End-to-end HTTP: upload/network, server queue, prefill and decode; not pure decode time.',
        warm_rate_semantics='Per-layout windows distinguish first versus warm second episode; one second episode per worker is not sustained warm throughput.',
        perception_latency_semantics='Tool receipts include observation/render/projection plus HTTP; server queue and compute are not separable.',
        server_batch_note='Client concurrency is measured; actual server batching/cache/scheduler settings remain unknown.')
    result['p50_request_wall_seconds'] = statistics.median(latency_values) if latency_values else None
    result['latency_by_prompt_length'] = latency_buckets(
        [request for worker in result['workers'] for request in worker['requests']])
    result['per_layout'] = {}
    for layout, values in sorted(layouts.items()):
        intervals = values['timings']
        span = (max(finish for _, finish in intervals) - min(start for start, _ in intervals)
                if len(intervals) == values['expected'] and intervals else None)
        result['per_layout'][str(layout)] = {
            'expected_episodes': values['expected'], 'scored_episodes': values['scored'],
            'episode_window_seconds': span,
            'scored_episodes_per_window_hour': values['scored'] * 3600 / span if span else None,
            'episode_wall_seconds': values['durations'],
            'median_episode_wall_seconds': statistics.median(values['durations']) if values['durations'] else None,
            'request_latency_by_prompt_length': latency_buckets(values['requests']),
            'cache_condition': 'first episode in simulator' if layout == 0 else 'second episode in same persistent simulator' if layout == 1 else 'unspecified'}
    return result


def latency_buckets(requests):
    bins = [(0, 8192), (8192, 16384), (16384, 32768), (32768, 65536), (65536, 131072), (131072, None)]
    result = []
    for low, high in bins:
        selected = [row for row in requests if row['prompt_tokens'] >= low and (high is None or row['prompt_tokens'] < high)]
        if not selected:
            continue
        latencies = sorted(row['request_wall_seconds'] for row in selected)
        seconds = sum(latencies)
        result.append({'prompt_tokens_min_inclusive': low, 'prompt_tokens_max_exclusive': high,
            'requests': len(selected), 'p50_request_wall_seconds': statistics.median(latencies),
            'p95_request_wall_seconds': latencies[max(0, math.ceil(.95 * len(latencies)) - 1)],
            'completion_tokens': sum(row['completion_tokens'] for row in selected),
            'summed_request_seconds': seconds,
            'completion_tokens_per_summed_request_second': sum(row['completion_tokens'] for row in selected) / seconds if seconds else None})
    return result


def assess_expansion(summary, *, simulator_gpu_uuids, max_gpu_memory_fraction=.85,
                     max_host_memory_fraction=.85):
    """Conservative admission heuristic, never a guarantee of capacity for new work."""
    reasons = []
    if summary['complete_scored_workers'] != summary['expected_workers']:
        reasons.append('At least one worker lacks a committed native score.')
    if summary.get('request_errors'):
        reasons.append('Inference requests have errors or incomplete accounting.')
    resources = summary.get('resources', {})
    if not resources.get('sample_count') or resources.get('sampling_error_count'):
        reasons.append('Resource telemetry missing or contains sampling failures.')
    for uuid in simulator_gpu_uuids:
        peak = resources.get('gpus', {}).get(uuid, {}).get('peak_memory_used_fraction')
        if peak is None or peak > max_gpu_memory_fraction:
            reasons.append('Simulator GPU VRAM headroom not demonstrated: ' + uuid)
    cg = resources.get('cgroup', {})
    fraction = cg.get('peak_memory_fraction')
    if fraction is None or fraction > max_host_memory_fraction:
        reasons.append('Host cgroup memory headroom not demonstrated.')
    if any((cg.get(key) or 0) > 0 for key in ('oom_delta', 'oom_kill_delta', 'limit_failcnt_delta')):
        reasons.append('Cgroup reports a memory-limit or OOM event during this stage.')
    return {'allowed': not reasons, 'reasons': reasons,
            'thresholds': {'max_gpu_memory_fraction': max_gpu_memory_fraction,
                           'max_host_memory_fraction': max_host_memory_fraction},
            'meaning': 'Admission heuristic based on sampled prior load; not a memory or throughput guarantee.'}
