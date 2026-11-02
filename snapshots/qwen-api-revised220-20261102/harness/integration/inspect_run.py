"""Print compact public progress and measured API usage; never print payloads."""
from __future__ import annotations
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import sys


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def inspect(root):
    root = Path(root)
    records = []
    if (root / 'usage.jsonl').exists():
        for line in (root / 'usage.jsonl').read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break  # Last line may be in flight.
    starts = {r['attempt_id'] for r in records if r['event'] == 'request_started'}
    settled = {r['attempt_id'] for r in records if r['event'] == 'request_settled'}
    result = {'run': root.name, 'events': dict(Counter(r['event'] for r in records)),
              'generation_attempts': len(starts), 'unsettled_attempts': len(starts-settled),
              'conservative_cost_usd': str(sum((Decimal(r['charged_usd']) for r in records
                  if r['event'] == 'request_settled'), Decimal(0))), 'tasks': []}
    for task in ('general_pickup', 'stack_bowls', 'align_blocks'):
        eid = task + '_s0_l0'
        usage = [r for r in records if r.get('episode_id') == eid]
        billed = [r for r in usage if r['event'] == 'request_settled']
        responses = [r for r in usage if r['event'] == 'response_received']
        entry = {'task': task, 'generation_attempts': sum(r['event']=='request_started' for r in usage),
                 'settled_calls': len(billed),
                 'conservative_cost_usd': str(sum((Decimal(r['charged_usd']) for r in billed), Decimal(0))),
                 'api_latency_s': sum(r['latency_s'] for r in responses)}
        for key in ('input_tokens', 'cached_input_tokens', 'cache_write_tokens', 'output_tokens', 'reasoning_tokens'):
            entry[key] = sum(r['token_breakdown'].get(key) or 0 for r in billed)
        report = read_json(root / task / 'episode_report.json')
        progress = report or read_json(root / task / 'episode_progress.json')
        if progress:
            entry['completed'] = bool(report and report.get('ok'))
            entry['program_iterations_recorded'] = len(progress.get('iterations', []))
            entry['termination_reason'] = progress.get('termination_reason')
            entry['official'] = progress.get('official')
            entry['episode_wall_s'] = progress.get('wall_seconds')
            entry['error'] = progress.get('error')
            if progress.get('iterations'):
                latest = progress['iterations'][-1]
                entry['latest_execution_ok'] = latest.get('execution', {}).get('program_ok')
                entry['environment_steps_after'] = latest.get('environment_steps_after')
        result['tasks'].append(entry)
    summary = read_json(root / 'launcher_summary.json')
    if summary:
        result['launcher'] = summary
    return result


if __name__ == '__main__':
    print(json.dumps(inspect(sys.argv[1]), indent=2))
