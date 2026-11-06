"""Bounded, aggregate-only CUDA allocation accounting for one full-episode preflight."""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
import linecache
from pathlib import Path
import sys


MAX_EVENTS = 65536


@lru_cache(maxsize=1)
def _model_lines():
    """Resolve Qwen class scopes in the installed implementation (all methods are named forward)."""
    mod = sys.modules.get('transformers.models.qwen3_5.modeling_qwen3_5')
    if mod is None or not getattr(mod, '__file__', None):
        return {}
    source = Path(mod.__file__).read_text()
    tags = {'GatedDeltaNet': 'linear_attention', 'Attention': 'standard_attention',
            'RMSNorm': 'normalization', 'MLP': 'mlp', 'Vision': 'vision'}
    lines = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            tag = ('vision' if 'Vision' in node.name else
                   next((tag for name, tag in tags.items() if name in node.name), None))
            if tag:
                lines.update(dict.fromkeys(range(node.lineno, node.end_lineno + 1), tag))
    return lines


def _classify(frames):
    """Tag allocating call site, not ownership inferred from the size of a tensor."""
    phase = 'forward_or_other'
    parts = []
    for frame in frames:
        path, name, line = frame.get('filename', ''), frame.get('name', ''), frame.get('line', 0)
        if 'checkpoint.py' in path and ('recomput' in name or name == 'unpack_hook'):
            phase = 'checkpoint_recomputation'
        elif phase != 'checkpoint_recomputation' and (name in ('_engine_run_backward', 'backward', 'unpack')
                or (name.endswith('_bwd') and '/fla/ops/' in path)):
            phase = 'backward_or_reload'
        if '/peft/tuners/lora/' in path:
            text = linecache.getline(path, line)
            if 'lora_A(' in text or '_cast_input_dtype' in text or 'result.to(torch_result_dtype)' in text:
                parts.append('lora_adapter_activity')
        elif path.endswith('modeling_qwen3_5.py'):
            parts.append(_model_lines().get(line))
        elif '/fla/ops/' in path:
            parts.append('linear_attention')
        elif 'sdpa_attention' in name:
            parts.append('standard_attention')
        elif path.endswith('training.py') and name in ('_project_loss', '_inputs', 'unpack'):
            parts.append({'_project_loss': 'loss_projection', '_inputs': 'input_transfer',
                          'unpack': 'activation_reload'}[name])
        elif ('weight_offload' in path or 'weight_stag' in path) and name in ('unpack', 'forward'):
            if 'host.to(' in linecache.getline(path, line):
                parts.append('base_weight_staging')
    if 'lora_adapter_activity' in parts:
        return 'lora_adapter_activity', phase
    return next((p for p in parts if p), 'unattributed'), phase


def _base_blocks(segments, device, pointers):
    blocks = {}
    for segment in segments:
        if segment.get('device') == device:
            for b in segment['blocks']:
                if b['state'] == 'active_allocated':
                    addr = b['address']
                    blocks[addr] = (b['size'], pointers.get(addr, 'other_preexisting'), 'preexisting')
    return blocks


def _summarize(initial, events, *, peak, ending, limit):
    """Replay from baseline only if the complete interval is retained; reject gaps or unknown addresses."""
    live = dict(initial)
    component = Counter()
    phase = Counter()
    for size, category, origin in live.values():
        component[category] += size
        phase[origin] += size
    total = observed_peak = sum(component.values())
    peak_components, peak_phases = dict(component), dict(phase)
    oom = []
    issues = []
    if len(events) >= limit:
        issues.append('event ring reached capacity; prefix may be absent')
    for event in events:
        act, size = event['action'], event['size']
        if act == 'oom':
            category, origin = _classify(event.get('frames', ()))
            oom.append({'requested_bytes': size, 'device_free_bytes': event.get('device_free'),
                        'call_site': category, 'phase': origin})
        if issues:
            continue
        if act == 'alloc':
            addr = event['addr']
            if addr in live:
                issues.append('allocation reused a still-live address in this replay')
                continue
            category, origin = _classify(event.get('frames', ()))
            live[addr] = (size, category, origin)
            component[category] += size
            phase[origin] += size
            total += size
            if total > observed_peak:
                observed_peak = total
                peak_components = {k: v for k, v in component.items() if v}
                peak_phases = {k: v for k, v in phase.items() if v}
        elif act == 'free_requested':
            item = live.pop(event['addr'], None)
            if item is None:
                issues.append('free of an unknown address in this replay')
                continue
            allocated_size, category, origin = item
            component[category] -= allocated_size
            phase[origin] -= allocated_size
            total -= allocated_size
    if not issues and (total != ending or observed_peak != peak):
        issues.append('event-size replay and allocator counters differ (e.g. allocator rounding)')
    result = dict(trace_entries=len(events), event_capacity=limit, coverage_issues=issues, oom_events=oom[-3:])
    if not issues or (len(issues) == 1 and issues[0].startswith('event-size')):
        result.update(event_size_peak_bytes=observed_peak, event_size_ending_bytes=total,
                      allocator_minus_event_peak_bytes=peak - observed_peak,
                      event_size_peak_by_call_site_bytes=peak_components,
                      event_size_peak_by_phase_bytes=peak_phases,
                      peak_and_ending_counters_match=not issues)
    return result


@contextmanager
def profile_preflight(model, attempt):
    """Write small scalar aggregates, never serializing stacks; a profiler error cannot replace a model error."""
    import torch
    cuda = torch.cuda
    report = {}
    attempt['cuda_memory_profile'] = report
    recording = False
    try:
        dev = cuda.current_device()
        ptrs = {p.untyped_storage().data_ptr(): ('trainable_parameters' if p.requires_grad else 'frozen_weights')
                for p in model.parameters() if p.is_cuda}
        ptrs.update({b.untyped_storage().data_ptr(): 'registered_buffers' for b in model.buffers()
                     if b.is_cuda and b.untyped_storage().data_ptr() not in ptrs})
        before = _base_blocks(cuda.memory_snapshot(include_traces=False), dev, ptrs)
        baseline = Counter()
        for size, kind, _ in before.values():
            baseline[kind] += size
        report.update(baseline_active_blocks_bytes=dict(baseline), baseline_allocated_bytes=cuda.memory_allocated(),
                      baseline_reserved_bytes=cuda.memory_reserved(), device_total_bytes=cuda.mem_get_info()[1])
        cuda.memory._record_memory_history(enabled='all', context='alloc', stacks='python',
            max_entries=MAX_EVENTS, clear_history=True, device=dev,
            skip_actions=['free_completed', 'segment_alloc', 'segment_free', 'snapshot'])
        recording = True
    except Exception as err:
        report['setup_error_type'] = type(err).__name__
    try:
        yield
    finally:
        try:
            peak, ending = cuda.max_memory_allocated(), cuda.memory_allocated()
            report.update(allocator_peak_allocated_bytes=peak, allocator_peak_reserved_bytes=cuda.max_memory_reserved(),
                          ending_allocated_bytes=ending, ending_reserved_bytes=cuda.memory_reserved(),
                          ending_device_free_bytes=cuda.mem_get_info()[0])
            if recording:
                snap = cuda.memory._snapshot()
                # End-state block sizes are actual allocator block sizes, independently
                # of history truncation/request-size rounding. Keep this instant distinct
                # from an earlier peak or a failed (never allocated) request.
                at_end = Counter()
                for segment in snap['segments']:
                    if segment.get('device') != dev:
                        continue
                    for b in segment['blocks']:
                        if b['state'] != 'active_allocated':
                            continue
                        addr = b['address']
                        if addr in ptrs:
                            site = ptrs[addr]
                        else:
                            site = _classify(b.get('frames', ()))[0]
                        at_end[site] += b['size']
                report['ending_active_blocks_by_call_site_bytes'] = dict(at_end)
                report['ending_active_blocks_sum_equals_allocator_counter'] = sum(at_end.values()) == ending
                report['ending_snapshot_was_at_peak'] = sum(at_end.values()) == ending == peak
                events = snap['device_traces'][dev]
                report.update(_summarize(before, events, peak=peak, ending=ending, limit=MAX_EVENTS))
        except Exception as err:
            report['analysis_error_type'] = type(err).__name__
        finally:
            if recording:
                try:
                    cuda.memory._record_memory_history(enabled=None, clear_history=True, device=dev)
                except Exception as err:
                    report['disable_error_type'] = type(err).__name__
