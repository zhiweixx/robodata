from robodata.sft.memory_profile import _summarize


def test_concurrent_allocations_peak_and_oom():
    events = [
        dict(action='alloc', addr=20, size=60, frames=[]),
        dict(action='alloc', addr=30, size=11, frames=[dict(filename='/a/fla/ops/gated.py', line=3, name='kernel')]),
        dict(action='oom', size=80, device_free=32, frames=[]),
        dict(action='free_requested', addr=20, size=60),
        dict(action='free_requested', addr=30, size=11),
    ]
    result = _summarize({1:(100,'frozen_weights','preexisting')}, events, peak=171, ending=100, limit=20)
    assert result['peak_and_ending_counters_match']
    assert result['event_size_peak_by_call_site_bytes'] == {'frozen_weights':100,'unattributed':60,'linear_attention':11}
    assert result['oom_events'][0]['requested_bytes'] == 80
    assert result['event_size_peak_by_phase_bytes'] == {'preexisting':100,'forward_or_other':71}


def test_incomplete_trace_never_makes_a_peak_claim():
    ev = [dict(action='free_requested', addr=8, size=3)]
    for cap in (1, 10):
        result = _summarize({}, ev, peak=10, ending=0, limit=cap)
        assert result['coverage_issues'] and 'event_size_peak_bytes' not in result
    mismatch = _summarize({}, [dict(action='alloc', addr=8, size=3)], peak=512, ending=512, limit=10)
    assert not mismatch['peak_and_ending_counters_match']
    assert mismatch['allocator_minus_event_peak_bytes'] == 509
