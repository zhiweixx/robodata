"""Resample a native joint path without exceeding its segment speeds or tick cap."""
from __future__ import annotations
import math
import numpy as np


def retime_native_path(path, measured_start, *, native_dt_s, control_dt_s,
                       max_tick_delta_rad=.045, start_tolerance_rad=.001):
    path=np.asarray(path,dtype=np.float64)
    start=np.asarray(measured_start,dtype=np.float64)
    if path.ndim!=2 or not len(path) or path.shape[1:]!=start.shape or not np.isfinite(path).all() or not np.isfinite(start).all():
        raise ValueError("native path and measured start must be finite nonempty matching joint arrays")
    if any(not math.isfinite(x) or x<=0 for x in (native_dt_s,control_dt_s,max_tick_delta_rad,start_tolerance_rad)):
        raise ValueError("retiming intervals and tolerances must be finite and positive")
    start_gap=float(np.max(np.abs(path[0]-start)))
    if start_gap>start_tolerance_rad:
        raise ValueError(f"native path start differs from measured joints by {start_gap:.9g}rad (limit{start_tolerance_rad})")
    native_deltas=np.max(np.abs(np.diff(path,axis=0)),axis=1)
    speed_cap=max_tick_delta_rad/control_dt_s
    segment_times=np.maximum(native_dt_s,native_deltas/speed_cap)
    knots=path
    times=np.concatenate(([0.],np.cumsum(segment_times)))
    prefix_duration=0.
    if start_gap>1e-12:
        # Explicit bounded numerical-start connector; reject substantive start
        # discontinuities above instead of manufacturing an unchecked bridge.
        prefix_duration=max(native_dt_s,start_gap/speed_cap)
        times=np.concatenate(([0.],times+prefix_duration))
        knots=np.vstack((start,path))
    duration=float(times[-1])
    count=max(1,int(math.ceil(duration/control_dt_s)))
    sample_times=np.minimum(np.arange(1,count+1)*control_dt_s,duration)
    sampled=np.column_stack([np.interp(sample_times,times,knots[:,j]) for j in range(path.shape[1])])
    observed_delta=float(np.max(np.abs(np.diff(np.vstack((start,sampled)),axis=0))))
    if observed_delta>max_tick_delta_rad+1e-9:
        raise ValueError("retimed path violated its mathematically bounded per-tick increment")
    return sampled,{
        "retiming":"per-native-segment nonaccelerating duration; uniform control sampling",
        "native_start_gap_rad":start_gap,"native_start_tolerance_rad":start_tolerance_rad,
        "native_peak_knot_delta_rad":float(native_deltas.max()) if len(native_deltas) else 0.,
        "native_duration_s":max(0,len(path)-1)*native_dt_s,
        "retimed_duration_s":duration,"control_duration_s":count*control_dt_s,
        "numerical_start_connector_s":prefix_duration,
        "max_tick_delta_rad":observed_delta,"tick_delta_cap_rad":max_tick_delta_rad,
        "slowed_native_segments":int(np.count_nonzero(segment_times>native_dt_s)),
    }
