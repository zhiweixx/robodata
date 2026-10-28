"""Preserve public measured execution errors in ASPIRE program call receipts.

The upstream validator, namespace, tool invocation, exception handling and program
execution are inherited unchanged. Only the already-public return value's receipt
is enriched. No additional observation or robot action is performed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from pathlib import Path
from types import MappingProxyType

from cap_harness.contracts import ExecutionResult
from cap_harness.errors import ApiError
from cap_harness.registry import ToolRegistry
from cap_harness.runtime import ProgramExecutor


_SCALARS = frozenset({
    "position_error_m", "position_tolerance_m", "orientation_error_rad",
    "orientation_tolerance_rad", "joint_tolerance_rad", "tolerance",
    "max_steps", "stall_window_steps",
})
_COUNTS = frozenset({"waypoint_count"})
_ARM_MEASUREMENTS = frozenset({
    "joint_error_rad", "target_positions", "measured_positions", "targets", "measured",
})
_FLAGS = frozenset({"trajectory_executed", "joint_target_converged"})
_COMPLETIONS = frozenset({"target_reached", "stable_aperture_stall"})
_ARMS = frozenset({"left", "right"})


def _number(value):
    # Do not stringify unknown values or traverse arbitrary nested diagnostics.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def measured_details(values):
    """Strict schema for existing public measurements, never evaluator state.

Even a trusted-looking key cannot carry arbitrary strings, nested mappings or
arrays. Unknown keys and arm names are discarded rather than recursively copied.
"""
    if not isinstance(values, Mapping):
        return {}
    result = {}
    for key, value in values.items():
        if key in _SCALARS:
            number = _number(value)
            if number is not None:
                result[key] = number
        elif key in _COUNTS:
            # Native trajectory lengths are Python integers. Do not coerce
            # booleans, floating-point values or nested diagnostics into counts.
            if type(value) is int and value >= 0:
                result[key] = value
        elif key in _ARM_MEASUREMENTS and isinstance(value, Mapping):
            arms = {arm: number for arm, measurement in value.items()
                    if arm in _ARMS and (number := _number(measurement)) is not None}
            if arms:
                result[key] = arms
        elif key in _FLAGS and type(value) is bool:
            result[key] = value
        elif key == "completion" and isinstance(value, str) and value in _COMPLETIONS:
            result[key] = value
    return result


class ReceiptProgramExecutor(ProgramExecutor):
    """Upstream ASPIRE executor with filtered, measured motion-error receipts."""

    def _invoke_tool(self, name, function, args, kwargs):
        public_result = super()._invoke_tool(name, function, args, kwargs)
        if isinstance(public_result, ExecutionResult):
            # super already called ExecutionResult.public_view() and appended
            # exactly one call trace. Enrich that trace; return its same value.
            trace = self._calls[-1]
            summary = dict(trace.outputs_summary or {})
            diagnostics = measured_details(public_result.diagnostics)
            if diagnostics:
                summary["measured_diagnostics"] = diagnostics
            error = public_result.error
            message = trace.error_message
            if isinstance(error, ApiError):
                message = error.message
                summary["error_recoverable"] = error.recoverable
                summary["error_details"] = measured_details(error.details)
            self._calls[-1] = replace(
                trace, outputs_summary=MappingProxyType(summary), error_message=message)
        return public_result


def run_program_with_receipts(*, api, program_path, extension_allowlist=None, recorder=None):
    """Mirror upstream run_program_with_api setup, using its executor subclass."""
    path = Path(program_path).expanduser().resolve()
    source = path.read_text(encoding="utf-8")
    if not source.strip():
        raise ValueError("program file must contain non-empty Python source")
    registry = (ToolRegistry(public_extension_allowlist=extension_allowlist)
                if extension_allowlist is not None else ToolRegistry())
    tools = api.register_tools(registry)
    if recorder is not None:
        recorder.save_program(path)
    return ReceiptProgramExecutor(tools, recorder=recorder).execute_program(source)
