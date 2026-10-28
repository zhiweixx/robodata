"""Keep native tick progress in logs, separate from generated-program stdout.

Only RoboDojo's take_action call is wrapped. Agent-authored print statements,
tool errors, and other native diagnostics keep their original public output.
The episode runner archives the complete native stdout separately; callers
without an archive still send the progress to the process's original stdout.
This does not compact history or limit any output, actions, or model calls.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from contextvars import ContextVar
import io
from pathlib import Path
import re
import sys


_ARCHIVE = ContextVar("robodojo_native_stdout_archive", default=None)
# Exact native EvalEnv progress record, with or without its green ANSI color.
# A lookbehind recognizes both newline and carriage-return record boundaries;
# a similar-looking sentence embedded in a diagnostic must not be removed.
_TICK_PROGRESS = re.compile(
    r"(?<![^\r\n])env[0-9]+ step: "
    r"(?:\x1b\[92m[0-9]+ / [0-9]+\x1b\[0m|[0-9]+ / [0-9]+)"
    r"(?:\r\n|\r|\n|$)"
)


@contextmanager
def archive_native_stdout(path):
    """Archive raw native stdout for one program, outside model feedback.

Exclusive creation protects prior execution evidence. The ContextVar scopes
the destination to this execution; it is never exposed as a public tool.
"""
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        token = _ARCHIVE.set(stream)
        try:
            yield
        finally:
            _ARCHIVE.reset(token)


@contextmanager
def route_native_progress():
    """Route only native progress away from ASPIRE's public stdout capture."""
    public = sys.stdout
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            yield
    finally:
        # Even a failed native action keeps its stdout and restores the stream.
        raw = captured.getvalue()
        archive = _ARCHIVE.get()
        if archive is not None:
            archive.write(raw)
            archive.flush()
        else:
            # Pilot/smoke callers without a per-program archive retain progress
            # in their process/Slurm log, not in the model's StringIO capture.
            process_log = sys.__stdout__
            if process_log is not None:
                process_log.write("".join(m.group() for m in _TICK_PROGRESS.finditer(raw)))
                process_log.flush()
        public.write(_TICK_PROGRESS.sub("", raw))
