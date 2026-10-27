"""Structured progress events - one stream, many consumers.

The CLI printer below is ONE consumer. delivery-api / the web UI will consume
the same event stream later (the founder watches a live step list, not a
spinner). The shape is deliberately identical across repos (spec-generator
carries the same module); when events cross a process boundary to the UI, the
shape promotes into vmodel-contracts with a minor version bump (ADR-0002).
Until then: intra-module, documented in ARCHITECTURE.md 'Progress events'.

Event shape:
    stage:     machine name, stable per pipeline step (e.g. 'distill_intent')
    label:     human-readable present-tense phrase ('Distilling intent model')
    status:    'started' | 'item' | 'completed' | 'failed'
    ts:        epoch seconds when the event fired
    elapsed_s: seconds the stage took (completed/failed only)
    detail:    optional context ('turn 4', 'retrying after schema feedback')
"""
from __future__ import annotations

from . import cli_io
import time
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from . import model_config as _mc
from vmodel_contracts.runtime import format_console_line


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    label: str
    status: str          # started | item | completed | failed
    ts: float
    elapsed_s: float | None = None
    detail: str | None = None


class ProgressBus:
    """Synchronous fan-out. A broken consumer must never crash a run, so
    subscriber errors are counted and safely reported (progress is observability,
    not control flow)."""

    def __init__(self) -> None:
        self._subscribers: list = []
        self.subscriber_failures: dict[int, int] = {}
        self._last_warning: dict[int, float] = {}

    def subscribe(self, fn) -> None:
        self._subscribers.append(fn)

    def emit(self, event: ProgressEvent) -> None:
        from .stop_fence import check_execution,StopRequested
        try:check_execution()
        except StopRequested:return
        for index, fn in enumerate(self._subscribers):
            try:
                fn(event)
            except Exception as error:  # observability never kills the run
                self.subscriber_failures[index] = self.subscriber_failures.get(index, 0) + 1
                now = time.monotonic()
                if index not in self._last_warning or now - self._last_warning[index] >= 60:
                    self._last_warning[index] = now
                    # No event payload, exception text or callable repr in operator logs.
                    stage = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(event.stage))[:80]
                    try:
                        _logger.warning('progress.subscriber_failed subscriber=%d stage=%s category=%s count=%d',
                                        index, stage, type(error).__name__, self.subscriber_failures[index])
                    except Exception:
                        pass  # even a broken diagnostic handler must stay nonfatal


@contextmanager
def track(bus: ProgressBus | None, stage: str, label: str, detail: str | None = None):
    """Wrap a pipeline step: emits started, then completed/failed with elapsed."""
    t0 = time.time()
    if bus is not None:
        bus.emit(ProgressEvent(stage, label, "started", t0, None, detail))
    try:
        yield
    except Exception:
        if bus is not None:
            bus.emit(ProgressEvent(stage, label, "failed", time.time(), time.time() - t0, detail))
        raise
    if bus is not None:
        bus.emit(ProgressEvent(stage, label, "completed", time.time(), time.time() - t0, detail))


def _hhmmss(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def stderr_printer():
    """The CLI consumer. Plain stderr lines; no TUI dependencies. A per-slot
    item prints when it BEGINS (so a hung item is visible live) and its line
    gains the end time when it finishes - in place on a terminal, as a second
    line when stderr is redirected."""
    open_item = {"detail": None}

    def _print(ev: ProgressEvent) -> None:
        with _mc._CONSOLE_LOCK:  # never interleave with the heartbeat thread
            _print_inner(ev)

    def _print_inner(ev: ProgressEvent) -> None:
        detail = f" ({ev.detail})" if ev.detail else ""
        if ev.status == "started":
            print(format_console_line(f"  -> {ev.label}{detail}...", ev.ts, stream=cli_io.stderr()), file=cli_io.stderr())
        elif ev.status == "item_started":
            line = f"     - [{_hhmmss(ev.ts)}] {ev.detail}"
            if cli_io.stderr().isatty():
                print(line, end="", file=cli_io.stderr(), flush=True)
            else:
                print(line, file=cli_io.stderr())
            open_item["detail"] = ev.detail
        elif ev.status == "item":
            if ev.elapsed_s is not None:
                line = f"     - [{_hhmmss(ev.ts - ev.elapsed_s)}-{_hhmmss(ev.ts)}] {ev.detail}"
            else:
                line = f"     - {ev.detail}"
            if open_item["detail"] is not None and cli_io.stderr().isatty():
                print("\r" + line + "  ", file=cli_io.stderr())  # overwrite the running line
            else:
                print(line, file=cli_io.stderr())
            open_item["detail"] = None
        elif ev.status == "completed":
            print(format_console_line(f"  ok {ev.label}{detail} [{ev.elapsed_s:.1f}s]", ev.ts, "success", cli_io.stderr()), file=cli_io.stderr())
        else:
            print(format_console_line(f"  !! {ev.label}{detail} FAILED after {ev.elapsed_s:.1f}s", ev.ts, "error", cli_io.stderr()), file=cli_io.stderr())
    return _print

