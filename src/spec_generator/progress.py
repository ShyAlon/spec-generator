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
    status:    'started' | 'completed' | 'failed'
    ts:        epoch seconds when the event fired
    elapsed_s: seconds the stage took (completed/failed only)
    detail:    optional context ('turn 4', 'retrying after schema feedback')
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    label: str
    status: str          # started | completed | failed
    ts: float
    elapsed_s: float | None = None
    detail: str | None = None


class ProgressBus:
    """Synchronous fan-out. A broken consumer must never crash a run, so
    subscriber errors are swallowed by design (progress is observability,
    not control flow)."""

    def __init__(self) -> None:
        self._subscribers: list = []

    def subscribe(self, fn) -> None:
        self._subscribers.append(fn)

    def emit(self, event: ProgressEvent) -> None:
        for fn in self._subscribers:
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - observability never kills the run
                pass


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


def stderr_printer():
    """The CLI consumer. Plain stderr lines; no TUI dependencies."""
    def _print(ev: ProgressEvent) -> None:
        detail = f" ({ev.detail})" if ev.detail else ""
        if ev.status == "started":
            print(f"  -> {ev.label}{detail}...", file=sys.stderr)
        elif ev.status == "completed":
            print(f"  ok {ev.label}{detail} [{ev.elapsed_s:.1f}s]", file=sys.stderr)
        else:
            print(f"  !! {ev.label}{detail} FAILED after {ev.elapsed_s:.1f}s", file=sys.stderr)
    return _print
