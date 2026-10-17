"""Framed generator progress on the existing subprocess pipe; prose is diagnostic."""
from __future__ import annotations
from datetime import datetime, timezone
from . import cli_io
import os
import threading
from uuid import uuid4
from vmodel_contracts.progress_signal import ProgressSignal

PREFIX = "VMODEL_PROGRESS_JSONL:"
active_sink: StructuredProgress | None = None
STAGE_PHASE = {"needs":"needs", "srs":"srs", "architecture":"architecture", "items":"specs", "tests":"tests", "validation":"validation"}

class StructuredProgress:
    def __init__(self, task_id: str, session_id: str, fallback_phase: str):
        self.task_id, self.session_id, self.fallback_phase=task_id,session_id,fallback_phase
        self._lock=threading.Lock()
        self._cursor=0
        self._enabled=bool(os.environ.get("VMODEL_PROGRESS_TASK_ID") and
                           os.environ.get("VMODEL_PROGRESS_SESSION_ID")==session_id)

    def emit(self, stage: str, label: str, status: str, detail: str | None = None):
        with self._lock:
            self._cursor+=1
            signal=ProgressSignal(event_id=str(uuid4()),cursor=self._cursor,task_id=self.task_id,
                session_id=self.session_id,phase_key=self.fallback_phase,
                stage=stage,label=label or stage,status=status if status!="item" else "item_completed",
                detail=detail,timestamp=datetime.now(timezone.utc))
            cli_io.stdout().write(PREFIX+signal.model_dump_json()+"\n")
            cli_io.stdout().flush()

    def subscriber(self, event):
        if self._enabled:self.emit(event.stage,event.label,event.status,event.detail)
