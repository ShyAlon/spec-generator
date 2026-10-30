"""Private per-job runtime state, never process-wide environment mutation."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass,field
from threading import RLock
from vmodel_contracts.runtime import AdaptiveCallWatchdog

_current=ContextVar('producer_job_runtime',default=None)

@dataclass
class JobRuntime:
    models:dict
    llm_log:str
    console:object
    progress:object=None
    settings:dict=field(default_factory=dict)
    calls:int=0
    diagnostics:dict=field(default_factory=dict)
    lock:object=field(default_factory=RLock)
    watchdog:object=field(default_factory=lambda:AdaptiveCallWatchdog(multiplier=8.0,floor_s=240.0))
    checkpoint_factory:object=field(default=None,kw_only=True)
    proposal_sink:object=field(default=None,kw_only=True)
    documents:object=field(default=None,kw_only=True)
    worker_binding:tuple|None=field(default=None,init=False)

    def bind_worker(self,*,owner_id,task_id,job_id,generation,session_root):
        from pathlib import Path
        root=Path(session_root);sink=Path(self.llm_log)
        if (not root.is_absolute() or root.resolve()!=root or not sink.is_absolute()
            or sink.resolve()!=sink or not sink.is_relative_to(root)):
            raise ValueError('private owned runtime log sink required')
        binding=(owner_id,task_id,job_id,generation,str(root))
        with self.lock:
            if self.worker_binding is not None and self.worker_binding!=binding:
                raise ValueError('runtime already belongs to another worker job')
            self.worker_binding=binding

@contextmanager
def job_runtime(runtime):
    if not isinstance(runtime,JobRuntime):raise TypeError('explicit job runtime required')
    from .model_config import StageModel
    if not runtime.models or any(not isinstance(value,StageModel) for value in runtime.models.values()):
        raise ValueError('explicit stage models required')
    if not runtime.llm_log or not callable(runtime.console):raise ValueError('private telemetry sinks required')
    token=_current.set(runtime)
    try:yield runtime
    finally:_current.reset(token)


def current_runtime():return _current.get()
