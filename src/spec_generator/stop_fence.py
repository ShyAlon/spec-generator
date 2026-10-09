"""Durable cooperative stop fence, independent from the long execution lock.

No signals, graph mutation, artifact deletion or provider cancellation. This is
an uninstalled primitive; registry/publication/checkpointer wiring is separate.
"""
import json,os,threading,fcntl
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar

class StopRequested(RuntimeError):pass
class SessionWriterBusy(RuntimeError):pass
_check=ContextVar('cooperative_stop_check',default=None)
_current=ContextVar('cooperative_stop_fence',default=None)
_stopped_usage=ContextVar('cooperative_stopped_usage_sink',default=None)

@contextmanager
def execution_fence(check):
    token=_check.set(check)
    try:check();yield
    finally:_check.reset(token)

def check_execution():
    check=_check.get()
    if check is not None:check()

class StopFence:
    def __init__(self,path,*,execution_anchor=None):
        self.path=Path(path);self.lock=threading.RLock()
        self.generation=1;self.requested=False;self.active=0;self.recovery_pending=False;self._guard_depth=threading.local();self._writer_file=None
        if execution_anchor is None and self.path.name=='stop-request.json' and self.path.parent.name=='interview':
            anchor_path=self.path.parent.parent/'execution-generation.json'
            if anchor_path.exists() or anchor_path.is_symlink():
                raise ValueError('explicit installed execution binding required for lineage namespace')
        self._execution_anchor=execution_anchor;self._anchor_root=None;self._anchor_digest=None
        if execution_anchor is not None:
            from vmodel_contracts.execution_anchor import read_execution_anchor
            if self.path.name!='stop-request.json' or self.path.parent.name!='interview':
                raise ValueError('canonical execution control marker required')
            self._anchor_root=self.path.parent.parent
            receipt=read_execution_anchor(self._anchor_root,expected=execution_anchor)
            self._anchor_digest=receipt['document_sha256'];self.generation=execution_anchor.generation
        if self.path.exists():
            data=json.loads(self.path.read_text())
            self.generation=data['generation'];self.requested=data['requested'];self.recovery_pending=self.requested
            if type(self.generation) is not int or self.generation<1 or type(self.requested) is not bool:
                raise ValueError('invalid durable stop fence')
            if execution_anchor is not None and (self.requested is not True or self.generation!=execution_anchor.generation+1):
                raise ValueError('Stop marker differs from installed execution generation')

    @contextmanager
    def marker_guard(self):
        # Local RLock plus a separate short cross-process publication lock.
        # Never hold this over a provider call. Reentrant calls reuse the fd.
        with self.lock:
            depth=getattr(self._guard_depth,'depth',0)
            if depth:
                self._guard_depth.depth=depth+1
                try:yield
                finally:self._guard_depth.depth=depth
                return
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.with_suffix('.marker-lock').open('a') as file:
                fcntl.flock(file,fcntl.LOCK_EX)
                self._guard_depth.depth=1
                try:
                    self._refresh_marker()
                    yield
                finally:
                    self._guard_depth.depth=0
                    fcntl.flock(file,fcntl.LOCK_UN)

    def _refresh_marker(self):
        if self._execution_anchor is not None:
            from vmodel_contracts.execution_anchor import read_execution_anchor
            receipt=read_execution_anchor(self._anchor_root,expected=self._execution_anchor)
            if receipt['document_sha256']!=self._anchor_digest:
                raise ValueError('immutable execution anchor bytes changed')
        if not self.path.exists():
            if self._execution_anchor is not None and self.requested:
                raise ValueError('installed execution Stop marker disappeared')
            return
        data=json.loads(self.path.read_text())
        if type(data.get('generation')) is not int or data['generation']<1 or data.get('requested') is not True:
            raise ValueError('invalid durable stop fence')
        if self._execution_anchor is not None and data['generation']!=self._execution_anchor.generation+1:
            raise ValueError('Stop marker differs from installed execution generation')
        if not self.requested:
            # Another registry/process's marker does not prove quiescence.
            self.recovery_pending=True
        self.generation=max(self.generation,data['generation']);self.requested=True

    def _writer_enter(self):
        if self.active==0:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            file=self.path.with_suffix('.writer-lock').open('a')
            try:fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                file.close();raise SessionWriterBusy('another session writer remains active')
            self._writer_file=file
        self.active+=1

    def _writer_exit(self):
        with self.lock:
            self.active-=1
            if self.active==0 and self._writer_file is not None:
                fcntl.flock(self._writer_file,fcntl.LOCK_UN)
                self._writer_file.close();self._writer_file=None

    def _external_writer_active(self):
        if self.active:return True
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix('.writer-lock').open('a') as file:
            try:fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return True
            fcntl.flock(file,fcntl.LOCK_UN)
            return False

    @property
    def state(self):
        with self.marker_guard():return 'STOP_REQUESTED' if self.requested and (self.active or self.recovery_pending) else 'STOPPED' if self.requested else None

    def request_stop(self):
        with self.marker_guard():
            if not self.requested:
                generation=self.generation+1
                self.path.parent.mkdir(parents=True,exist_ok=True)
                temp=self.path.with_suffix('.stop-tmp')
                with temp.open('w') as f:
                    json.dump({'generation':generation,'requested':True},f);f.flush();os.fsync(f.fileno())
                temp.replace(self.path)
                fd=os.open(self.path.parent,os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
                self.generation=generation;self.requested=True
            if not self.active and self._external_writer_active():self.recovery_pending=True
            return self.state

    def writer_quiescent(self):
        """Same-host writer-lock evidence, not provider-ledger settlement."""
        with self.marker_guard():
            return not self.active and not self._external_writer_active()

    def confirm_quiescent_after_recovery(self, *, no_live_worker):
        # Caller must prove no prior process/thread can still write. Loading the
        # marker by itself does not prove this and never unlocks Delete.
        with self.marker_guard():
            if no_live_worker is not True or self.active or self._external_writer_active():raise StopRequested('worker quiescence unverified')
            self.recovery_pending=False
            return self.state

    def check(self,generation):
        with self.marker_guard():
            if self.requested or generation!=self.generation:raise StopRequested('session stop requested')

    @contextmanager
    def running(self):
        with self.lock:
            generation=self.generation;self.check(generation);self._writer_enter()
        try:
            def publish(call):
                with self.marker_guard():
                    self.check(generation);return call()
            token=_publish.set(publish)
            current_token=_current.set(self)
            try:
                with execution_fence(lambda:self.check(generation)):yield generation
            finally:
                _publish.reset(token);_current.reset(current_token)
        finally:
            self._writer_exit()

_publish=ContextVar('cooperative_stop_publish',default=None)

def guarded_publish(call):
    guard=_publish.get()
    if guard is None:
        check_execution();return call()
    return guard(call)


def track_inflight_worker():
    """Register before handing work to a deadline thread; release on its exit.

    A graph timeout does not prove its provider thread has stopped. Do not check
    Stop here: an accepted response must still settle its usage after Stop.
    """
    fence=_current.get()
    if fence is None:return lambda:None
    with fence.lock:fence._writer_enter()
    released=False
    def release():
        nonlocal released
        with fence.lock:
            if not released:fence._writer_exit();released=True
    return release


def record_stopped_usage(response,*,stage,provider,model):
    sink=_stopped_usage.get()
    if sink is None:raise StopRequested('stopped response usage sink unavailable')
    sink(response,stage=stage,provider=provider,model=model)

@contextmanager
def stopped_usage_sink(sink):
    """Usage is intentionally outside result publication fencing."""
    token=_stopped_usage.set(sink)
    try:yield
    finally:_stopped_usage.reset(token)
