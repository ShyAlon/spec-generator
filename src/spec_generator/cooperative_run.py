"""Opt-in same-host producer execution. No marker path is inferred from input.

Trusted runner supplies the fence and usage log for its bound job. This adapter
holds writer lifetime and preserves late accepted-call usage; it does not wire
CLI, launch a process, cancel a call, or declare delivery quiescence.
"""
from contextlib import contextmanager
from .stop_fence import stopped_usage_sink


@contextmanager
def producer_execution(fence,usage_log):
    from .model_config import _usage_of
    def stopped(response,*,stage,provider,model):
        usage=_usage_of(response)
        # No zero usage assumption for an accepted call with missing metadata.
        if not usage:raise RuntimeError('accepted stopped-call usage unavailable')
        usage_log.record(stage,provider,model,usage)
    with fence.running() as generation,stopped_usage_sink(stopped):
        yield generation


@contextmanager
def cli_execution(fence,usage_log):
    """Private binding for an already-authorized same-host CLI invocation.

    No environment flag or producer-supplied path creates authority. The
    caller must provide its job-bound fence and hold this scope for all writes.
    """
    with producer_execution(fence,usage_log) as generation:
        token=_cli_fence.set((fence,generation))
        try:yield generation
        finally:_cli_fence.reset(token)


def cli_stop_binding():
    return _cli_fence.get()


from contextvars import ContextVar
_cli_fence=ContextVar('producer_cli_stop_binding',default=None)
