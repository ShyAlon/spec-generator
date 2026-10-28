"""Private pre-provider seam. Cloud mode has no permissive fallback.

The gateway must be bound to a verified durable job outside producer code.
No prices, owner limits or token bounds are inferred here.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import os
import uuid

_socket_gateway=None
_socket_fd=None
_gateway=ContextVar('attempt_admission_gateway',default=None)
_required=ContextVar('attempt_admission_required',default=False)

class AdmissionUnavailable(RuntimeError):pass

class AdmissionRefused(AdmissionUnavailable):
    """The gateway answered with a clean refusal frame (the channel is still in sync).
    reason is the server's own text ("AdmissionDenied: ..."), or '' when an older server gave none."""
    def __init__(self,message,reason=''):
        super().__init__(message);self.reason=reason

@contextmanager
def admission_gateway(gateway,*,required=False):
    token=_gateway.set(gateway);required_token=_required.set(required)
    try:yield
    finally:
        _required.reset(required_token);_gateway.reset(token)

def _cloud():
    return _required.get() or os.environ.get('VMODEL_TENANT_OWNER') is not None or os.environ.get('INTERVIEW_ENGINE_TENANTS')=='1'

def _reserve(stage,provider,model,messages,max_output_tokens):
    gateway=_gateway.get()
    if gateway is None and os.environ.get('VMODEL_ADMISSION_FD'):
        global _socket_gateway,_socket_fd
        if _socket_gateway is not None and _socket_fd!=os.environ['VMODEL_ADMISSION_FD']:
            raise AdmissionUnavailable('admission descriptor changed during worker lifetime')
        if _socket_gateway is None:
            from .admission_client import SocketAdmissionGateway
            try:
                _socket_gateway=SocketAdmissionGateway(int(os.environ['VMODEL_ADMISSION_FD']))
                _socket_fd=os.environ['VMODEL_ADMISSION_FD']
            except Exception as exc:raise AdmissionUnavailable('invalid inherited admission descriptor') from exc
        gateway=_socket_gateway
    if gateway is None:raise AdmissionUnavailable('cloud provider admission is unavailable')
    # Gateway computes/rejects exact serialized input bound and pinned price.
    attempt=gateway.reserve(attempt_id=uuid.uuid4().hex,stage=stage,provider=provider,
        model=model,messages=messages,max_output_tokens=max_output_tokens)
    if not attempt:raise AdmissionUnavailable('cloud provider attempt was not admitted')
    gateway.start(attempt)
    return gateway,attempt

def admitted_invoke(call,*,stage,provider,model,messages,max_output_tokens):
    from .stop_fence import check_execution,StopRequested,record_stopped_usage
    check_execution()
    if not _cloud():
        response=call()
        try:check_execution()
        except StopRequested:
            record_stopped_usage(response,stage=stage,provider=provider,model=model)
            raise
        return response
    gateway,attempt=_reserve(stage,provider,model,messages,max_output_tokens)
    try:
        response=call()
        gateway.settle(attempt,response)
        check_execution()
        return response
    except BaseException:
        try:gateway.uncertain(attempt)
        except Exception as error:
            _uncertainty_write_failed(error)
        raise


def admitted_stream(call,*,stage,provider,model,messages,max_output_tokens):
    from .stop_fence import check_execution,StopRequested,record_stopped_usage
    from types import SimpleNamespace
    check_execution()
    cloud=_cloud()
    gateway,attempt=_reserve(stage,provider,model,messages,max_output_tokens) if cloud else (None,None)
    stopped=False;usage={};metadata={}
    try:
        # Once accepted, drain the stream even when Stop arrives. Never yield
        # additional chunks into result/partial sinks. A caller timeout does
        # not end the pump's writer lifetime.
        for chunk in call():
            usage.update(getattr(chunk,'usage_metadata',None) or {})
            metadata.update(getattr(chunk,'response_metadata',None) or {})
            try:check_execution()
            except StopRequested:stopped=True
            if not stopped:yield chunk
        try:check_execution()
        except StopRequested:stopped=True
        if stopped:
            if not cloud:
                record_stopped_usage(SimpleNamespace(usage_metadata=usage,response_metadata=metadata),
                    stage=stage,provider=provider,model=model)
            raise StopRequested('accepted stream finished; stopped result discarded')
    finally:
        # Messages wrapper cannot prove raw terminal usage; wire replacement
        # must settle it. Missing usage stays held, never refunded or zeroed.
        if cloud:gateway.uncertain(attempt)


def checked_model_invoke(chat_model,messages,*,stage,provider,model,max_output_tokens):
    from .wire_binding import wire_model
    if wire_model(chat_model,stage=stage,provider=provider,model=model):
        from .stop_fence import check_execution
        check_execution()
        try:
            from .child_execution import invocation
            with invocation(stage):response=chat_model.invoke(messages)
        except Exception as exc:
            # OpenAI wraps transport control exceptions as connection errors.
            # Preserve Stop identity rather than presenting a provider outage.
            from .stop_fence import StopRequested
            cause=exc;seen=set()
            while cause is not None and id(cause) not in seen:
                if isinstance(cause,StopRequested):raise cause
                seen.add(id(cause));cause=cause.__cause__ or cause.__context__
            raise
        check_execution()
        return response
    return admitted_invoke(lambda:chat_model.invoke(messages),stage=stage,provider=provider,
        model=model,messages=messages,max_output_tokens=max_output_tokens)


def checked_model_stream(chat_model,messages,*,stage,provider,model,max_output_tokens):
    from .wire_binding import wire_model
    if not wire_model(chat_model,stage=stage,provider=provider,model=model):
        yield from admitted_stream(lambda:chat_model.stream(messages),stage=stage,provider=provider,
            model=model,messages=messages,max_output_tokens=max_output_tokens)
        return
    from .stop_fence import check_execution,StopRequested
    check_execution();stopped=False
    # The SDK transport owns exactly one reservation and raw terminal settlement.
    # Drain accepted calls while suppressing late chunks. Never stack a messages
    # reservation or mark already-settled raw usage uncertain at scope exit.
    from .child_execution import invocation
    with invocation(stage):
        for chunk in chat_model.stream(messages):
            try:check_execution()
            except StopRequested:stopped=True
            if not stopped:yield chunk
    check_execution()


def _uncertainty_write_failed(error):
    # Evidence must never replace the original provider/accounting failure.
    try:
        import logging
        logging.getLogger(__name__).warning('attempt.uncertainty_write_failed category=%s',type(error).__name__)
    except Exception:pass
