"""Private SDK-client installation supplied by a trusted worker.

No endpoint, credential, transport, or config default is created here. Clients
remain owned by the runner through real writer quiescence, not scope exit.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import httpx
from .attempt_admission import AdmissionUnavailable

_binding=ContextVar('producer_wire_clients',default=None)

@contextmanager
def wire_clients(gateway,client_factory):
    token=_binding.set((gateway,client_factory,{}))
    try:yield
    finally:_binding.reset(token)


def model_clients(*,stage,provider,model):
    binding=_binding.get()
    if binding is None:return None
    if provider not in ('deepseek','openai'):
        raise AdmissionUnavailable('provider has no reviewed SDK wire installation')
    from .provider_transport import AdmittedTransport,AsyncAdmittedTransport
    gateway,factory,_=binding
    sync,async_client=factory(stage=stage,provider=provider,model=model,gateway=gateway)
    for client,client_type,transport_type in ((sync,httpx.Client,AdmittedTransport),(async_client,httpx.AsyncClient,AsyncAdmittedTransport)):
        transport=getattr(client,'_transport',None)
        if (not isinstance(client,client_type) or not isinstance(transport,transport_type)
            or transport.gateway is not gateway
            or (transport.stage,transport.provider,transport.model)!=(stage,provider,model)
            or any(value is not None for value in getattr(client,'_mounts',{}).values())):
            raise AdmissionUnavailable('job-bound SDK clients required without alternate mounts')
    return {'http_client':sync,'http_async_client':async_client,'stream_usage':True}


def register_model(chat_model,*,stage,provider,model):
    binding=_binding.get()
    if binding is not None:binding[2][id(chat_model)]=(chat_model,stage,provider,model)
    return chat_model


def wire_model(chat_model,*,stage,provider,model):
    binding=_binding.get()
    if binding is None:return False
    item=binding[2].get(id(chat_model))
    if item is None or item[0] is not chat_model or item[1:]!=(stage,provider,model):
        raise AdmissionUnavailable('SDK model is not bound to this worker stage')
    return True


def validate_provider(provider):
    if _binding.get() is not None and provider not in ('deepseek','openai'):
        raise AdmissionUnavailable('provider has no reviewed SDK wire installation')
