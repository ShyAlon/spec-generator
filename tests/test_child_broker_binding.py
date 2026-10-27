import asyncio
import socket
import pytest
from requi_child_contract.broker import BrokerClient,BrokerTransport,AsyncBrokerTransport
from spec_generator import wire_binding as module


def test_explicit_broker_binding_default_unchanged():
    parent,child=socket.socketpair();client=BrokerClient(child.fileno(),next_call_id=lambda stage:'call-1')
    try:
        assert module.model_clients(stage='intent',provider='deepseek',model='m') is None
        with module.broker_clients(client):
            pair=module.model_clients(stage='intent',provider='deepseek',model='m')
            assert isinstance(pair['http_client']._transport,BrokerTransport)
            assert isinstance(pair['http_async_client']._transport,AsyncBrokerTransport)
            obj=object();module.register_model(obj,stage='intent',provider='deepseek',model='m')
            assert module.wire_model(obj,stage='intent',provider='deepseek',model='m')
            with pytest.raises(module.AdmissionUnavailable):module.wire_model(obj,stage='wrong',provider='deepseek',model='m')
            pair['http_client'].close();asyncio.run(pair['http_async_client'].aclose())
        assert module.model_clients(stage='intent',provider='deepseek',model='m') is None
    finally:client.close();parent.close();child.close()
