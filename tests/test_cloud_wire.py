"""Cloud child wire setup: descriptor-gated binding, fail-closed, ollama-only."""
import os
import socket

import httpx
import pytest

from spec_generator.cloud_wire import admitted_child_run, _ollama_wire_factory
from spec_generator.wire_binding import model_clients


def test_no_descriptor_is_a_noop(monkeypatch):
    monkeypatch.delenv('VMODEL_ADMISSION_FD', raising=False)
    with admitted_child_run():
        assert model_clients(stage='needs', provider='ollama', model='m') is None
    assert model_clients(stage='needs', provider='ollama', model='m') is None


def test_invalid_descriptor_fails_closed(monkeypatch):
    monkeypatch.setenv('VMODEL_ADMISSION_FD', '999999')
    with pytest.raises(OSError):
        with admitted_child_run():
            pytest.fail('an invalid descriptor must abort before the run starts')


def test_binding_active_during_run_and_reset_after(monkeypatch):
    left, right = socket.socketpair()
    try:
        monkeypatch.setenv('VMODEL_ADMISSION_FD', str(left.fileno()))
        with admitted_child_run():
            clients = model_clients(stage='needs', provider='ollama', model='m')
            assert clients is not None
            transport = clients['http_client']._transport
            from spec_generator.provider_transport import AdmittedTransport
            assert isinstance(transport, AdmittedTransport)
            assert (transport.stage, transport.provider, transport.model) == ('needs', 'ollama', 'm')
            clients['http_client'].close()
        assert model_clients(stage='needs', provider='ollama', model='m') is None
    finally:
        left.close()
        right.close()


def test_factory_refuses_non_ollama_provider():
    factory = _ollama_wire_factory()
    with pytest.raises(ValueError, match='Ollama'):
        factory(stage='needs', provider='deepseek', model='m', gateway=None)


def test_fixed_origin_guard_rejects_foreign_destination():
    sync, _ = _ollama_wire_factory()(stage='needs', provider='ollama', model='m', gateway=None)
    guard = sync._transport.inner  # the fixed-origin layer inside the admitted transport
    body = b'{"model":"m","max_tokens":1,"messages":[]}'
    for url in ('https://evil.example/v1/chat/completions', 'http://localhost:11434/v1/models'):
        request = httpx.Request('POST', url, content=body)
        with pytest.raises(ValueError):
            guard.handle_request(request)
    sync.close()


def test_base_url_rejects_credentialed_or_decorated_url(monkeypatch):
    for bad in ('http://user:pw@localhost:11434/v1', 'http://localhost:11434/v1?x=1', 'gopher://localhost/'):
        monkeypatch.setenv('OLLAMA_BASE_URL', bad)
        with pytest.raises(ValueError, match='plain reviewed Ollama base URL'):
            _ollama_wire_factory()
