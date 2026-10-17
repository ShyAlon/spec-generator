"""A connect failure the ledger releases is retried as a new attempt; anything else stays uncertain."""
import asyncio, json
import httpx, pytest

from spec_generator import model_config, provider_transport as pt
from spec_generator.attempt_admission import AdmissionRefused

BODY = {'model': 'fake', 'max_tokens': 10}
PROOF = lambda name: {'kind': 'connect-failure', 'error_class': name, 'bytes_sent': 0, 'response_seen': False}


class Gateway:
    def __init__(self, refuse_release=False, has_release=True):
        self.events = []; self.released = []; self.refuse = refuse_release
        if not has_release:
            self.release = None; del self.release
    def reserve_serialized(self, **kw): self.events.append('reserve'); return kw['attempt_id']
    def start(self, a): self.events.append('start')
    def settle_raw(self, a, u): self.events.append('settle')
    def uncertain(self, a): self.events.append('uncertain')
    def release(self, a, ev):
        if self.refuse: raise AdmissionRefused('no', 'release limit')
        self.released.append(ev); self.events.append('release')


class NoRelease:
    events = []
    def reserve_serialized(self, **kw): self.events.append('reserve'); return kw['attempt_id']
    def start(self, a): self.events.append('start')
    def settle_raw(self, a, u): self.events.append('settle')
    def uncertain(self, a): self.events.append('uncertain')


def ok(request): return httpx.Response(200, json={'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})


def flaky(n, exc=httpx.ConnectError):
    calls = []
    def h(request):
        calls.append(1)
        if len(calls) <= n: raise exc('boom', request=request)
        return ok(request)
    h.calls = calls
    return h


@pytest.fixture
def waits(monkeypatch):
    seen = []; lines = []
    monkeypatch.setattr(pt, '_sleep', lambda s: seen.append(s))
    monkeypatch.setattr(model_config, '_console_line', lambda m: lines.append(m))
    monkeypatch.setattr(model_config, '_log_llm', lambda d: lines.append(json.dumps(d)))
    return seen, lines


def post(g, inner):
    with httpx.Client(transport=pt.AdmittedTransport(httpx.MockTransport(inner), g, stage='s', provider='p', model='fake')) as c:
        return c.post('https://fake.invalid/chat/completions', json=BODY)


def test_connect_failure_is_released_then_retried_as_a_new_attempt(waits):
    seen, lines = waits; g = Gateway(); h = flaky(2)
    assert post(g, h).status_code == 200
    assert g.events == ['reserve', 'start', 'release', 'reserve', 'start', 'release', 'reserve', 'start', 'settle']
    assert 'uncertain' not in g.events and len(h.calls) == 3
    assert g.released == [PROOF('ConnectError')] * 2
    assert [2.0 <= w < 2.6 or 5.0 <= w < 5.6 for w in seen] == [True, True]
    assert any('boundary=provider-connect' in l and 'try=1/3' in l and 'ConnectError' in l and 'released=yes' in l for l in lines)


def test_connect_timeout_counts_too(waits):
    g = Gateway(); assert post(g, flaky(1, httpx.ConnectTimeout)).status_code == 200
    assert g.released == [PROOF('ConnectTimeout')]


def test_gives_up_after_three_tries_with_every_attempt_released(waits):
    g = Gateway(); h = flaky(9)
    with pytest.raises(httpx.ConnectError): post(g, h)
    assert len(h.calls) == 3 and g.events.count('release') == 3 and 'uncertain' not in g.events


@pytest.mark.parametrize('exc', [httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError, httpx.ReadError])
def test_failures_after_the_request_may_have_left_stay_uncertain_and_are_not_retried(waits, exc):
    g = Gateway(); h = flaky(1, exc)
    with pytest.raises(exc): post(g, h)
    assert g.events == ['reserve', 'start', 'uncertain'] and len(h.calls) == 1 and not g.released


def test_http_error_response_stays_uncertain(waits):
    g = Gateway()
    with pytest.raises(Exception): post(g, lambda r: httpx.Response(503))
    assert g.events == ['reserve', 'start', 'uncertain'] and not g.released


def test_a_refused_release_keeps_the_uncertain_behaviour_and_does_not_retry(waits):
    seen, lines = waits; g = Gateway(refuse_release=True); h = flaky(1)
    with pytest.raises(httpx.ConnectError): post(g, h)
    assert g.events == ['reserve', 'start', 'uncertain'] and len(h.calls) == 1 and not seen
    assert any('release-refused' in l and 'release limit' in l for l in lines)


def test_a_gateway_without_release_keeps_the_uncertain_behaviour(waits):
    g = NoRelease(); g.events = []; h = flaky(1)
    with pytest.raises(httpx.ConnectError): post(g, h)
    assert g.events == ['reserve', 'start', 'uncertain'] and len(h.calls) == 1


def test_stop_request_during_the_wait_ends_the_retry(waits, monkeypatch):
    import spec_generator.stop_fence as sf
    n = []
    def chk():
        n.append(1)
        if len(n) >= 3: raise RuntimeError('stop requested')
    monkeypatch.setattr(sf, 'check_execution', chk)
    g = Gateway(); h = flaky(9)
    with pytest.raises(RuntimeError, match='stop requested'): post(g, h)
    assert len(h.calls) == 1


def test_async_transport_releases_and_retries(monkeypatch):
    lines = []
    async def nosleep(s): lines.append(s)
    monkeypatch.setattr(pt, '_asleep', nosleep)
    monkeypatch.setattr(model_config, '_console_line', lambda m: lines.append(m))
    monkeypatch.setattr(model_config, '_log_llm', lambda d: None)
    g = Gateway(); calls = []
    async def h(request):
        calls.append(1)
        if len(calls) == 1: raise httpx.ConnectError('x', request=request)
        return ok(request)
    async def go():
        async with httpx.AsyncClient(transport=pt.AsyncAdmittedTransport(httpx.MockTransport(h), g, stage='s', provider='p', model='fake')) as c:
            return await c.post('https://fake.invalid/chat/completions', json=BODY)
    assert asyncio.run(go()).status_code == 200
    assert g.events == ['reserve', 'start', 'release', 'reserve', 'start', 'settle']


def test_socket_gateway_sends_the_release_request():
    import socket, threading
    from spec_generator.admission_client import SocketAdmissionGateway
    a, b = socket.socketpair(); seen = []
    def server():
        f = b.makefile('rwb'); seen.append(json.loads(f.readline())); f.write(b'{"ok":true,"result":{"state":"released"}}\n'); f.flush()
    t = threading.Thread(target=server); t.start()
    gw = SocketAdmissionGateway(a.fileno())
    assert gw.release('a1', PROOF('ConnectError')) == {'state': 'released'}
    t.join(); assert seen == [{'action': 'release', 'payload': {'attempt': 'a1', 'evidence': PROOF('ConnectError')}}]
