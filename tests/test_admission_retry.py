"""Bounded retry of a refused reserve, and the real refusal text. No provider, no database."""
import asyncio, json, socket, threading
import httpx, pytest

from spec_generator import model_config, provider_transport as pt
from spec_generator.admission_client import SocketAdmissionGateway
from spec_generator.attempt_admission import AdmissionRefused, AdmissionUnavailable

BODY = {'model': 'fake', 'max_tokens': 10}


class Gateway:
    def __init__(self, failures):
        self.failures = list(failures); self.events = []; self.ids = []
    def reserve_serialized(self, **kw):
        self.ids.append(kw['attempt_id']); self.events.append('reserve')
        if self.failures:
            raise self.failures.pop(0)
        return kw['attempt_id']
    def start(self, a): self.events.append('start')
    def settle_raw(self, a, u): self.events.append('settle')
    def uncertain(self, a): self.events.append('uncertain')


def _ok(request):
    return httpx.Response(200, json={'id': 'x', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})


@pytest.fixture
def waits(monkeypatch):
    seen = []; lines = []
    monkeypatch.setattr(pt, '_sleep', lambda s: seen.append(s))
    monkeypatch.setattr(model_config, '_console_line', lambda m: lines.append(m))
    monkeypatch.setattr(model_config, '_log_llm', lambda d: lines.append(json.dumps(d)))
    return seen, lines


def _post(g, inner=_ok):
    with httpx.Client(transport=pt.AdmittedTransport(httpx.MockTransport(inner), g, stage='review', provider='test', model='fake')) as c:
        return c.post('https://fake.invalid/chat/completions', json=BODY)


def test_two_refusals_then_success_reuses_the_attempt_id_and_bills_once(waits):
    seen, lines = waits
    g = Gateway([AdmissionRefused('r', ''), AdmissionRefused('r', 'OperationalError')])
    assert _post(g).status_code == 200
    assert g.events == ['reserve', 'reserve', 'reserve', 'start', 'settle']
    assert len(set(g.ids)) == 1                      # same attempt id: the ledger replays, never a second attempt
    assert [round(x) for x in seen] == [2, 5]
    retry_lines = [l for l in lines if 'boundary=admission-retry' in l]
    assert len(retry_lines) == 2 and 'elapsed=' in retry_lines[0] and '+00:00' in retry_lines[0] and 'try=1/3' in retry_lines[0]


def test_three_refusals_stop_after_three_tries_and_never_reach_the_provider(waits):
    seen, _ = waits; called = []
    g = Gateway([AdmissionRefused('r', '')] * 3)
    with pytest.raises(AdmissionRefused):
        _post(g, lambda r: called.append(1))
    assert g.events == ['reserve'] * 3 and len(seen) == 2 and not called


def test_deterministic_refusal_is_not_retried(waits):
    seen, lines = waits
    g = Gateway([AdmissionRefused('r', 'AdmissionDenied: explicit reviewed estimate policy required')])
    with pytest.raises(AdmissionRefused) as e:
        _post(g)
    assert g.events == ['reserve'] and not seen
    assert 'estimate policy' in e.value.reason
    assert any('decision=stop' in l for l in lines)


@pytest.mark.parametrize('exc', [AdmissionUnavailable('timed out'), RuntimeError('x')])
def test_channel_failures_are_not_retried(waits, exc):
    seen, _ = waits
    g = Gateway([exc])
    with pytest.raises(Exception):
        _post(g)
    assert g.events == ['reserve'] and not seen


def test_start_and_provider_failures_are_not_retried(waits):
    seen, _ = waits
    g = Gateway([])
    with pytest.raises(AdmissionUnavailable):
        _post(g, lambda r: httpx.Response(500, json={}))
    assert g.events == ['reserve', 'start', 'uncertain'] and not seen


def test_async_transport_retries_the_same_way(monkeypatch, waits):
    seen, _ = waits
    async def nosleep(s): seen.append(s)
    monkeypatch.setattr(pt, '_asleep', nosleep)
    g = Gateway([AdmissionRefused('r', '')])
    async def go():
        async def inner(request): return _ok(request)
        class T(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request): return _ok(request)
        async with httpx.AsyncClient(transport=pt.AsyncAdmittedTransport(T(), g, stage='review', provider='test', model='fake')) as c:
            return await c.post('https://fake.invalid/chat/completions', json=BODY)
    assert asyncio.run(go()).status_code == 200
    assert g.events == ['reserve', 'reserve', 'start', 'settle'] and len(seen) == 1


# ---- the worker side of the socket: the real refusal text reaches the log ----
def _serve(reply_bytes_list):
    a, b = socket.socketpair()
    def run():
        f = a.makefile('rwb')
        for reply in reply_bytes_list:
            f.readline()
            if reply is None:
                break
            f.write(reply + b'\n'); f.flush()
        f.close(); a.close()
    threading.Thread(target=run, daemon=True).start()
    return b


def test_client_exposes_the_servers_reason():
    b = _serve([json.dumps({'ok': False, 'error': 'worker attempt admission denied', 'reason': 'AdmissionDenied: owner attempt concurrency exhausted'}).encode()])
    g = SocketAdmissionGateway(b.fileno())
    with pytest.raises(AdmissionRefused) as e:
        g.reserve_serialized(x=1)
    assert e.value.reason == 'AdmissionDenied: owner attempt concurrency exhausted'
    assert 'concurrency exhausted' in str(e.value)


def test_old_server_without_a_reason_is_named_as_such():
    b = _serve([json.dumps({'ok': False, 'error': 'worker attempt admission denied'}).encode()])
    g = SocketAdmissionGateway(b.fileno())
    with pytest.raises(AdmissionRefused) as e:
        g.reserve_serialized(x=1)
    assert e.value.reason == '' and 'gave no reason' in str(e.value)
    assert pt._retryable(e.value)


def test_closed_channel_is_unavailable_not_refused():
    b = _serve([None])
    g = SocketAdmissionGateway(b.fileno())
    with pytest.raises(AdmissionUnavailable) as e:
        g.reserve_serialized(x=1)
    assert not isinstance(e.value, AdmissionRefused) and not pt._retryable(e.value)


def test_a_refusal_is_reported_as_an_admission_stop_with_the_real_text_not_as_connectivity():
    from spec_generator.errors import translate_llm_error
    refused = AdmissionRefused('worker attempt admission refused: AdmissionDenied: owner spend allowance exhausted',
                               'AdmissionDenied: owner spend allowance exhausted')
    try:
        try:
            raise refused
        except AdmissionRefused as inner:
            raise RuntimeError('Connection error.') from inner
    except RuntimeError as outer:
        friendly = translate_llm_error(outer, 'review', 'deepseek', 'deepseek-flash', None)
    assert 'stopped by admission' in str(friendly) or 'stopped by admission' in (friendly.args[0] if friendly.args else '')
    assert 'owner spend allowance exhausted' in repr(vars(friendly)) + str(friendly) + repr(friendly.args)
