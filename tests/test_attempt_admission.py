import pytest
from types import SimpleNamespace
from spec_generator import attempt_admission as admission

@pytest.fixture
def cloud(monkeypatch):monkeypatch.setenv('VMODEL_TENANT_OWNER','test-owner')

class Gateway:
    def __init__(self):self.events=[]
    def reserve(self,**kw):self.events.append(('reserve',kw));return kw['attempt_id']
    def start(self,attempt):self.events.append(('start',attempt))
    def settle(self,attempt,response):self.events.append(('settle',attempt))
    def uncertain(self,attempt):self.events.append(('uncertain',attempt))

OPTIONS=dict(stage='needs',provider='test',model='fake',messages=['private input'],max_output_tokens=100)

def test_missing_admission_never_contacts_provider(cloud):
    calls=[]
    with pytest.raises(admission.AdmissionUnavailable):admission.admitted_invoke(lambda:calls.append(1),**OPTIONS)
    assert calls==[]

def test_admitted_attempt_is_started_before_contact_and_settled(cloud):
    gateway=Gateway()
    def provider():
        assert [e[0] for e in gateway.events]==['reserve','start']
        return SimpleNamespace(usage_metadata={'input_tokens':1,'output_tokens':2})
    with admission.admission_gateway(gateway):admission.admitted_invoke(provider,**OPTIONS)
    assert [e[0] for e in gateway.events]==['reserve','start','settle']

def test_provider_failure_is_uncertain_not_refunded(cloud):
    gateway=Gateway()
    def provider():raise RuntimeError('fake timeout')
    with admission.admission_gateway(gateway),pytest.raises(RuntimeError):admission.admitted_invoke(provider,**OPTIONS)
    assert [e[0] for e in gateway.events]==['reserve','start','uncertain']

def test_rejection_does_not_call_provider(cloud):
    gateway=Gateway();gateway.reserve=lambda **kw:None;calls=[]
    with admission.admission_gateway(gateway),pytest.raises(admission.AdmissionUnavailable):admission.admitted_invoke(lambda:calls.append(1),**OPTIONS)
    assert calls==[]

def test_stream_has_precontact_gate_and_preserves_unknown_usage(cloud):
    gateway=Gateway()
    def stream():
        assert gateway.events[-1][0]=='start'
        yield 'fake chunk'
    with admission.admission_gateway(gateway):assert list(admission.admitted_stream(stream,**OPTIONS))==['fake chunk']
    assert gateway.events[-1][0]=='uncertain'

def test_threaded_checked_invoke_fails_closed_and_context_reaches_gateway(cloud,monkeypatch):
    from spec_generator import model_config as module
    monkeypatch.setattr(module,'for_stage',lambda stage:SimpleNamespace(provider='test',model='fake',max_tokens=100))
    monkeypatch.setattr(module,'_check_truncated',lambda *args:None)
    class Fake:
        calls=0
        def invoke(self,messages):
            self.calls+=1
            return SimpleNamespace(content='ok',usage_metadata={'input_tokens':1,'output_tokens':1})
    fake=Fake()
    with pytest.raises(admission.AdmissionUnavailable):module.invoke_checked(fake,['input'],'needs')
    assert fake.calls==0
    gateway=Gateway()
    with admission.admission_gateway(gateway):assert module.invoke_checked(fake,['input'],'needs').content=='ok'
    assert fake.calls==1 and gateway.events[-1][0]=='settle'


def test_threaded_stream_context_is_preserved_and_missing_gateway_stops_contact(cloud,monkeypatch):
    from spec_generator import model_config as module
    monkeypatch.setattr(module,'for_stage',lambda stage:SimpleNamespace(provider='test',model='fake',max_tokens=100))
    monkeypatch.setattr(module,'_check_truncated',lambda *args:None)
    class Fake:
        calls=0
        def stream(self,messages):
            self.calls+=1
            yield SimpleNamespace(content='ok',usage_metadata={'input_tokens':1,'output_tokens':1},response_metadata={})
    fake=Fake()
    with pytest.raises(admission.AdmissionUnavailable):list(module.stream_checked(fake,['input'],'needs'))
    assert fake.calls==0
    gateway=Gateway()
    with admission.admission_gateway(gateway):list(module.stream_checked(fake,['input'],'needs'))
    assert fake.calls==1 and gateway.events[-1][0]=='uncertain'



def test_explicit_worker_context_requires_admission_without_environment(monkeypatch):
    monkeypatch.delenv('VMODEL_TENANT_OWNER',raising=False)
    monkeypatch.delenv('INTERVIEW_ENGINE_TENANTS',raising=False)
    monkeypatch.delenv('VMODEL_ADMISSION_FD',raising=False)
    calls=[]
    with admission.admission_gateway(None,required=True):
        with pytest.raises(admission.AdmissionUnavailable):admission.admitted_invoke(lambda:calls.append(1),**OPTIONS)
    assert calls==[]
    gateway=Gateway()
    with admission.admission_gateway(gateway,required=True):
        admission.admitted_invoke(lambda:SimpleNamespace(usage_metadata={'input_tokens':1,'output_tokens':2}),**OPTIONS)
    assert [e[0] for e in gateway.events]==['reserve','start','settle']
    assert not admission._cloud()
