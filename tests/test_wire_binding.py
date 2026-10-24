import json,base64
from types import SimpleNamespace
import pytest,httpx
from spec_generator.wire_binding import wire_clients
from spec_generator.provider_transport import AdmittedTransport,AsyncAdmittedTransport
from spec_generator.attempt_admission import admission_gateway,AdmissionUnavailable
from test_provider_transport import Gateway,Chunks

@pytest.mark.parametrize('stream',[False,True])
def test_actual_model_factory_checked_call_reserves_once_exact_tools(monkeypatch,stream):
    from spec_generator import model_config as module
    monkeypatch.setattr(module,'for_stage',lambda stage:SimpleNamespace(provider='deepseek',model='deepseek-flash',max_tokens=10,temperature=0,seed=None))
    monkeypatch.setattr(module,'_check_truncated',lambda *args:None)
    monkeypatch.setenv('DEEPSEEK_API_KEY','fake-not-used')
    g=Gateway();clients=[];captured=[]
    usage={'prompt_tokens':2,'completion_tokens':1,'total_tokens':3,'prompt_cache_hit_tokens':1,'prompt_cache_miss_tokens':1}
    def fake(r):
        captured.append(r.content)
        assert g.events[-1][0]=='start'
        payload=json.loads(r.content)
        if not payload.get('stream'):
            return httpx.Response(200,json={'id':'fake','object':'chat.completion','model':'deepseek-flash','choices':[{'index':0,'message':{'role':'assistant','content':'ok'},'finish_reason':'stop'}],'usage':usage})
        common={'id':'fake','object':'chat.completion.chunk','model':'deepseek-flash'}
        events=[common|{'choices':[{'index':0,'delta':{'role':'assistant','content':'ok'},'finish_reason':None}]},common|{'choices':[],'usage':usage}]
        raw=b''.join(('data: '+json.dumps(e)+'\n\n').encode() for e in events)+b'data: [DONE]\n\n'
        return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Chunks([raw[i:i+9] for i in range(0,len(raw),9)]))
    def factory(**kw):
        args={k:kw[k] for k in ('stage','provider','model')}
        pair=(httpx.Client(transport=AdmittedTransport(httpx.MockTransport(fake),g,**args)),httpx.AsyncClient(transport=AsyncAdmittedTransport(httpx.MockTransport(fake),g,**args)))
        clients.extend(pair);return pair
    try:
        with admission_gateway(g,required=True),wire_clients(g,factory):
            model=module.build_chat_model('needs',structured=False)
            assert model.max_retries==0
            if stream:assert ''.join(c.content for c in module.stream_checked(model,['fake'],'needs'))=='ok'
            else:assert module.invoke_checked(model,['fake'],'needs').content=='ok'
        assert [e[0] for e in g.events]==['reserve','start','settle']
        assert base64.b64decode(g.events[0][1]['serialized_request'])==captured[0]
    finally:
        import asyncio
        for c in clients:
            if isinstance(c,httpx.Client):c.close()
            else:asyncio.run(c.aclose())


def test_unbound_model_and_alternate_transport_fail_before_contact():
    from spec_generator.attempt_admission import checked_model_invoke
    from spec_generator.wire_binding import model_clients
    calls=[];g=Gateway()
    with wire_clients(g,lambda **kw:(httpx.Client(transport=httpx.MockTransport(lambda r:calls.append(1))),None)):
        with pytest.raises(AdmissionUnavailable):model_clients(stage='needs',provider='deepseek',model='fake')
        with pytest.raises(AdmissionUnavailable):checked_model_invoke(SimpleNamespace(invoke=lambda m:calls.append(1)),[],stage='needs',provider='deepseek',model='fake',max_output_tokens=10)
        with pytest.raises(AdmissionUnavailable):model_clients(stage='needs',provider='anthropic',model='fake')
    assert calls==[]


def test_actual_forced_tool_adapter_remains_wire_bound(monkeypatch):
    from spec_generator import model_config as module
    monkeypatch.setattr(module,'for_stage',lambda stage:SimpleNamespace(provider='deepseek',model='deepseek-flash',max_tokens=10,temperature=0,seed=None))
    monkeypatch.setattr(module,'_check_truncated',lambda *args:None)
    monkeypatch.setenv('DEEPSEEK_API_KEY','fake-not-used')
    g=Gateway();clients=[];captured=[]
    def fake(request):
        captured.append(request.content)
        tool={'id':'fake-tool','type':'function','function':{'name':'return_needs_artifact','arguments':'{"payload":{"x":"ok"}}'}}
        return httpx.Response(200,json={'id':'fake','object':'chat.completion','model':'deepseek-flash','choices':[{'index':0,'message':{'role':'assistant','content':None,'tool_calls':[tool]},'finish_reason':'tool_calls'}],'usage':{'prompt_tokens':2,'completion_tokens':1,'total_tokens':3}})
    def factory(**kw):
        gateway=kw.pop('gateway')
        pair=(httpx.Client(transport=AdmittedTransport(httpx.MockTransport(fake),gateway,**kw)),httpx.AsyncClient(transport=AsyncAdmittedTransport(httpx.MockTransport(fake),gateway,**kw)))
        clients.extend(pair);return pair
    try:
        with admission_gateway(g,required=True),wire_clients(g,factory):
            model=module.build_chat_model('needs',schema={'type':'object','properties':{'x':{'type':'string'}}})
            assert json.loads(module.invoke_checked(model,['fake'],'needs').content)=={'x':'ok'}
        assert [e[0] for e in g.events]==['reserve','start','settle']
        raw=base64.b64decode(g.events[0][1]['serialized_request'])
        assert raw==captured[0] and json.loads(raw)['tools'] and json.loads(raw)['tool_choice']
    finally:
        import asyncio
        for client in clients:
            if isinstance(client,httpx.Client):client.close()
            else:asyncio.run(client.aclose())


def test_actual_forced_tool_adapter_remains_wire_bound(monkeypatch):
    from spec_generator import model_config as module
    monkeypatch.setattr(module,'for_stage',lambda stage:SimpleNamespace(provider='deepseek',model='deepseek-flash',max_tokens=10,temperature=0,seed=None))
    monkeypatch.setattr(module,'_check_truncated',lambda *args:None)
    monkeypatch.setenv('DEEPSEEK_API_KEY','fake-not-used')
    g=Gateway();clients=[];captured=[]
    def fake(request):
        captured.append(request.content)
        tool={'id':'fake-tool','type':'function','function':{'name':'return_needs_artifact','arguments':'{"payload":{"x":"ok"}}'}}
        return httpx.Response(200,json={'id':'fake','object':'chat.completion','model':'deepseek-flash','choices':[{'index':0,'message':{'role':'assistant','content':None,'tool_calls':[tool]},'finish_reason':'tool_calls'}],'usage':{'prompt_tokens':2,'completion_tokens':1,'total_tokens':3}})
    def factory(**kw):
        gateway=kw.pop('gateway')
        pair=(httpx.Client(transport=AdmittedTransport(httpx.MockTransport(fake),gateway,**kw)),httpx.AsyncClient(transport=AsyncAdmittedTransport(httpx.MockTransport(fake),gateway,**kw)))
        clients.extend(pair);return pair
    try:
        with admission_gateway(g,required=True),wire_clients(g,factory):
            model=module.build_chat_model('needs',schema={'type':'object','properties':{'x':{'type':'string'}}})
            assert json.loads(module.invoke_checked(model,['fake'],'needs').content)=={'x':'ok'}
        assert [e[0] for e in g.events]==['reserve','start','settle']
        raw=base64.b64decode(g.events[0][1]['serialized_request'])
        assert raw==captured[0] and json.loads(raw)['tools'] and json.loads(raw)['tool_choice']
    finally:
        import asyncio
        for client in clients:
            if isinstance(client,httpx.Client):client.close()
            else:asyncio.run(client.aclose())


def test_ollama_model_uses_admitted_wire_and_settles_usage(monkeypatch):
    """The local Ollama route is admitted like any provider: same job-bound clients,
    same reserve/start/settle, non-streaming usage from the response envelope."""
    from spec_generator import model_config as module
    monkeypatch.setattr(module,'for_stage',lambda stage:SimpleNamespace(provider='ollama',model='qwen2.5-coder:14b',max_tokens=10,temperature=0,seed=None))
    monkeypatch.setattr(module,'_check_truncated',lambda *args:None)
    g=Gateway();clients=[];captured=[]
    usage={'prompt_tokens':5,'completion_tokens':2,'total_tokens':7}
    def fake(r):
        captured.append(r.content)
        payload=json.loads(r.content)
        assert not payload.get('stream')
        return httpx.Response(200,json={'id':'fake','object':'chat.completion','model':'qwen2.5-coder:14b','choices':[{'index':0,'message':{'role':'assistant','content':'ok'},'finish_reason':'stop'}],'usage':usage})
    def factory(**kw):
        args={k:kw[k] for k in ('stage','provider','model')}
        pair=(httpx.Client(transport=AdmittedTransport(httpx.MockTransport(fake),g,**args)),httpx.AsyncClient(transport=AsyncAdmittedTransport(httpx.MockTransport(fake),g,**args)))
        clients.extend(pair);return pair
    try:
        with admission_gateway(g,required=True),wire_clients(g,factory):
            model=module.build_chat_model('needs',structured=False)
            assert module.invoke_checked(model,['fake'],'needs').content=='ok'
        assert [e[0] for e in g.events]==['reserve','start','settle']
        sent=json.loads(base64.b64decode(g.events[0][1]['serialized_request']))
        assert sent==json.loads(captured[0]) and sent['model']=='qwen2.5-coder:14b'
    finally:
        import asyncio
        for c in clients:
            if isinstance(c,httpx.Client):c.close()
            else:asyncio.run(c.aclose())


def test_unlisted_provider_still_refused_under_binding():
    from spec_generator.wire_binding import model_clients,validate_provider
    g=Gateway()
    with wire_clients(g,lambda **kw:(None,None)):
        with pytest.raises(AdmissionUnavailable):validate_provider('anthropic')
        with pytest.raises(AdmissionUnavailable):model_clients(stage='needs',provider='anthropic',model='fake')
