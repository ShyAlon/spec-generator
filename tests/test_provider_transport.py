import base64,json
import httpx,pytest
from spec_generator.provider_transport import AdmittedTransport
from spec_generator.attempt_admission import AdmissionUnavailable

class Gateway:
    def __init__(self):self.events=[]
    def reserve_serialized(self,**kw):self.events.append(('reserve',kw));return kw['attempt_id']
    def start(self,a):self.events.append(('start',a))
    def settle_raw(self,a,u):self.events.append(('settle',u))
    def uncertain(self,a):self.events.append(('uncertain',a))

def test_real_sdk_serialization_with_tools_and_raw_usage_no_network():
    from langchain_openai import ChatOpenAI
    g=Gateway();captured=[]
    raw_usage={'prompt_tokens':12,'completion_tokens':3,'total_tokens':15,'prompt_cache_hit_tokens':4,'prompt_cache_miss_tokens':8}
    def fake(request):
        assert g.events[-1][0]=='start'
        captured.append(request.content)
        return httpx.Response(200,json={'id':'fake','object':'chat.completion','model':'deepseek-flash','choices':[{'index':0,'message':{'role':'assistant','content':'fake'},'finish_reason':'stop'}],'usage':raw_usage})
    transport=AdmittedTransport(httpx.MockTransport(fake),g,stage='needs',provider='deepseek',model='deepseek-flash')
    with httpx.Client(transport=transport) as client:
        model=ChatOpenAI(model='deepseek-flash',api_key='fake-not-used',base_url='https://fake.invalid',max_tokens=10,max_retries=0,http_client=client,extra_body={'thinking':{'type':'disabled'}})
        model.bind_tools([{'type':'function','function':{'name':'test','description':'fake tool','parameters':{'type':'object','properties':{'x':{'type':'string'}}}}}]).invoke('fake input')
    reserved=base64.b64decode(g.events[0][1]['serialized_request'])
    assert reserved==captured[0]
    body=json.loads(reserved);assert body['tools'] and body['thinking']=={'type':'disabled'} and body['messages']
    assert [e[0] for e in g.events]==['reserve','start','settle']
    assert g.events[-1][1]==raw_usage

@pytest.mark.parametrize('payload',[{'model':'wrong','max_tokens':10},{'model':'fake'},{'model':'fake','max_tokens':True},{'model':'fake','max_tokens':10,'stream':True},{'model':'fake','max_tokens':10,'max_completion_tokens':10}])
def test_unsupported_request_never_reaches_inner(payload):
    g=Gateway();calls=[]
    with httpx.Client(transport=AdmittedTransport(httpx.MockTransport(lambda r:calls.append(1)),g,stage='needs',provider='test',model='fake')) as c:
        with pytest.raises(AdmissionUnavailable):c.post('https://fake.invalid/chat/completions',json=payload)
    assert not calls and not g.events

@pytest.mark.parametrize('status,body',[(500,{}),(200,{}),(200,{'usage':None})])
def test_missing_raw_usage_preserves_uncertainty(status,body):
    g=Gateway()
    with httpx.Client(transport=AdmittedTransport(httpx.MockTransport(lambda r:httpx.Response(status,json=body)),g,stage='needs',provider='test',model='fake')) as c:
        with pytest.raises(AdmissionUnavailable):c.post('https://fake.invalid/chat/completions',json={'model':'fake','max_tokens':10})
    assert [e[0] for e in g.events]==['reserve','start','uncertain']


def test_real_sdk_streaming_terminal_usage_uses_wire_boundary():
    from langchain_openai import ChatOpenAI
    g=Gateway();captured=[]
    usage={'prompt_tokens':2,'completion_tokens':1,'total_tokens':3,'prompt_cache_hit_tokens':1,'prompt_cache_miss_tokens':1}
    common={'id':'fake','object':'chat.completion.chunk','model':'deepseek-flash'}
    events=[common|{'choices':[{'index':0,'delta':{'role':'assistant','content':'fake'},'finish_reason':None}]},common|{'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]},common|{'choices':[],'usage':usage}]
    raw=b''.join(('data: '+json.dumps(e)+'\n\n').encode() for e in events)+b'data: [DONE]\n\n'
    def inner(r):
        captured.append(r.content)
        return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Chunks([raw[i:i+9] for i in range(0,len(raw),9)]))
    with httpx.Client(transport=AdmittedTransport(httpx.MockTransport(inner),g,stage='needs',provider='deepseek',model='deepseek-flash')) as client:
        model=ChatOpenAI(model='deepseek-flash',api_key='fake-not-used',base_url='https://fake.invalid',max_tokens=10,max_retries=0,http_client=client,stream_usage=True)
        chunks=list(model.stream('fake'))
    assert ''.join(c.content for c in chunks)=='fake'
    assert json.loads(captured[0])['stream_options']=={'include_usage':True}
    assert base64.b64decode(g.events[0][1]['serialized_request'])==captured[0]
    assert [e[0] for e in g.events]==['reserve','start','settle'] and g.events[-1][1]==usage


class AsyncChunks(httpx.AsyncByteStream):
    def __init__(self,chunks):self.chunks=chunks
    async def __aiter__(self):
        for c in self.chunks:yield c

@pytest.mark.parametrize('stream',[False,True])
def test_actual_async_sdk_raw_usage_and_serialized_request(stream):
    import asyncio
    from langchain_openai import ChatOpenAI
    from spec_generator.provider_transport import AsyncAdmittedTransport
    g=Gateway();captured=[];usage={'prompt_tokens':2,'completion_tokens':1,'total_tokens':3,'prompt_cache_hit_tokens':1,'prompt_cache_miss_tokens':1}
    async def fake(r):
        captured.append(r.content)
        if not stream:return httpx.Response(200,json={'id':'fake','object':'chat.completion','model':'deepseek-flash','choices':[{'index':0,'message':{'role':'assistant','content':'fake'},'finish_reason':'stop'}],'usage':usage})
        events=[{'id':'fake','object':'chat.completion.chunk','model':'deepseek-flash','choices':[{'index':0,'delta':{'role':'assistant','content':'fake'},'finish_reason':None}]},{'id':'fake','object':'chat.completion.chunk','model':'deepseek-flash','choices':[],'usage':usage}]
        raw=b''.join(('data: '+json.dumps(e)+'\n\n').encode() for e in events)+b'data: [DONE]\n\n'
        return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=AsyncChunks([raw[i:i+8] for i in range(0,len(raw),8)]))
    async def run():
        async with httpx.AsyncClient(transport=AsyncAdmittedTransport(httpx.MockTransport(fake),g,stage='needs',provider='deepseek',model='deepseek-flash')) as client:
            model=ChatOpenAI(model='deepseek-flash',api_key='fake-not-used',base_url='https://fake.invalid',max_tokens=10,max_retries=0,http_async_client=client,stream_usage=True)
            if stream:assert ''.join([c.content async for c in model.astream('fake')])=='fake'
            else:assert (await model.ainvoke('fake')).content=='fake'
    asyncio.run(run())
    assert base64.b64decode(g.events[0][1]['serialized_request'])==captured[0]
    assert [e[0] for e in g.events]==['reserve','start','settle'] and g.events[-1][1]==usage


def test_wire_stop_before_reserve_denies_and_after_accept_settles_then_discards(tmp_path):
    from spec_generator.stop_fence import StopFence,StopRequested
    f=StopFence(tmp_path/'stop.json');g=Gateway();calls=[]
    usage={'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}
    def fake(request):
        calls.append(1);f.request_stop();return httpx.Response(200,json={'usage':usage})
    with f.running():
        with httpx.Client(transport=AdmittedTransport(httpx.MockTransport(fake),g,stage='fake',provider='fake',model='fake')) as c:
            with pytest.raises(StopRequested):c.post('https://fake.invalid/chat/completions',json={'model':'fake','max_tokens':10})
            with pytest.raises(StopRequested):c.post('https://fake.invalid/chat/completions',json={'model':'fake','max_tokens':10})
    assert calls==[1] and sum(e[0]=='reserve' for e in g.events)==1
    assert ('settle',usage) in g.events


def test_wire_stop_stream_finishes_accepted_usage_before_discard(tmp_path):
    from spec_generator.stop_fence import StopFence,StopRequested
    f=StopFence(tmp_path/'stop.json');g=Gateway();usage={'prompt_tokens':2,'completion_tokens':3}
    class Accepted(httpx.SyncByteStream):
        def __iter__(self):
            f.request_stop()
            yield ('data: '+json.dumps({'usage':usage,'choices':[]})+'\n\ndata: [DONE]\n\n').encode()
    inner=httpx.MockTransport(lambda r:httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Accepted()))
    with f.running(),httpx.Client(transport=AdmittedTransport(inner,g,stage='fake',provider='fake',model='fake')) as c:
        with pytest.raises(StopRequested):c.post('https://fake.invalid/chat/completions',json={'model':'fake','max_tokens':10,'stream':True,'stream_options':{'include_usage':True}})
    assert [e[0] for e in g.events]==['reserve','start','settle']


class Chunks(httpx.SyncByteStream):
    def __init__(self,chunks):self.chunks=chunks
    def __iter__(self):yield from self.chunks

@pytest.mark.parametrize('mode',['complete','no_done','no_usage','bad_json','early_close','duplicate_usage'])
def test_stream_terminal_usage_fragmentation_and_unknown_holds(mode):
    g=Gateway();usage={'prompt_tokens':2,'completion_tokens':3}
    parts=[b'data: {"choices":[{"delta":{"content":"fake"}}]}\r\n\r\n']
    if mode!='no_usage':parts.append(('data: '+json.dumps({'usage':usage,'choices':[]})+'\n\n').encode())
    if mode=='duplicate_usage':parts.append(parts[-1])
    if mode=='bad_json':parts.append(b'data: bad\n\n')
    if mode!='no_done':parts.append(b'data: [DONE]\n\n')
    raw=b''.join(parts)
    inner=httpx.MockTransport(lambda r:httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Chunks([raw[i:i+7] for i in range(0,len(raw),7)])))
    with httpx.Client(transport=AdmittedTransport(inner,g,stage='needs',provider='test',model='fake')) as c:
        with c.stream('POST','https://fake.invalid/chat/completions',json={'model':'fake','max_tokens':10,'stream':True,'stream_options':{'include_usage':True}}) as response:
            if mode=='early_close':next(response.iter_raw())
            elif mode=='complete':assert b''.join(response.iter_raw())==raw
            else:
                with pytest.raises((AdmissionUnavailable,ValueError)):list(response.iter_raw())
    assert [e[0] for e in g.events]==['reserve','start','settle' if mode=='complete' else 'uncertain']
    if mode=='complete':assert g.events[-1][1]==usage



@pytest.mark.parametrize('choices',[
    [{'index':0,'delta':{'content':'','role':None},'finish_reason':'stop'}],
    [{'index':0,'delta':{},'finish_reason':'length'}],
    [{'index':0,'delta':{'content':'late'},'finish_reason':'stop'}],
    [{'index':0,'delta':{},'finish_reason':None}],None])
def test_documented_finish_only_usage_event_or_fail_closed(choices):
    g=Gateway();usage={'prompt_tokens':2,'completion_tokens':1}
    raw=('data: '+json.dumps({'choices':choices,'usage':usage})+'\n\ndata: [DONE]\n\n').encode()
    terminal=isinstance(choices,list) and choices[0]['finish_reason'] in ('stop','length') and not choices[0]['delta'].get('content')
    inner=httpx.MockTransport(lambda r:httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Chunks([raw])))
    with httpx.Client(transport=AdmittedTransport(inner,g,stage='needs',provider='deepseek',model='fake')) as c:
        if terminal:
            assert c.post('https://fake.invalid/chat/completions',json={'model':'fake','max_tokens':10,'stream':True,'stream_options':{'include_usage':True}}).content==raw
        else:
            with pytest.raises(AdmissionUnavailable):c.post('https://fake.invalid/chat/completions',json={'model':'fake','max_tokens':10,'stream':True,'stream_options':{'include_usage':True}})
    assert g.events[-1][0]==('settle' if terminal else 'uncertain')

