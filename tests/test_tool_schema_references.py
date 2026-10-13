import copy,json
import httpx,pytest
from spec_generator.model_config import _tool_payload_schema,_DeepSeekToolOutputAdapter
from spec_generator.schemas import arch_stack_options_schema
from spec_generator.provider_transport import AdmittedTransport,AsyncAdmittedTransport
from spec_generator.attempt_admission import AdmissionUnavailable

def resolve(document,ref):
    node=document
    for part in ref[2:].split('/'):
        node=node[part.replace('~1','/').replace('~0','~')]
    return node

def test_actual_sdk_framing_schema_resolves_without_mutation():
    from langchain_openai import ChatOpenAI
    schema=arch_stack_options_schema(['FR-001']);original=copy.deepcopy(schema);captured=[]
    def inner(request):
        captured.append(json.loads(request.read()))
        return httpx.Response(400,json={'error':{'message':'offline mock'}})
    with httpx.Client(transport=httpx.MockTransport(inner)) as c:
        model=_DeepSeekToolOutputAdapter(ChatOpenAI(model='deepseek-flash',api_key='not-used',base_url='https://fake.invalid',max_tokens=8192,max_retries=0,http_client=c),schema,'return_architecture_artifact')
        with pytest.raises(Exception):list(model.stream('offline fixture'))
    params=captured[0]['tools'][0]['function']['parameters']
    payload=params['properties']['payload'];ref=payload['properties']['options']['items']['$ref']
    assert ref=='#/properties/payload/$defs/TechnicalChoiceOption'
    assert resolve(params,ref)==original['$defs']['TechnicalChoiceOption']
    expected=copy.deepcopy(original);expected['properties']['options']['items']['$ref']=ref
    assert payload==expected and schema==original

@pytest.mark.parametrize('schema',[{'type':'object','properties':{'x':{'type':'string'}}},{'type':'array','items':{'type':'string'}}])
def test_no_ref_schema_unchanged(schema):assert _tool_payload_schema(schema)==schema

def test_recursive_and_escaped_refs():
    schema={'$defs':{'a/b':{'type':'object','properties':{'child':{'$ref':'#/$defs/a~1b'}}}},'anyOf':[{'$ref':'#'},{'$ref':'#/$defs/a~1b'}]}
    result=_tool_payload_schema(schema);params={'properties':{'payload':result}}
    assert resolve(params,result['anyOf'][1]['$ref'])==result['$defs']['a/b']
    assert result['anyOf'][0]['$ref']=='#/properties/payload'

@pytest.mark.parametrize('ref',['#/$defs/missing','https://fake.invalid/schema','#anchor'])
def test_bad_refs_stop_locally(ref):
    with pytest.raises(ValueError):_tool_payload_schema({'$ref':ref})

class Gateway:
    def __init__(self):self.events=[]
    def reserve_serialized(self,**kw):self.events.append('reserve');return kw['attempt_id']
    def start(self,a):self.events.append('start')
    def uncertain(self,a):self.events.append('uncertain')
    def settle_raw(self,*a):raise AssertionError('error response never settles')

@pytest.mark.parametrize('status',[400,401,429,500,503])
def test_sync_http_status_no_body_or_secrets_in_error(status):
    g=Gateway()
    with httpx.Client(transport=AdmittedTransport(httpx.MockTransport(lambda r:httpx.Response(status,text='private request echo sk-secret')),g,stage='architecture',provider='deepseek',model='deepseek-flash')) as c:
        with pytest.raises(AdmissionUnavailable) as e:c.post('https://fake.invalid/chat/completions',json={'model':'deepseek-flash','max_tokens':8192})
    assert str(e.value)==f'provider HTTP {status} response cannot prove usage'
    assert g.events==['reserve','start','uncertain']

@pytest.mark.parametrize('status',[400,429,503])
def test_async_http_status_holds(status):
    import asyncio
    g=Gateway()
    async def run():
        async with httpx.AsyncClient(transport=AsyncAdmittedTransport(httpx.MockTransport(lambda r:httpx.Response(status,text='secret')),g,stage='architecture',provider='deepseek',model='deepseek-flash')) as c:
            with pytest.raises(AdmissionUnavailable,match=f'HTTP {status}'):await c.post('https://fake.invalid/chat/completions',json={'model':'deepseek-flash','max_tokens':8192})
    asyncio.run(run());assert g.events==['reserve','start','uncertain']
