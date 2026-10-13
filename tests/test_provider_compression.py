"""Real SDK compression regression: decode once, settle before SDK DONE close."""
import asyncio,gzip,hashlib,json,zlib
import httpx,openai,pytest
from spec_generator.provider_transport import AdmittedTransport,AsyncAdmittedTransport

USAGE={'prompt_tokens':10,'completion_tokens':10,'total_tokens':20,'prompt_cache_hit_tokens':0,'prompt_cache_miss_tokens':10}
class Gateway:
 def __init__(self):self.events=[];self.wire=[]
 def reserve_serialized(self,**kw):self.events.append('reserve');return 'attempt'
 def start(self,a):self.events.append('start')
 def settle_raw(self,a,u):assert u==USAGE;self.events.append('settle')
 def uncertain(self,a):self.events.append('uncertain')
 def record_wire_evidence(self,a,e):self.wire.append(e)

def content():
 events=[{'id':'id','object':'chat.completion.chunk','created':1,'model':'deepseek-flash','choices':[{'index':0,'delta':{'content':'{"stack_keys":["a"]}'},'finish_reason':None}]},
 {'id':'id','object':'chat.completion.chunk','created':1,'model':'deepseek-flash','choices':[{'index':0,'delta':{},'finish_reason':'stop'}],'usage':USAGE}]
 return b': keepalive\r\n\r\n'+b''.join(b'data: '+json.dumps(e).encode()+b'\r\n\r\n' for e in events)+b'data: [DONE]\r\n\r\n'

def encoded(encoding):
 raw=content()
 return gzip.compress(raw,mtime=0) if encoding=='gzip' else zlib.compress(raw) if encoding=='deflate' else raw
class Chunks(httpx.SyncByteStream):
 def __init__(self,data):self.data=data;self.observed=bytearray()
 def __iter__(self):
  for i in range(0,len(self.data),7):
   chunk=self.data[i:i+7];self.observed.extend(chunk);yield chunk
class AsyncChunks(httpx.AsyncByteStream):
 def __init__(self,data):self.data=data;self.observed=bytearray()
 async def __aiter__(self):
  for i in range(0,len(self.data),7):
   chunk=self.data[i:i+7];self.observed.extend(chunk);yield chunk

def check(g,data):
 assert g.events==['reserve','start','settle']
 assert len(g.wire)==1
 e=g.wire[0];assert e['status']=='settled' and e['done_seen'] and e['terminal_usage']==USAGE and e['event_count']==3
 assert e['response_sha256']==hashlib.sha256(data).hexdigest() and e['response_bytes']==len(data)
 assert e['cr_count']==data.count(b'\r') and e['lf_count']==data.count(b'\n')

@pytest.mark.parametrize('encoding',['identity','gzip','deflate'])
@pytest.mark.parametrize('structured',[False,True])
def test_real_sync_sdk_compressed_sse_settles_exact_raw_hash(encoding,structured):
 data=encoded(encoding);g=Gateway();source=Chunks(data)
 def fake(r):return httpx.Response(200,headers={'content-type':'text/event-stream','content-encoding':encoding,'content-length':str(len(data))},stream=source)
 with httpx.Client(transport=AdmittedTransport(httpx.MockTransport(fake),g,stage='architecture',provider='deepseek',model='deepseek-flash')) as c:
  sdk=openai.OpenAI(api_key='fake',base_url='https://fake.invalid',http_client=c,max_retries=0)
  args=dict(model='deepseek-flash',messages=[{'role':'user','content':'fake'}],max_tokens=8192,stream_options={'include_usage':True})
  stream=sdk.beta.chat.completions.stream(**args,response_format={'type':'json_object'}) if structured else sdk.chat.completions.create(**args,stream=True)
  with stream as response:list(response)
 check(g,bytes(source.observed))

@pytest.mark.parametrize('encoding',['identity','gzip','deflate'])
@pytest.mark.parametrize('structured',[False,True])
def test_real_async_sdk_compressed_sse_settles_exact_raw_hash(encoding,structured):
 data=encoded(encoding);g=Gateway();source=AsyncChunks(data)
 async def fake(r):return httpx.Response(200,headers={'content-type':'text/event-stream','content-encoding':encoding,'content-length':str(len(data))},stream=source)
 async def run():
  async with httpx.AsyncClient(transport=AsyncAdmittedTransport(httpx.MockTransport(fake),g,stage='architecture',provider='deepseek',model='deepseek-flash')) as c:
   sdk=openai.AsyncOpenAI(api_key='fake',base_url='https://fake.invalid',http_client=c,max_retries=0)
   args=dict(model='deepseek-flash',messages=[{'role':'user','content':'fake'}],max_tokens=8192,stream_options={'include_usage':True})
   if structured:
    async with sdk.beta.chat.completions.stream(**args,response_format={'type':'json_object'}) as response:
     async for _ in response:pass
   else:
    async with await sdk.chat.completions.create(**args,stream=True) as response:
     async for _ in response:pass
 asyncio.run(run());check(g,bytes(source.observed))

@pytest.mark.parametrize('asynchronous',[False,True])
def test_compressed_missing_usage_remains_uncertain(asynchronous):
 from spec_generator.attempt_admission import AdmissionUnavailable
 data=gzip.compress(b'data: {"choices":[]}\r\n\r\ndata: [DONE]\r\n\r\n',mtime=0);g=Gateway()
 headers={'content-type':'text/event-stream','content-encoding':'gzip'}
 if asynchronous:
  async def fake(r):return httpx.Response(200,headers=headers,stream=AsyncChunks(data))
  async def run():
   async with httpx.AsyncClient(transport=AsyncAdmittedTransport(httpx.MockTransport(fake),g,stage='architecture',provider='deepseek',model='deepseek-flash')) as c:
    with pytest.raises(AdmissionUnavailable):
     await c.post('https://fake.invalid/chat/completions',json={'model':'deepseek-flash','max_tokens':8192,'stream':True,'stream_options':{'include_usage':True}})
  asyncio.run(run())
 else:
  def fake(r):return httpx.Response(200,headers=headers,stream=Chunks(data))
  with httpx.Client(transport=AdmittedTransport(httpx.MockTransport(fake),g,stage='architecture',provider='deepseek',model='deepseek-flash')) as c:
   with pytest.raises(AdmissionUnavailable):
    c.post('https://fake.invalid/chat/completions',json={'model':'deepseek-flash','max_tokens':8192,'stream':True,'stream_options':{'include_usage':True}})
 assert g.events==['reserve','start','uncertain']
 assert g.wire[0]['status']=='failed' and g.wire[0]['terminal_usage'] is None
