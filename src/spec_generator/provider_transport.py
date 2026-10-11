"""Uninstalled SDK wire-boundary prototype. Inner transport must be supplied.

No HTTP client, credential, endpoint, rate or bound is created here. A runner
must bind a durable-job gateway and reviewed serialized-request resolver first.
Synchronous JSON and terminal-usage SSE supported; async wire prototypes exist; no runner installs them.
"""
import base64
import hashlib
import json
import uuid
import httpx
from .attempt_admission import AdmissionUnavailable


class WireEvidence:
    """Bounded metadata + exact raw terminal usage; never keys/request bodies."""
    def __init__(self,gateway,attempt):
        self.gateway=gateway;self.attempt=attempt;self.sha=hashlib.sha256();self.bytes=0
        self.cr=0;self.lf=0;self.events=0;self.usage=None;self.done=False;self.reported=False
    def chunk(self,chunk):
        self.sha.update(chunk);self.bytes+=len(chunk);self.cr+=chunk.count(b'\r');self.lf+=chunk.count(b'\n')
    def emit(self,status,error=None):
        if self.reported:return
        self.reported=True
        callback=getattr(self.gateway,'record_wire_evidence',None)
        if callback is not None:
            # Persistence failure is visible, not permission to drop evidence.
            callback(self.attempt,{'status':status,'response_sha256':self.sha.hexdigest(),
                'response_bytes':self.bytes,'cr_count':self.cr,'lf_count':self.lf,
                'event_count':self.events,'terminal_usage':self.usage,'done_seen':self.done,
                'error_class':type(error).__name__ if error is not None else None})


class SSELines:
    """Bounded incremental LF/CRLF/CR decoder, including split CRLF."""
    def __init__(self):self.buffer=bytearray();self.skip_lf=False
    def feed(self,chunk):
        for value in chunk:
            if self.skip_lf:
                self.skip_lf=False
                if value==10:continue
            if value in (10,13):
                line=bytes(self.buffer);self.buffer.clear();self.skip_lf=value==13
                yield line
            else:
                self.buffer.append(value)
                if len(self.buffer)>65536:raise AdmissionUnavailable('oversized provider SSE line')


def _terminal_usage_choices(choices):
    # OpenAI's separate empty-choice usage event and DeepSeek's documented
    # finish-only choice event are both terminal, never a content-bearing event.
    if choices==[]:return True
    if not isinstance(choices,list) or not choices:return False
    for choice in choices:
        if not isinstance(choice,dict) or choice.get('finish_reason') not in ('stop','length','tool_calls','content_filter','insufficient_system_resource'):return False
        delta=choice.get('delta')
        if not isinstance(delta,dict) or any(value not in (None,'',[],{}) for value in delta.values()):return False
    return True


class AdmittedTransport(httpx.BaseTransport):
    def __init__(self, inner, gateway, *, stage, provider, model):
        self.inner=inner;self.gateway=gateway
        self.stage=stage;self.provider=provider;self.model=model

    def handle_request(self, request):
        from .stop_fence import check_execution
        check_execution()
        # This is the final SDK payload, not a reconstruction from messages.
        if request.method!='POST' or not request.url.path.endswith('/chat/completions'):
            raise AdmissionUnavailable('unsupported SDK operation')
        raw=request.read()
        try:
            payload=json.loads(raw)
            if not isinstance(payload,dict) or payload.get('model')!=self.model:
                raise ValueError('model mismatch')
            if type(payload.get('stream',False)) is not bool:
                raise ValueError('boolean stream flag required')
            if payload.get('stream') and (not isinstance(payload.get('stream_options'),dict) or payload['stream_options'].get('include_usage') is not True):
                raise ValueError('terminal streaming usage required')
            caps=[payload[k] for k in ('max_tokens','max_completion_tokens') if k in payload]
            if len(caps)!=1 or type(caps[0]) is not int or caps[0]<=0:
                raise ValueError('one explicit output cap required')
        except (ValueError,TypeError) as exc:
            raise AdmissionUnavailable('unsupported complete SDK request') from exc
        attempt=self.gateway.reserve_serialized(attempt_id=uuid.uuid4().hex,
            stage=self.stage,provider=self.provider,model=self.model,
            serialized_request=base64.b64encode(raw).decode('ascii'),max_output_tokens=caps[0])
        if not attempt:raise AdmissionUnavailable('serialized request was not admitted')
        self.gateway.start(attempt)
        response=None;settled=False
        try:
            # Headers and URL stay in the SDK transport, never in the ledger.
            response=self.inner.handle_request(request)
            if response.status_code<200 or response.status_code>=300:
                raise AdmissionUnavailable('provider response cannot prove usage')
            if payload.get('stream'):
                if 'text/event-stream' not in response.headers.get('content-type',''):
                    raise AdmissionUnavailable('unexpected streaming content type')
                return httpx.Response(response.status_code,headers=response.headers,
                    stream=UsageStream(response,self.gateway,attempt))
            body=response.read()
            envelope=json.loads(body)
            if not isinstance(envelope,dict) or not isinstance(envelope.get('usage'),dict):
                raise AdmissionUnavailable('raw provider usage unavailable')
            self.gateway.settle_raw(attempt,envelope['usage'])
            settled=True
            check_execution()
            return response
        except BaseException:
            if response is not None:response.close()
            if not settled:
                try:self.gateway.uncertain(attempt)
                except Exception:pass
            raise

    def close(self):self.inner.close()


class UsageStream(httpx.SyncByteStream):
    """Pass exact bytes through; settle on terminal usage followed by DONE, before SDK early-close.

    Early close, broken/malformed/oversized SSE or missing usage keeps uncertainty.
    """
    def __init__(self,response,gateway,attempt):
        self.response=response;self.gateway=gateway;self.attempt=attempt
        self.finished=False;self.marked=False;self.evidence=WireEvidence(gateway,attempt)

    def uncertain(self):
        if self.finished or self.marked:return
        self.marked=True
        try:self.gateway.uncertain(self.attempt)
        except Exception:pass

    def __iter__(self):
        lines=SSELines();data=[];size=0;usage=None;done=False
        try:
            for chunk in self.response.stream:
                self.evidence.chunk(chunk)
                for line in lines.feed(chunk):
                    size+=len(line)
                    if size>65536:raise AdmissionUnavailable('oversized provider SSE event')
                    if line.startswith(b'data:'):
                        data.append(line[5:].lstrip(b' '))
                    elif not line:
                        if data:
                            value=b'\n'.join(data);data=[];self.evidence.events+=1
                            if done:raise AdmissionUnavailable('provider event after DONE')
                            if value==b'[DONE]':
                                self.evidence.done=True
                                if usage is None:raise AdmissionUnavailable('DONE without terminal usage')
                                self.gateway.settle_raw(self.attempt,usage)
                                self.finished=True;done=True
                                from .stop_fence import check_execution
                                check_execution()
                            else:
                                event=json.loads(value)
                                if not isinstance(event,dict):raise AdmissionUnavailable('invalid provider SSE event')
                                if event.get('usage') is not None:
                                    if not isinstance(event['usage'],dict) or usage is not None:
                                        raise AdmissionUnavailable('ambiguous provider terminal usage')
                                    if not _terminal_usage_choices(event.get('choices')):raise AdmissionUnavailable('usage is not terminal')
                                    usage=event['usage'];self.evidence.usage=usage
                                elif usage is not None:raise AdmissionUnavailable('content after terminal usage')
                        size=0
                yield chunk
            if lines.buffer.strip() or data or not done or usage is None:
                raise AdmissionUnavailable('incomplete terminal provider usage')
        except BaseException as exc:
            self.evidence.emit('failed',exc)
            raise
        finally:
            try:self.evidence.emit('settled' if self.finished else 'closed_without_settlement')
            finally:
                try:self.uncertain()
                finally:self.response.close()

    def close(self):
        try:self.evidence.emit('settled' if self.finished else 'early_close')
        finally:
            try:self.uncertain()
            finally:self.response.close()


class AsyncAdmittedTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner, gateway, *, stage, provider, model):
        self.inner=inner;self.gateway=gateway
        self.stage=stage;self.provider=provider;self.model=model

    async def handle_async_request(self, request):
        from .stop_fence import check_execution
        check_execution()
        # This is the final SDK payload, not a reconstruction from messages.
        if request.method!='POST' or not request.url.path.endswith('/chat/completions'):
            raise AdmissionUnavailable('unsupported SDK operation')
        raw=await request.aread()
        try:
            payload=json.loads(raw)
            if not isinstance(payload,dict) or payload.get('model')!=self.model:
                raise ValueError('model mismatch')
            if type(payload.get('stream',False)) is not bool:
                raise ValueError('boolean stream flag required')
            if payload.get('stream') and (not isinstance(payload.get('stream_options'),dict) or payload['stream_options'].get('include_usage') is not True):
                raise ValueError('terminal streaming usage required')
            caps=[payload[k] for k in ('max_tokens','max_completion_tokens') if k in payload]
            if len(caps)!=1 or type(caps[0]) is not int or caps[0]<=0:
                raise ValueError('one explicit output cap required')
        except (ValueError,TypeError) as exc:
            raise AdmissionUnavailable('unsupported complete SDK request') from exc
        attempt=self.gateway.reserve_serialized(attempt_id=uuid.uuid4().hex,
            stage=self.stage,provider=self.provider,model=self.model,
            serialized_request=base64.b64encode(raw).decode('ascii'),max_output_tokens=caps[0])
        if not attempt:raise AdmissionUnavailable('serialized request was not admitted')
        self.gateway.start(attempt)
        response=None;settled=False
        try:
            # Headers and URL stay in the SDK transport, never in the ledger.
            response=await self.inner.handle_async_request(request)
            if response.status_code<200 or response.status_code>=300:
                raise AdmissionUnavailable('provider response cannot prove usage')
            if payload.get('stream'):
                if 'text/event-stream' not in response.headers.get('content-type',''):
                    raise AdmissionUnavailable('unexpected streaming content type')
                return httpx.Response(response.status_code,headers=response.headers,
                    stream=AsyncUsageStream(response,self.gateway,attempt))
            body=await response.aread()
            envelope=json.loads(body)
            if not isinstance(envelope,dict) or not isinstance(envelope.get('usage'),dict):
                raise AdmissionUnavailable('raw provider usage unavailable')
            self.gateway.settle_raw(attempt,envelope['usage'])
            settled=True
            check_execution()
            return response
        except BaseException:
            if response is not None:await response.aclose()
            if not settled:
                try:self.gateway.uncertain(attempt)
                except Exception:pass
            raise

    async def aclose(self):await self.inner.aclose()


class AsyncUsageStream(httpx.AsyncByteStream):
    """Pass exact bytes through; settle on terminal usage followed by DONE, before SDK early-close.

    Early close, broken/malformed/oversized SSE or missing usage keeps uncertainty.
    """
    def __init__(self,response,gateway,attempt):
        self.response=response;self.gateway=gateway;self.attempt=attempt
        self.finished=False;self.marked=False;self.evidence=WireEvidence(gateway,attempt)

    def uncertain(self):
        if self.finished or self.marked:return
        self.marked=True
        try:self.gateway.uncertain(self.attempt)
        except Exception:pass

    async def __aiter__(self):
        lines=SSELines();data=[];size=0;usage=None;done=False
        try:
            async for chunk in self.response.stream:
                self.evidence.chunk(chunk)
                for line in lines.feed(chunk):
                    size+=len(line)
                    if size>65536:raise AdmissionUnavailable('oversized provider SSE event')
                    if line.startswith(b'data:'):
                        data.append(line[5:].lstrip(b' '))
                    elif not line:
                        if data:
                            value=b'\n'.join(data);data=[];self.evidence.events+=1
                            if done:raise AdmissionUnavailable('provider event after DONE')
                            if value==b'[DONE]':
                                self.evidence.done=True
                                if usage is None:raise AdmissionUnavailable('DONE without terminal usage')
                                self.gateway.settle_raw(self.attempt,usage)
                                self.finished=True;done=True
                                from .stop_fence import check_execution
                                check_execution()
                            else:
                                event=json.loads(value)
                                if not isinstance(event,dict):raise AdmissionUnavailable('invalid provider SSE event')
                                if event.get('usage') is not None:
                                    if not isinstance(event['usage'],dict) or usage is not None:
                                        raise AdmissionUnavailable('ambiguous provider terminal usage')
                                    if not _terminal_usage_choices(event.get('choices')):raise AdmissionUnavailable('usage is not terminal')
                                    usage=event['usage'];self.evidence.usage=usage
                                elif usage is not None:raise AdmissionUnavailable('content after terminal usage')
                        size=0
                yield chunk
            if lines.buffer.strip() or data or not done or usage is None:
                raise AdmissionUnavailable('incomplete terminal provider usage')
        except BaseException as exc:
            self.evidence.emit('failed',exc)
            raise
        finally:
            try:self.evidence.emit('settled' if self.finished else 'closed_without_settlement')
            finally:
                try:self.uncertain()
                finally:await self.response.aclose()

    async def aclose(self):
        try:self.evidence.emit('settled' if self.finished else 'early_close')
        finally:
            try:self.uncertain()
            finally:await self.response.aclose()
