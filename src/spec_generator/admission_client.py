"""Local inherited-descriptor worker gateway. No reconnect or permissive retry."""
import json,os,socket,threading
from .attempt_admission import AdmissionUnavailable,AdmissionRefused

class SocketAdmissionGateway:
    def __init__(self,fd):
        self.socket=socket.socket(fileno=os.dup(fd))
        self.socket.settimeout(10)
        self.stream=self.socket.makefile('rwb');self.lock=threading.Lock()

    def _request(self,action,payload):
        try:
            raw=json.dumps({'action':action,'payload':payload}).encode()+b'\n'
            if len(raw)>1024*1024:raise ValueError('request too large')
            with self.lock:
                self.stream.write(raw);self.stream.flush()
                response=self.stream.readline(1024*1024+1)
            if len(response)>1024*1024 or not response.endswith(b'\n'):raise ValueError('invalid frame')
            reply=json.loads(response)
            if not isinstance(reply,dict) or set(reply)-{'ok','result','error','reason'}:raise ValueError('invalid reply')
            if reply.get('ok') is not True:
                reason=reply.get('reason');reason=reason[:300] if isinstance(reason,str) else ''
                raise AdmissionRefused('worker attempt admission refused'+(': '+reason if reason else ' (the server gave no reason)'),reason)
            return reply.get('result')
        except AdmissionRefused:raise
        except Exception as exc:raise AdmissionUnavailable('worker attempt admission unavailable or denied: '+type(exc).__name__+(': '+str(exc)[:200] if str(exc) else '')) from exc

    def reserve(self,**payload):return self._request('reserve',payload)
    def reserve_serialized(self,**payload):return self._request('reserve_serialized',payload)
    def settle_raw(self,attempt,raw_usage):return self._request('settle_raw',{'attempt':attempt,'raw_usage':raw_usage})
    def start(self,attempt):return self._request('start',{'attempt':attempt})
    def uncertain(self,attempt):return self._request('uncertain',{'attempt':attempt})
    def settle(self,attempt,response):return self._request('settle',{'attempt':attempt,'usage':getattr(response,'usage_metadata',None)})
    def close(self):self.stream.close();self.socket.close()
