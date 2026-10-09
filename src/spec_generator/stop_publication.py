"""Same-host producer sink adapters. Runner must supply an active StopFence.

Capture generation explicitly: LangGraph checkpoint writes can run on threads
whose ContextVars are not inherited. Reads never restore or clear Stop.
"""
from langgraph.checkpoint.base import BaseCheckpointSaver


def publication_guard(fence, generation):
    def publish(call):
        with fence.marker_guard():
            fence.check(generation)
            return call()
    return publish


class FencedSink:
    def __init__(self, inner, publish):
        self.inner=inner;self.publish=publish

    def __call__(self,*args,**kwargs):
        return self.publish(lambda:self.inner(*args,**kwargs))

    def __getattr__(self,name):
        value=getattr(self.inner,name)
        if not callable(value):return value
        # load is read-only; all named persistence hooks are writes.
        if name=='load':return value
        return lambda *args,**kwargs:self.publish(lambda:value(*args,**kwargs))


class FencedCheckpointer(BaseCheckpointSaver):
    def __init__(self,inner,publish):
        super().__init__(serde=inner.serde)
        self.inner=inner;self.publish=publish

    @property
    def config_specs(self):return self.inner.config_specs
    def get_tuple(self,config):return self.inner.get_tuple(config)
    def list(self,*args,**kwargs):return self.inner.list(*args,**kwargs)
    def get_next_version(self,*args,**kwargs):return self.inner.get_next_version(*args,**kwargs)
    def put(self,*args,**kwargs):return self.publish(lambda:self.inner.put(*args,**kwargs))
    def put_writes(self,*args,**kwargs):return self.publish(lambda:self.inner.put_writes(*args,**kwargs))
    # No delete/copy/prune or async fallback: unsupported operations fail closed.
