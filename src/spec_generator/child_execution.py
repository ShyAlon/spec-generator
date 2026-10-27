"""Private fresh-process generator graph identity and durable gate suspension."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib,json

_execution=ContextVar('child_generator_execution',default=None)
_call=ContextVar('child_generator_call',default=None)

@contextmanager
def graph_execution(graph,cfg):
    state=graph.get_state(cfg)
    anchor=((getattr(state,'config',None) or {}).get('configurable') or {}).get('checkpoint_id') or 'bootstrap'
    from .worker_runtime import current_runtime
    runtime=current_runtime()
    if runtime is not None and runtime.worker_binding is not None:
        owner,task,job,generation,root=runtime.worker_binding
        # Graph checkpoint latest can advance while an interrupted node has not
        # completed. Job + cumulative Pregel step/node survives that crash.
        # A new explicit job always has another namespace. No random retry ID.
        anchor='job:'+job+':generation:'+str(generation)
    token=_execution.set((str((cfg.get('configurable') or {}).get('thread_id')),str(anchor),{}))
    try:yield
    finally:_execution.reset(token)


def invoke_graph(graph,value,cfg):
    from .wire_binding import broker_active
    if not broker_active():return graph.invoke(value,cfg)
    with graph_execution(graph,cfg):return graph.invoke(value,cfg)

@contextmanager
def invocation(stage):
    context=_execution.get()
    if context is None:token=_call.set(None)
    else:
        sid,anchor,cursors=context
        from langchain_core.runnables.config import var_child_runnable_config
        configurable=(var_child_runnable_config.get(None) or {}).get('configurable') or {}
        if configurable:
            scratch=configurable.get('__pregel_scratchpad');step=getattr(scratch,'step',None)
            node=(configurable.get('checkpoint_ns') or '').split(':')[0]
            if type(step) is not int or step<0 or not node:raise PermissionError('generator durable graph cursor unreadable')
        else:step=0;node=''
        key=(anchor,step,node,stage);ordinal=cursors.get(key,0);cursors[key]=ordinal+1
        identity=[sid,anchor,step,node,stage,ordinal]
        token=_call.set((stage,'call:'+hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest()))
    try:yield
    finally:_call.reset(token)


def broker_call_id(stage):
    value=_call.get()
    if value is None or value[0]!=stage:raise PermissionError('generator durable graph cursor required')
    return value[1]


class GateSuspended(Exception):
    def __init__(self,decision,key):self.decision=decision;self.key=key


def gate_key(decision):
    return 'gate:'+hashlib.sha256(json.dumps(decision,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class DurableGate:
    """No queue/input in child. Parent persists answer before fresh execution."""
    def __init__(self,answers,publish):self.answers=dict(answers);self.publish=publish
    def __call__(self,decision):
        key=gate_key(decision)
        if key in self.answers:
            answer=self.answers[key]
            if not isinstance(answer,str):raise PermissionError('durable gate answer must be text')
            return answer
        self.publish({'kind':'gate','key':key,'decision':decision})
        raise GateSuspended(decision,key)
