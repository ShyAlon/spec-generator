"""Checkpoint backend supplied by the authenticated host, not environment/path.

Standalone execution is retired. A bound job requires its host factory.
Namespace is an internal relative workflow name derived only inside its already validated session root.
"""
from pathlib import Path

def configured():
    from .worker_runtime import current_runtime
    r=current_runtime()
    return r is not None and r.checkpoint_factory is not None

def checkpoint(outdir):
    from .worker_runtime import current_runtime
    r=current_runtime()
    if r is not None and r.checkpoint_factory is not None:
        if r.worker_binding is None:raise ValueError('bound job checkpoint runtime required')
        root=Path(r.worker_binding[4]);out=Path(outdir)
        if out.resolve()!=out or not out.is_relative_to(root):raise ValueError('checkpoint outside owned session')
        parts=out.relative_to(root).parts
        import re
        if not parts or any(not re.fullmatch('[A-Za-z0-9_-]+',part) for part in parts):raise ValueError('canonical checkpoint workflow namespace required')
        namespace=':'.join(parts)
        saver=r.checkpoint_factory(namespace)
        saver.setup()
        return saver
    raise ValueError('authenticated host checkpoint runtime required; run through delivery-api')

def read_state(path,*,session_id,expected_checkpoint_id,expected_review_phase=None):
    if not configured():
        raise ValueError('authenticated host checkpoint runtime required; run through delivery-api')
    if expected_review_phase is not None and expected_review_phase not in ('srs','architecture'):raise ValueError('implemented staged review phase required')
    saver=checkpoint(Path(path).parent)
    config={'configurable':{'thread_id':session_id}}
    row=saver.get_tuple(config)
    if row is None or row.config['configurable']['checkpoint_id']!=expected_checkpoint_id:
        raise ValueError('reviewed source checkpoint changed')
    from copy import deepcopy
    if expected_review_phase is not None:
        from .graph import build_graph
        from types import SimpleNamespace
        graph=build_graph(SimpleNamespace(record=lambda *args:None),review=True,checkpointer=saver)
        state=graph.get_state(row.config)
        if state.next!=('gate_'+expected_review_phase,):raise ValueError('staged checkpoint must await fresh edited gate')
        if saver.get_tuple(config)!=row:raise ValueError('reviewed source checkpoint changed across read')
        return deepcopy(state.values)
    return deepcopy(row.checkpoint['channel_values'])
