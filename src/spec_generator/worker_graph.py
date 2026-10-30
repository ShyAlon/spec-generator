"""Explicit fresh graph entrypoint for the private reusable worker.

Not the process-oriented CLI. Resume, rewind, gate decisions and repair commands
remain unavailable here until their revision-safe lifecycle is installed.
"""
from copy import deepcopy
from vmodel_contracts import IntentModel,ConOpsDocument,SystemBaseline,check_compatible
from .cooperative_run import cli_stop_binding
from .worker_runtime import current_runtime
from .stop_fence import guarded_publish


def run_fresh_graph(*,capability,session_root,intent,conops,baseline,log,progress,stop_after,graph_factory=None):
    """Run an explicitly seeded fresh needs/SRS checkpoint, no auto-resume.

    The caller supplies the already-bound worker scope, exact input documents,
    reviewed runtime, progress consumer and accounting log. No user path, argv,
    ambient output directory, console replacement or environment write is used.
    """
    from .graph import build_graph
    from .graph import checkpoint_serde
    from .artifact_lifecycle import ArchitectureLifecycle
    from .cli import _make_partials,_make_persist
    binding=cli_stop_binding();runtime=current_runtime()
    if binding is None or runtime is None:raise ValueError('private bound job runtime required')
    if stop_after not in ('needs','srs'):raise ValueError('fresh worker currently supports needs/SRS only')
    fence,generation=binding
    if generation!=capability.generation:raise ValueError('worker graph generation mismatch')
    fence.check(generation)
    for document in (intent,conops,baseline):check_compatible(document)
    intent=IntentModel.model_validate(intent).model_dump(mode='json')
    conops=ConOpsDocument.model_validate(conops).model_dump(mode='json')
    baseline=SystemBaseline.model_validate(baseline).model_dump(mode='json')
    if intent['session_id']!=capability.session_id or conops['session_id']!=capability.session_id:
        raise ValueError('worker graph session mismatch')
    if not conops['decision']['founder_confirmed']:raise ValueError('confirmed ConOps required')
    out=session_root/'spec'
    # Do not silently consume or overwrite any prior producer artifacts.
    from .document_runtime import configured as documents_configured
    if not documents_configured() and out.exists():
        entries=list(out.iterdir())
        retained_baseline=out/'system-baseline.json'
        if entries:
            import json
            if (entries!=[retained_baseline] or retained_baseline.resolve()!=retained_baseline
                or not retained_baseline.is_file() or retained_baseline.stat().st_size>2*1024*1024
                or SystemBaseline.model_validate(json.loads(retained_baseline.read_text())).model_dump(mode='json')!=baseline):
                raise ValueError('fresh worker output is not empty; explicit resume required')
    if not documents_configured():guarded_publish(lambda:out.mkdir(parents=True,exist_ok=True))
    from .checkpoint_runtime import checkpoint as open_checkpoint
    saver=guarded_publish(lambda:open_checkpoint(out))
    if documents_configured():
        from .document_runtime import exists
        if saver.get_tuple({'configurable':{'thread_id':capability.session_id}}) is not None or any(exists(out/name) for name in ('needs.json','srs.json','architecture.json','items.json','tests.json','validation.json','artifact-lifecycle.jsonl','lineage.json')):
            raise ValueError('fresh worker output is not empty; explicit resume required')
    lifecycle=ArchitectureLifecycle(out,capability.session_id,baseline)
    graph=(graph_factory or build_graph)(log,checkpointer=saver,
        progress=progress,review=False,stop_after=stop_after,
        stop_fence=fence,stop_generation=generation,
        persist=_make_persist(intent,out,greenfield=baseline['mode']=='greenfield',conops_present=True,architecture_lifecycle=lifecycle),
        partials=_make_partials(out,lifecycle))
    config={'configurable':{'thread_id':capability.session_id}}
    result=graph.invoke(deepcopy({'intent_model':intent,'conops':conops,'system_baseline':baseline}),config)
    fence.check(generation)
    return result
