"""Private closed-phase proposal runner. Never installs artifacts or restarts tasks."""
from copy import deepcopy
from pathlib import Path
import uuid
from .revision_checkpoint import prepare_revision_seed,seed_fresh_revision_checkpoint,validate_revision_output


def run_revision_proposal(*,capability,session_root,checkpoint_id,phase,change_request,
                          validate_authority,log,progress=None,graph_factory=None):
    """Read exact owned source read-only, run a fresh fenced proposal checkpoint.

    Caller must supply the captured worker capability and a current authority
    callback, under its private job runtime/admission/fence. Proposal output is
    isolated from live artifacts. No retry, source mutation or acceptance.
    """
    from .cooperative_run import cli_stop_binding
    from .worker_runtime import current_runtime
    from .stop_fence import guarded_publish
    from .stop_publication import FencedCheckpointer,publication_guard
    from .graph import build_graph,checkpoint_serde
    binding=cli_stop_binding();runtime=current_runtime()
    if binding is None or runtime is None or not callable(validate_authority):
        raise ValueError('bound private worker runtime and current authority required')
    root=Path(session_root)
    owner=capability.scope.owner_id
    if (not root.is_absolute() or root.resolve()!=root or
        root.parts[-4:]!=('tenants',owner,'sessions',capability.session_id)):
        raise ValueError('exact owned revision session namespace required')
    fence,generation=binding
    if generation!=capability.generation:raise ValueError('revision worker generation mismatch')
    if Path(fence.path)!=root/'interview'/'stop-request.json':raise ValueError('revision fence belongs to another session')
    if phase not in runtime.models:raise ValueError('explicit revision phase model required')
    expected_binding=(owner,capability.task_id,runtime.worker_binding[2] if runtime.worker_binding else None,generation,str(root))
    if runtime.worker_binding!=expected_binding:raise ValueError('revision runtime belongs to another worker')
    anchor=getattr(fence,'_execution_anchor',None)
    if generation>1:
        if (anchor is None or anchor.owner_id!=owner or anchor.task_id!=capability.task_id
            or anchor.execution_session_id!=capability.session_id or anchor.generation!=generation):
            raise ValueError('installed proposal semantic/execution binding required')
        semantic_session_id=anchor.semantic_session_id
    else:
        if anchor is not None:raise ValueError('legacy proposal must not carry later-generation anchor')
        semantic_session_id=capability.session_id
    source_path=root/'spec'/'checkpoints.sqlite'
    config={'configurable':{'thread_id':capability.session_id}}
    def source():
        validate_authority();fence.check(generation)
        from .checkpoint_runtime import read_state as read_closed_checkpoint
        return read_closed_checkpoint(source_path,session_id=capability.session_id,expected_checkpoint_id=checkpoint_id)
    original=source()
    prepare_revision_seed(source=original,phase=phase,change_request=change_request,session_id=semantic_session_id)
    parent=root/'spec'/'revision-proposals'
    if parent.resolve()!=parent:raise ValueError('unsafe revision proposal directory')
    if runtime.proposal_sink is None:guarded_publish(lambda:parent.mkdir(exist_ok=True))
    destination=parent/uuid.uuid4().hex
    if runtime.proposal_sink is None:guarded_publish(lambda:destination.mkdir())
    from .checkpoint_runtime import checkpoint
    backend=guarded_publish(lambda:checkpoint(destination))
    saver=FencedCheckpointer(backend,publication_guard(fence,generation))
    graph=(graph_factory or build_graph)(log,checkpointer=saver,progress=progress,
        review=False,stop_after=phase,stop_fence=fence,stop_generation=generation)
    seed_fresh_revision_checkpoint(source=original,source_checkpoint_id=checkpoint_id,expected_checkpoint_id=checkpoint_id,
        phase=phase,change_request=change_request,session_id=semantic_session_id,execution_session_id=capability.session_id,graph=graph,config=config)
    result=graph.invoke(None,config)
    fence.check(generation)
    # Graph SRS channel is a freshly assembled body, while on-disk artifacts
    # are serialized contracts. Serialize this new body through the current
    # model; never change an explicit old stamp or migrate source evidence.
    if phase=='srs' and isinstance(result.get('srs'),dict) and 'contract_version' not in result['srs']:
        from vmodel_contracts import SRSDocument
        from .graph import srs_document_data
        result={**result,'srs':SRSDocument.model_validate(srs_document_data(result['intent_model'],result['srs'])).model_dump(mode='json')}
    proposal=validate_revision_output(source=original,result=result,phase=phase,
        change_request=change_request,session_id=semantic_session_id)
    # This detects source drift, not an atomic evidence CAS. Commit remains
    # disabled until the owner-side transaction compares the full evidence.
    if source()!=original:raise ValueError('source checkpoint content changed during revision')
    snapshot=graph.get_state(config)
    import json,os
    body={**proposal,'proposal_id':destination.name,'task_id':capability.task_id,
        'session_id':capability.session_id,'generation':generation,'source_checkpoint_id':checkpoint_id,
        'change_request':change_request.strip(),'proposal_checkpoint_id':snapshot.config['configurable']['checkpoint_id']}
    def publish():
        validate_authority();fence.check(generation)
        if runtime.proposal_sink is not None:
            if not callable(runtime.proposal_sink):raise ValueError('bound proposal sink required')
            runtime.proposal_sink(body);return
        with (destination/'proposal.json').open('x') as file:
            json.dump(body,file,sort_keys=True,indent=2);file.flush();os.fsync(file.fileno())
        fd=os.open(destination,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    guarded_publish(publish)
    return body
