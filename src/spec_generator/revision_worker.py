"""Private closed-phase proposal runner. Never installs artifacts or restarts tasks."""
from copy import deepcopy
from pathlib import Path
import sqlite3,uuid
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
    from langgraph.checkpoint.sqlite import SqliteSaver
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
    source_path=root/'spec'/'checkpoints.sqlite'
    config={'configurable':{'thread_id':capability.session_id}}
    def source():
        validate_authority();fence.check(generation)
        from .revision_source import read_closed_checkpoint
        return read_closed_checkpoint(source_path,session_id=capability.session_id,expected_checkpoint_id=checkpoint_id)
    original=source()
    prepare_revision_seed(source=original,phase=phase,change_request=change_request,session_id=capability.session_id)
    parent=root/'spec'/'revision-proposals'
    if parent.resolve()!=parent:raise ValueError('unsafe revision proposal directory')
    guarded_publish(lambda:parent.mkdir(exist_ok=True))
    destination=parent/uuid.uuid4().hex
    guarded_publish(lambda:destination.mkdir())
    conn=guarded_publish(lambda:sqlite3.connect(str(destination/'checkpoints.sqlite'),check_same_thread=False))
    try:
        # No artifact persistence or prior partial reuse: the checkpoint holds a
        # private proposal only until typed validation and evidence CAS succeed.
        saver=FencedCheckpointer(SqliteSaver(conn,serde=checkpoint_serde()),publication_guard(fence,generation))
        graph=(graph_factory or build_graph)(log,checkpointer=saver,progress=progress,
            review=False,stop_after=phase,stop_fence=fence,stop_generation=generation)
        seed_fresh_revision_checkpoint(source=original,source_checkpoint_id=checkpoint_id,expected_checkpoint_id=checkpoint_id,
            phase=phase,change_request=change_request,session_id=capability.session_id,graph=graph,config=config)
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
            change_request=change_request,session_id=capability.session_id)
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
            with (destination/'proposal.json').open('x') as file:
                json.dump(body,file,sort_keys=True,indent=2);file.flush();os.fsync(file.fileno())
            fd=os.open(destination,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        guarded_publish(publish)
        return body
    finally:conn.close()
