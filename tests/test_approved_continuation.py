"""Real graph scheduling over a reference saver, no graph invoke/provider or source mutation."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from vmodel_contracts.execution_anchor import ExecutionAnchor
from spec_generator.graph import build_graph,checkpoint_serde
from spec_generator.stop_fence import StopFence
from spec_generator.cooperative_run import cli_execution
from spec_generator.revision_checkpoint import prepare_approved_continuation_checkpoint
from reference_checkpoint import reference_checkpoint


def fixture(tmp_path):
    source=json.loads((Path(__file__).parent/'fixtures/revision-source.json').read_text())
    source['gate_feedback'].pop('srs',None)
    anchor=ExecutionAnchor('00000000-0000-0000-0000-000000000001','task','execution2','s',2,'a'*64,'install')
    root=tmp_path/'tenants'/anchor.owner_id/'sessions'/anchor.execution_session_id
    (root/'interview').mkdir(parents=True);(root/'execution-generation.json').write_text(json.dumps(anchor.document()))
    key=tmp_path/'fresh-reference'
    graph=build_graph(SimpleNamespace(record=lambda *a:None),review=True,checkpointer=reference_checkpoint(key, serde=checkpoint_serde()))
    kwargs=dict(installed_state=source,execution_anchor=anchor,phase='srs',artifact_sha256=hashlib.sha256(json.dumps(source['srs'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        approval_reference='constructed-test-receipt',graph=graph,config={'configurable':{'thread_id':anchor.execution_session_id}},validate_authority=lambda:None)
    return StopFence(root/'interview/stop-request.json',execution_anchor=anchor),kwargs


def test_approved_gate_schedules_next_without_invocation(tmp_path):
    fence,kwargs=fixture(tmp_path)
    with cli_execution(fence,None):result=prepare_approved_continuation_checkpoint(**kwargs)
    assert result['next_node']=='architecture' and not result['provider_invoked'] and not result['dispatch_started']
    with cli_execution(fence,None):
        with pytest.raises(ValueError,match='must be empty'):prepare_approved_continuation_checkpoint(**kwargs)


def test_unapproved_or_stale_continuation_never_writes(tmp_path):
    fence,kwargs=fixture(tmp_path)
    with pytest.raises(ValueError,match='writer fence'):prepare_approved_continuation_checkpoint(**kwargs)
    with cli_execution(fence,None):
        with pytest.raises(ValueError,match='digest'):prepare_approved_continuation_checkpoint(**(kwargs|{'artifact_sha256':'f'*64}))
        def revoked():raise PermissionError('revoked')
        with pytest.raises(PermissionError):prepare_approved_continuation_checkpoint(**(kwargs|{'validate_authority':revoked}))
    assert not kwargs['graph'].get_state(kwargs['config']).values
