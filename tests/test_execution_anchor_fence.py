import json
import pytest
from vmodel_contracts.execution_anchor import ExecutionAnchor
from spec_generator.stop_fence import StopFence,StopRequested


def setup(tmp_path):
    anchor=ExecutionAnchor('00000000-0000-0000-0000-000000000001','task','execution2','semantic',2,'a'*64,'installation')
    root=tmp_path/'tenants'/anchor.owner_id/'sessions'/anchor.execution_session_id;(root/'interview').mkdir(parents=True)
    path=root/'execution-generation.json';path.write_text(json.dumps(anchor.document()))
    return anchor,root,path,root/'interview'/'stop-request.json'


def test_anchor_initializes_later_generation_and_stop_stays_one_way(tmp_path):
    anchor,root,path,marker=setup(tmp_path);fence=StopFence(marker,execution_anchor=anchor)
    assert fence.generation==2 and not fence.requested
    with fence.running() as generation:assert generation==2
    fence.request_stop();assert fence.generation==3 and fence.requested
    recovered=StopFence(marker,execution_anchor=anchor)
    assert recovered.generation==3 and recovered.requested
    with pytest.raises(StopRequested):recovered.check(2)


@pytest.mark.parametrize('generation,requested',[(1,True),(2,True),(4,True),(2,False)])
def test_stale_or_false_marker_cannot_initialize_lineage(tmp_path,generation,requested):
    anchor,root,path,marker=setup(tmp_path);marker.write_text(json.dumps({'generation':generation,'requested':requested}))
    with pytest.raises(ValueError):StopFence(marker,execution_anchor=anchor)


def test_missing_anchor_never_falls_back_to_generation_one(tmp_path):
    anchor,root,path,marker=setup(tmp_path);path.unlink()
    with pytest.raises(FileNotFoundError):StopFence(marker,execution_anchor=anchor)


def test_anchor_mutation_or_removal_blocks_running_fence(tmp_path):
    anchor,root,path,marker=setup(tmp_path);fence=StopFence(marker,execution_anchor=anchor)
    path.write_text(json.dumps(anchor.document(),indent=2))
    with pytest.raises(ValueError,match='bytes changed'):fence.check(2)
    path.unlink()
    with pytest.raises(FileNotFoundError):fence.check(2)


def test_stop_marker_removal_after_stop_cannot_restore_execution(tmp_path):
    anchor,root,path,marker=setup(tmp_path);fence=StopFence(marker,execution_anchor=anchor)
    fence.request_stop();marker.unlink()
    with pytest.raises(ValueError,match='disappeared'):fence.check(2)


def test_legacy_default_fence_still_generation_one(tmp_path):
    fence=StopFence(tmp_path/'stop.json');assert fence.generation==1
    with fence.running() as generation:assert generation==1


def test_lineage_namespace_without_explicit_binding_cannot_use_legacy_default(tmp_path):
    anchor,root,path,marker=setup(tmp_path)
    with pytest.raises(ValueError,match='explicit installed execution binding'):
        StopFence(marker)
