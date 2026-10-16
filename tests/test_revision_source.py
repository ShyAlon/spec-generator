import pytest
from spec_generator.revision_source import read_closed_checkpoint
from test_revision_worker import setup
import os

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="pinned SQLite reader requires /proc/self/fd (Linux); covered by CI")


def test_pinned_checkpoint_returns_exact_state_without_new_sidecars(tmp_path):
    cap,root,runtime,cid,state=setup(tmp_path);path=root/'spec'/'checkpoints.sqlite'
    before=path.read_bytes()
    assert read_closed_checkpoint(path,session_id='s',expected_checkpoint_id=cid)==state
    assert path.read_bytes()==before and not path.with_name('checkpoints.sqlite-wal').exists()


@pytest.mark.parametrize('kind',['wal','shm','journal','symlink'])
def test_checkpoint_reader_refuses_unclosed_sidecars_and_symlink_sources(tmp_path,kind):
    cap,root,runtime,cid,state=setup(tmp_path);path=root/'spec'/'checkpoints.sqlite'
    if kind=='symlink':path.rename(path.with_suffix('.old'));path.symlink_to(path.with_suffix('.old'))
    else:path.with_name(path.name+'-'+kind).write_bytes(b'fake')
    with pytest.raises((ValueError,OSError)):
        read_closed_checkpoint(path,session_id='s',expected_checkpoint_id=cid)


def test_checkpoint_namespace_swap_after_open_is_rejected(tmp_path,monkeypatch):
    from spec_generator import revision_source as module
    cap,root,runtime,cid,state=setup(tmp_path);path=root/'spec'/'checkpoints.sqlite'
    original=module.os.open;swapped=[]
    def read(name,flags,*args,**kwargs):
        fd=original(name,flags,*args,**kwargs)
        if name=='checkpoints.sqlite' and not swapped:
            (root/'spec').rename(root/'old-spec');(root/'spec').mkdir();swapped.append(True)
        return fd
    monkeypatch.setattr(module.os,'open',read)
    with pytest.raises(ValueError,match='namespace changed'):
        read_closed_checkpoint(path,session_id='s',expected_checkpoint_id=cid)


def test_checkpoint_same_inode_edit_during_read_is_rejected(tmp_path,monkeypatch):
    from langgraph.checkpoint.sqlite import SqliteSaver
    cap,root,runtime,cid,state=setup(tmp_path);path=root/'spec'/'checkpoints.sqlite'
    original=SqliteSaver.get_tuple
    def changed(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        with path.open('ab') as stream:stream.write(b'changed')
        return result
    monkeypatch.setattr(SqliteSaver,'get_tuple',changed)
    with pytest.raises(ValueError,match='changed during pinned read'):
        read_closed_checkpoint(path,session_id='s',expected_checkpoint_id=cid)


def test_symlink_checkpoint_ancestor_is_rejected(tmp_path):
    cap,root,runtime,cid,state=setup(tmp_path)
    (root/'spec').rename(root/'old-spec');(root/'spec').symlink_to(root/'old-spec',target_is_directory=True)
    with pytest.raises(OSError):
        read_closed_checkpoint(root/'spec'/'checkpoints.sqlite',session_id='s',expected_checkpoint_id=cid)


def test_pinned_staged_sqlite_readback_requires_exact_edited_gate(tmp_path):
    from spec_generator.graph import build_graph,checkpoint_serde
    from spec_generator.revision_checkpoint import prepare_revision_seed,validate_revision_output,stage_reviewed_lineage_checkpoint,validate_staged_lineage_state
    from test_revision_lineage_checkpoint import lineage,fence
    from langgraph.checkpoint.sqlite import SqliteSaver
    from spec_generator.cooperative_run import cli_execution
    import sqlite3
    from types import SimpleNamespace
    cap,root,runtime,cid,state=setup(tmp_path)
    result=prepare_revision_seed(source=state,phase='srs',change_request='Describe offline behavior',session_id='s')['seed']
    result['srs']=state['srs'];result['gate_feedback'].pop('srs')
    digest=validate_revision_output(source=state,result=result,phase='srs',change_request='Describe offline behavior',session_id='s')['artifact_sha256']
    binding=lineage()|{'regenerated_artifact_sha256':digest}
    target=tmp_path/'staged'/'checkpoints.sqlite';target.parent.mkdir()
    conn=sqlite3.connect(target);g=build_graph(SimpleNamespace(record=lambda *a:None),review=True,checkpointer=SqliteSaver(conn,serde=checkpoint_serde()))
    with cli_execution(fence(tmp_path),None):
        receipt=stage_reviewed_lineage_checkpoint(source=state,proposal_result=result,lineage=binding,change_request='Describe offline behavior',graph=g,config={'configurable':{'thread_id':'next'}})
    conn.close();checkpoint=receipt['destination_checkpoint']['configurable']['checkpoint_id']
    read=read_closed_checkpoint(target,session_id='next',expected_checkpoint_id=checkpoint,expected_review_phase='srs')
    assert validate_staged_lineage_state(source=state,staged_state=read,lineage=binding,change_request='Describe offline behavior')['approved_upstream_unchanged']
    with pytest.raises(ValueError,match='fresh edited gate'):
        read_closed_checkpoint(target,session_id='next',expected_checkpoint_id=checkpoint,expected_review_phase='architecture')


def test_arbitrary_checkpoint_cannot_claim_staged_gate_readiness(tmp_path):
    cap,root,runtime,cid,state=setup(tmp_path)
    with pytest.raises(ValueError,match='fresh edited gate'):
        read_closed_checkpoint(root/'spec'/'checkpoints.sqlite',session_id='s',expected_checkpoint_id=cid,expected_review_phase='srs')
