import pytest
from spec_generator.revision_source import read_closed_checkpoint
from test_revision_worker import setup


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
