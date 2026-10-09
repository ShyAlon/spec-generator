"""Pinned read-only revision checkpoint snapshot, closed-WAL sources only.

Caller holds the same-host writer lease. Active WAL sources are rejected rather
than reading a partial main file or reopening an unpinned path.
"""
import os,stat,sqlite3
from pathlib import Path
from copy import deepcopy


def _directory(path):
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts:raise ValueError('canonical absolute checkpoint directory required')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for name in path.parts[1:]:
            child=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=child
        return fd
    except BaseException:os.close(fd);raise


def _stamp(info):return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns


def read_closed_checkpoint(path,*,session_id,expected_checkpoint_id):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from .graph import checkpoint_serde
    path=Path(path)
    if path.name!='checkpoints.sqlite':raise ValueError('canonical source checkpoint required')
    if not Path('/proc/self/fd').is_dir():raise ValueError('pinned SQLite reader requires proc descriptor support')
    directory=_directory(path.parent);fd=None
    try:
        def no_wal():
            for name in ('checkpoints.sqlite-wal','checkpoints.sqlite-shm','checkpoints.sqlite-journal'):
                try:os.stat(name,dir_fd=directory,follow_symlinks=False)
                except FileNotFoundError:continue
                raise ValueError('active checkpoint sidecar requires explicit quiescent snapshot')
        no_wal()
        fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):raise ValueError('regular source checkpoint required')
        # Immutable mode is safe only with no sidecars and the writer lease.
        # It cannot create SHM/journals or read a different path target.
        conn=sqlite3.connect(f'file:/proc/self/fd/{fd}?mode=ro&immutable=1',uri=True,check_same_thread=False)
        try:
            row=SqliteSaver(conn,serde=checkpoint_serde()).get_tuple({'configurable':{'thread_id':session_id}})
            if row is None or row.config['configurable']['checkpoint_id']!=expected_checkpoint_id:
                raise ValueError('reviewed source checkpoint changed')
            result=deepcopy(row.checkpoint['channel_values'])
        finally:conn.close()
        no_wal()
        if _stamp(os.fstat(fd))!=_stamp(before) or _stamp(os.stat(path.name,dir_fd=directory,follow_symlinks=False))!=_stamp(before):
            raise ValueError('source checkpoint changed during pinned read')
        current=_directory(path.parent)
        try:
            if _stamp(os.fstat(current))[:2]!=_stamp(os.fstat(directory))[:2]:
                raise ValueError('checkpoint directory namespace changed')
        finally:os.close(current)
        return result
    finally:
        if fd is not None:os.close(fd)
        os.close(directory)
