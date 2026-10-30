"""Host-owned workflow documents. A path is a validated logical key only."""
from pathlib import Path

def configured():
    from .worker_runtime import current_runtime
    runtime=current_runtime()
    return runtime is not None and runtime.documents is not None

def _bound(path):
    from .worker_runtime import current_runtime
    runtime=current_runtime()
    if runtime is None or runtime.documents is None or runtime.worker_binding is None:raise ValueError('bound host documents required')
    root=Path(runtime.worker_binding[4]);path=Path(path)
    if path.resolve()!=path or not path.is_relative_to(root):raise ValueError('document outside owned session')
    key=path.relative_to(root).as_posix()
    import re
    if len(key)>500 or not key or any(not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,199}',part) for part in key.split('/')):
        raise ValueError('canonical workflow document key required')
    return runtime.documents,key

def read(path):
    documents,key=_bound(path);return documents.read(key)

def write(path,raw):
    documents,key=_bound(path);return documents.write(key,raw)

def clear(path):
    documents,key=_bound(path);return documents.clear(key)


def read_bytes(path):
    if configured():
        raw=read(path)
        if raw is None:raise FileNotFoundError(str(path))
        return raw
    return Path(path).read_bytes()

def read_text(path):return read_bytes(path).decode('utf-8')

def exists(path):
    return read(path) is not None if configured() else Path(path).is_file()

def append(path,raw):
    documents,key=_bound(path);return documents.append(key,raw)

def atomic_text(path,text):
    from .stop_fence import guarded_publish
    if configured():return guarded_publish(lambda:write(path,text.encode('utf-8')))
    import os
    path=Path(path)
    def publish():
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+'.tmp');tmp.write_text(text,encoding='utf-8');os.replace(tmp,path)
    return guarded_publish(publish)

def remove(path):
    from .stop_fence import guarded_publish
    if configured():return guarded_publish(lambda:clear(path))
    return guarded_publish(lambda:Path(path).unlink(missing_ok=True))


def plain_log(path,stream):
    """Preserve the terminal, retain plain diagnostic bytes in the bound backend."""
    from vmodel_contracts.runtime import install_plain_log,PlainLogMirror
    if not configured():return install_plain_log(path,stream)
    # Capture the actual bound backend, not the ContextVar at a later flush.
    documents,key=_bound(path)
    class Sink:
        def write(self,text):documents.append(key,text.encode('utf-8'));return len(text)
        def flush(self):pass
    return PlainLogMirror(stream,Sink())
