"""Per-test graph reference checkpoints, not a durable storage backend.

Keys let separately constructed graphs share a reference saver in one test.
Cross-process persistence and fencing belong to delivery's real PG suites.
"""
from langgraph.checkpoint.memory import MemorySaver

_SAVERS={}


def reference_checkpoint(key, *, serde=None):
    key=str(key)
    if key not in _SAVERS:
        _SAVERS[key]=MemorySaver(serde=serde)
    return _SAVERS[key]


def clear_reference_checkpoints():
    _SAVERS.clear()
