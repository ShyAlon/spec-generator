"""Where the generator command reads answers and writes text.

By default this is the process's own stdin, stdout and stderr, so the command line behaves exactly as before.
A trusted caller that runs the generator inside its own process (the hosted service) binds its own streams and its
own answer handler for one run with bind(). The binding is per run (a context variable), never process-wide, and a
run that was not bound can never see another run's streams."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import sys


@dataclass(frozen=True)
class RunIO:
    stdout: object
    stderr: object
    decide: object  # callable(decision) -> str: the founder's answer to a typed decision


_bound = ContextVar('spec_generator_run_io', default=None)


@contextmanager
def bind(io):
    if not isinstance(io, RunIO) or not callable(io.decide):
        raise TypeError('RunIO with an answer handler required')
    token = _bound.set(io)
    try:
        yield
    finally:
        _bound.reset(token)


def bound():
    return _bound.get() is not None


def stdout():
    io = _bound.get()
    return io.stdout if io is not None else sys.stdout


def stderr():
    io = _bound.get()
    return io.stderr if io is not None else sys.stderr


def emit(decision):
    """Publish a typed decision on stdout for a supervising process; a bound run asks through its handler instead."""
    if _bound.get() is None:
        from vmodel_contracts.decisions import emit_decision
        emit_decision(decision)


def read(decision, prompt):
    """The founder's answer text: from the bound handler, or from standard input on the command line."""
    io = _bound.get()
    if io is not None:
        return io.decide(decision)
    return input(prompt)


def set_stderr(stream):
    """Tee or replace the error stream for the rest of this run (the command line replaces the process stream)."""
    io = _bound.get()
    if io is None:
        sys.stderr = stream
    else:
        _bound.set(replace(io, stderr=stream))
