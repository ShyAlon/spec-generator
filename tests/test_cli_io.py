"""Hermetic checks that the generator command takes its answers and streams from a bound caller. No provider."""
import ast
import builtins
import io
import pathlib
import sys
import threading
import types

import pytest

from spec_generator import cli, cli_io

SRC = pathlib.Path(cli.__file__).parent


def run_io(answers):
    feed = iter(answers)
    return cli_io.RunIO(stdout=io.StringIO(), stderr=io.StringIO(), decide=lambda decision: next(feed))


def test_unbound_streams_are_the_process_streams():
    assert cli_io.stdout() is sys.stdout and cli_io.stderr() is sys.stderr and not cli_io.bound()


def test_bound_streams_are_used_then_restored():
    run = run_io([])
    with cli_io.bind(run):
        assert cli_io.stdout() is run.stdout and cli_io.stderr() is run.stderr and cli_io.bound()
    assert cli_io.stdout() is sys.stdout and not cli_io.bound()


def test_binding_is_not_visible_to_another_thread():
    seen = []
    with cli_io.bind(run_io([])):
        thread = threading.Thread(target=lambda: seen.append(cli_io.bound()))
        thread.start()
        thread.join()
    assert seen == [False]


def test_bind_requires_an_answer_handler():
    with pytest.raises(TypeError):
        with cli_io.bind(cli_io.RunIO(io.StringIO(), io.StringIO(), None)):
            pass


def test_set_stderr_changes_only_the_bound_run():
    run = run_io([])
    other = io.StringIO()
    with cli_io.bind(run):
        cli_io.set_stderr(other)
        assert cli_io.stderr() is other
    assert cli_io.stderr() is sys.stderr
    assert run.stderr is not other


def test_unbound_read_uses_standard_input(monkeypatch):
    monkeypatch.setattr(builtins, 'input', lambda prompt: 'a')
    assert cli_io.read(object(), 'prompt') == 'a'


def test_bound_read_uses_the_handler_and_emit_is_silent(capsys):
    with cli_io.bind(run_io(['f'])):
        cli_io.emit(object())
        assert cli_io.read(object(), 'prompt') == 'f'
    assert capsys.readouterr() == ('', '')


def test_no_direct_process_stream_left_in_the_command():
    for name in ('cli.py', 'progress.py', 'structured_progress.py'):
        text = (SRC / name).read_text()
        assert 'sys.stderr' not in text and 'sys.stdout' not in text, name
    for node in ast.walk(ast.parse((SRC / 'cli.py').read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print':
            assert any(k.arg == 'file' for k in node.keywords), 'print on line %d names no stream' % node.lineno


def test_main_takes_argv_and_ignores_the_process_arguments(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['generator', '--no-such-option'])
    with pytest.raises(SystemExit) as stop:
        cli.main(['--help'])
    assert stop.value.code == 0


class FakeGraph:
    def __init__(self):
        self.resumes = []
        self.waiting = False

    def invoke(self, value, config):
        from langgraph.types import Command
        if isinstance(value, Command):
            self.resumes.append(value.resume)
            self.waiting = value.resume['action'] != 'accept'
        else:
            self.waiting = True
        return {}

    def get_state(self, config):
        task = types.SimpleNamespace(interrupts=[types.SimpleNamespace(value={'gate': 'srs', 'artifact': {}, 'advisories': []})])
        return types.SimpleNamespace(next=('gate',) if self.waiting else (), tasks=[task] if self.waiting else [])


def gate(monkeypatch, tmp_path, answers):
    monkeypatch.setattr(cli, '_render_gate', lambda *args: 'rendered gate')
    graph, run = FakeGraph(), run_io(answers)
    with cli_io.bind(run):
        cli._run_with_gates(graph, {}, {'intent_model': {}}, tmp_path)
    return graph, run


def test_gate_answers_come_from_the_bound_handler(monkeypatch, tmp_path, capsys):
    graph, run = gate(monkeypatch, tmp_path, ['x', 'f', 'feedback - shorter', 'a'])
    assert graph.resumes == [{'action': 'regenerate', 'feedback': 'shorter'}, {'action': 'accept'}]
    assert 'REVIEW GATE' in run.stderr.getvalue()
    assert capsys.readouterr() == ('', '')


def test_reject_at_a_bound_gate_stops(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        gate(monkeypatch, tmp_path, ['r'])


def test_edit_is_not_offered_at_a_bound_gate(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli, '_render_gate', lambda *args: 'rendered gate')
    answers = iter(['e', 'a'])

    def decide(decision):
        seen.append(decision)
        return next(answers)

    graph = FakeGraph()
    with cli_io.bind(cli_io.RunIO(io.StringIO(), io.StringIO(), decide)):
        cli._run_with_gates(graph, {}, {'intent_model': {}}, tmp_path)
    assert graph.resumes == [{'action': 'accept'}]
    assert len(seen) == 2 and 'Edit' not in repr(seen[0])
