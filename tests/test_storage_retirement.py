"""Retired file source reader cannot return through an import or dependency."""
import importlib.util
from pathlib import Path


def test_file_source_reader_and_runtime_dependency_are_absent():
    assert importlib.util.find_spec('spec_generator.revision_source') is None
    project=(Path(__file__).parents[1]/'pyproject.toml').read_text()
    assert 'langgraph-checkpoint-sqlite' not in project


def test_workflow_addressing_has_no_legacy_checkpoint_filename():
    for name in ('worker_graph.py','revision_worker.py','srs_rework.py','checkpoint_runtime.py'):
        assert 'checkpoints.sqlite' not in (Path(__file__).parents[1]/'src/spec_generator'/name).read_text()
