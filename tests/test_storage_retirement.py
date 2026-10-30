"""Retired file source reader cannot return through an import or dependency."""
import importlib.util
from pathlib import Path


def test_file_source_reader_and_runtime_dependency_are_absent():
    assert importlib.util.find_spec('spec_generator.revision_source') is None
    project=(Path(__file__).parents[1]/'pyproject.toml').read_text()
    assert 'langgraph-checkpoint-sqlite' not in project
