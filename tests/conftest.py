import pytest

import spec_generator.model_config as mc


@pytest.fixture(autouse=True)
def _bulk_mode_default(monkeypatch):
    """The pre-ADR-0022 fixtures model the one-bulk-call-per-stage shape.
    Per-slot micro-calls are exercised by test_adr0022_determinism.py, which
    deletes this env var explicitly."""
    monkeypatch.setenv("SPEC_GENERATOR_BULK", "1")

@pytest.fixture(autouse=True)
def _reset_llm_call_budget(monkeypatch):
    """Each test is one run: the run-wide LLM call budget starts clean."""
    monkeypatch.setattr(mc, "_CALLS_USED", 0)
