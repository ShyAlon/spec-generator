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


SYSTEM_BASELINE = {
    "contract_version": "0.13.0", "baseline_id": "test-baseline", "system_name": "Test system", "mode": "existing",
    "repositories": [{"contract_version": "0.13.0", "name": "spec-generator", "languages": ["Python", "ts"],
        "module_entry_points": ["python -m spec_generator.cli"], "existing_artifacts": [],
        "existing_contracts": [], "runtime": ["Python"], "persistence": [],
        "integrations": [], "immutable_decisions": []}], "immutable_decisions": []}
