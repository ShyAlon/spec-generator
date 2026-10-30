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
    """Each test is one isolated run, independent of the developer's shared .env."""
    monkeypatch.setattr(mc, "_CALLS_USED", 0)
    monkeypatch.setattr(mc, "_DOTENV", {})


SYSTEM_BASELINE = {
    "contract_version": "0.16.0", "baseline_id": "test-baseline", "system_name": "Test system", "mode": "existing",
    "repositories": [{"contract_version": "0.16.0", "name": "spec-generator", "languages": ["Python", "ts"],
        "module_entry_points": ["python -m spec_generator.cli"], "existing_artifacts": [],
        "existing_contracts": [], "runtime": ["Python"], "persistence": ["PostgreSQL 14"],
        "integrations": [], "immutable_decisions": []}], "immutable_decisions": []}

class ModelFactoryLedger:
    """Fresh schema-bound models with aggregate per-stage call accounting."""
    def __init__(self, model_cls):
        self.model_cls = model_cls
        self.models = []

    def build(self, stage, schema=None):
        try:
            model = self.model_cls(stage, schema or {})
        except TypeError:
            model = self.model_cls(stage)
        model.schema = schema or {}
        self.models.append(model)
        return model

    def calls(self, stage):
        return sum(getattr(m, "calls", 0) for m in self.models if m.stage == stage)


class StageCounter:
    def __init__(self, ledger, stage): self.ledger, self.stage = ledger, stage
    @property
    def calls(self): return self.ledger.calls(self.stage)
    @property
    def last_human(self):
        models=[m for m in self.ledger.models if m.stage == self.stage and hasattr(m, "last_human")]
        return models[-1].last_human if models else None


class SchemaRoutingArchitectureModel:
    """Current architecture micro-contract fixture, one schema per instance."""
    def __init__(self, stage, schema=None):
        self.stage, self.schema, self.calls = stage, schema or {}, 0
    def stream(self, messages):
        import json
        from langchain_core.messages import AIMessage
        from test_per_item_resume import _arch_answer
        self.calls += 1
        yield AIMessage(content=json.dumps(_arch_answer(self.schema)))


def schema_routing_factory(stage, schema, fallback):
    """Route architecture's current micro-contract while preserving a test's stage fake."""
    return SchemaRoutingArchitectureModel(stage, schema or {}) if stage == "architecture" else fallback()


@pytest.fixture
def supported_host_runtime(tmp_path):
    from host_checkpoint_fixture import hosted_runtime
    from spec_generator.worker_runtime import job_runtime
    runtime=hosted_runtime(tmp_path)
    with job_runtime(runtime):yield runtime
