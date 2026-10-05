from conftest import SYSTEM_BASELINE
"""ADR-0022: the model owns prose; code owns everything enumerable.

The determinism harness has two levers:
- temperature 0 + a fixed seed for reproducible local/openai calls;
- per-slot micro-calls for SRS fills and item bodies (bulk remains available
  via SPEC_GENERATOR_BULK=1 if wall time is worse on the founder's hardware).
"""
import json

from langchain_core.messages import AIMessage

from spec_generator import model_config
from spec_generator.graph import (GenState, _items_body_payload, _items_plan_payload,
                                  _srs_overview_payload, _srs_slot_payload, build_graph)
from spec_generator.schemas import (items_body_schema, items_plan_schema,
                                    srs_overview_schema, srs_slot_fill_schema)
from spec_generator.telemetry import UsageLog
from test_progress import CONTRACT_SRS, INTENT, PAYLOADS, FakeModel


def test_all_stage_defaults_are_deterministic(monkeypatch):
    monkeypatch.delenv("SPEC_GENERATOR_SEED", raising=False)
    for stage in ("needs", "srs", "architecture", "items", "tests", "review"):
        cfg = model_config.for_stage(stage)
        assert cfg.temperature == 0.0
        assert cfg.seed == 42
    monkeypatch.setenv("SPEC_GENERATOR_SEED", "99")
    assert model_config.for_stage("srs").seed == 99


def test_ollama_build_receives_temperature_zero_and_seed(monkeypatch):
    monkeypatch.setattr(model_config, "_ollama_structured_support", lambda base_url=None: (True, "test server"))
    seen = {}
    class FakeOllama:
        def __init__(self, **kw): seen.update(kw)
    import langchain_ollama
    monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeOllama)
    exact_schema = {"type": "object"}
    model_config.build_chat_model("srs", schema=exact_schema)
    assert seen["temperature"] == 0.0 and seen["seed"] == 42
    assert seen["format"] is exact_schema


def test_bulk_is_escape_hatch_not_default(monkeypatch):
    monkeypatch.delenv("SPEC_GENERATOR_BULK", raising=False)
    assert model_config.bulk_fills_enabled() is False
    monkeypatch.setenv("SPEC_GENERATOR_BULK", "1")
    assert model_config.bulk_fills_enabled() is True


def test_micro_call_schemas_have_no_model_owned_ids():
    assert set(srs_overview_schema()["properties"]) == {"purpose", "scope_summary"}
    assert set(srs_slot_fill_schema()["properties"]) == {"title", "statement", "priority", "acceptance"}
    assert set(items_body_schema()["properties"]) == {"body_markdown"}
    plans = items_plan_schema(["R-1", "R-2"])
    assert plans["items"]["properties"]["requirement_ids"]["items"] == {"enum": ["R-1", "R-2"]}
    assert "body_markdown" not in plans["items"]["properties"]


def test_micro_call_prompts_are_small_and_id_clean():
    state = {"intent_model": dict(INTENT, session_id="20260909-094216-e900"),
             "needs": PAYLOADS["needs"], "srs": CONTRACT_SRS,
             "architecture": PAYLOADS["architecture"]}
    slot = {"requirement_id": "R-1", "kind": "functional", "need_ids": ["N-1"],
            "intent_fields": ["problem_statement"]}
    for payload in (_srs_overview_payload(state), _srs_slot_payload(state, slot),
                    _items_plan_payload(state, ["R-1"]),
                    _items_body_payload(state, {"spec_id": "SPEC-1", "slug": "api",
                                                "kind": "api_surface",
                                                "title": "API", "requirement_ids": ["R-1"],
                                                "depends_on": []})):
        assert "20260909-094216-e900" not in payload
        assert payload.rstrip().index("FORMAT CONTRACT") > 0
    assert "PREVIOUS FILL" in _srs_slot_payload(state, slot)  # revision is per-slot
    assert '"R-1"' in _items_plan_payload(state, ["R-1"])


def test_full_pipeline_uses_per_slot_micro_calls(monkeypatch):
    """One requirement => overview + one slot; items => plan + one body;
    tests => one call per slot (a single 36-item call outgrows the call
    budget - 2026-09-10 live kill at 600s after the last item closed)."""
    monkeypatch.delenv("SPEC_GENERATOR_BULK", raising=False)
    calls = []

    class RoutingModel:
        def __init__(self, stage, schema): self.stage, self.schema = stage, schema
        def stream(self, messages):
            calls.append((self.stage, self.schema, messages[-1].content))
            if self.stage == "srs":
                props = set(self.schema.get("properties", {}))
                if props == {"purpose", "scope_summary"}:
                    out = {"purpose": "p", "scope_summary": "sc"}
                else:
                    out = {"title": "Spec file", "statement": "system shall emit a spec",
                           "priority": "must", "acceptance": ["spec exists"]}
            elif self.stage == "items":
                if self.schema.get("type") == "array":
                    out = [{"slug": "api", "kind": "api_surface", "title": "API",
                            "requirement_ids": ["R-1"], "depends_on": []}]
                else:
                    out = {"body_markdown": "# API"}
            elif self.stage == "tests":
                wanted = self.schema["properties"]["fills"]["required"]
                out = {"fills": {tid: PAYLOADS["tests"]["fills"][tid] for tid in wanted}}
            elif self.stage == "validation":
                out = {"items": []}
            elif self.stage == "architecture":
                from test_per_item_resume import _arch_answer
                out = _arch_answer(self.schema)
            else:
                out = PAYLOADS[self.stage]
            yield AIMessage(content=json.dumps(out))

    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: RoutingModel(stage, kw.get("schema") or {}))
    r = build_graph(UsageLog("s")).invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
                                           {"configurable": {"thread_id": "s"}})
    # toy payloads trip the ungrounded-vocabulary guard by design; what matters
    # here is that no STRUCTURAL violation survives (coverage, unknown ids)
    assert not any("covered by no item spec" in v or "unknown requirements" in v
                   for v in r["violations"])
    assert [s for s, _, _ in calls].count("srs") == 2
    assert [s for s, _, _ in calls].count("items") == 2
    assert [s for s, _, _ in calls].count("tests") == 2  # one call per slot
    assert [s for s, _, _ in calls].count("validation") == 1  # one call per need
    assert r["srs"]["functional"][0]["requirement_id"] == "R-1"  # code-owned
    assert r["items"][0]["spec_id"] == "SPEC-1" and r["items"][0]["slug"] == "api"  # code-owned numbering
    arch_calls = [s for s, _, _ in calls].count("architecture")
    assert arch_calls == 7  # 3 plans + 1 stack entry + 1 component body + 1 ADR body
    assert r["architecture"]["stack_entries"][0]["key"] == "language"
    assert r["architecture"]["decisions"][0]["decision_id"] == "ADR-1"  # code-owned


def test_compact_test_context_excludes_unlinked_artifact_bodies():
    from spec_generator.graph import _compact_test_context
    state = {
        "srs": {"functional": [{"requirement_id": "R-1", "acceptance": [
            {"criterion_id": "AC-1", "statement": "linked"},
            {"criterion_id": "AC-2", "statement": "unlinked secret"},
        ]}], "non_functional": []},
        "items": [{"spec_id": "SPEC-1", "body_markdown": "large body one"},
                  {"spec_id": "SPEC-2", "body_markdown": "large body two"}],
        "architecture": {"stack": ["Python"], "components": [{"name": "api"}]},
    }
    compact = _compact_test_context(state, {"criterion_id": "AC-1"})
    encoded = json.dumps(compact)
    assert "linked" in encoded and "unlinked secret" not in encoded
    assert "large body" not in encoded
    assert compact["id_glossary"]["specs"] == ["SPEC-1", "SPEC-2"]


def test_srs_slot_puts_shared_intent_before_varying_slot():
    payload = _srs_slot_payload({
        "intent_model": {"problem_statement": "p", "core_value_proposition": "v",
                         "in_scope": [], "out_of_scope": [], "constraints": []},
        "needs": {"needs": [{"need_id": "N-1", "statement": "need"}]},
        "srs": {},
    }, {"requirement_id": "R-1", "need_ids": ["N-1"]})
    assert payload.index("INTENT (shared context)") < payload.index("THE SLOT")
