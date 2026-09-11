from conftest import SYSTEM_BASELINE
"""2026-09-09 third live failure (issues 1, 2, 4):
1. Items stage invented requirement ids (R-001..R-005) and left all ten real
   requirements uncovered - references are now an ENUM of the real ids in the
   grammar, and coverage is validated AT THE STAGE with named retry feedback.
2. The session id leaked into spec names - pipeline-owned bookkeeping is
   stripped from everything shown to the model (_prompt_view).
4. LangGraph msgpack warnings - contract enums registered via
   allowed_msgpack_modules; new state is dumped mode='json' (plain strings)."""
import json
import logging
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from spec_generator import model_config
from spec_generator.graph import (_items_payload, _items_validator, _prompt_view,
                                  build_graph, checkpoint_serde)
from spec_generator.schemas import items_schema
from spec_generator.skeleton import ModelStructureError
from spec_generator.telemetry import UsageLog
from test_progress import INTENT, PAYLOADS, FakeModel


def test_items_grammar_enums_real_requirement_ids():
    sch = items_schema(["R-1", "R-7", "R-10"])
    refs = sch["items"]["properties"]["requirement_ids"]["items"]
    assert refs == {"enum": ["R-1", "R-7", "R-10"]}
    assert sch["items"]["properties"]["kind"]["enum"]


def test_items_grammar_reaches_the_model_build(monkeypatch, tmp_path):
    seen = {}

    def build(stage, structured=True, schema=None):
        seen[stage] = schema
        return FakeModel(stage)

    monkeypatch.setattr(model_config, "build_chat_model", build)
    g = build_graph(UsageLog("s"))
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "s"}})
    refs = seen["items"]["items"]["properties"]["requirement_ids"]["items"]
    assert refs == {"enum": ["R-1"]}


def test_unknown_requirement_id_retried_with_named_feedback(monkeypatch, tmp_path):
    """His exact failure: invented R-001. First attempt bad, retry (feedback
    names the id) recovers - and the feedback the model sees says WHY."""
    seen_messages = []

    class InventingItems(FakeModel):
        def invoke(self, messages):
            self.calls += 1
            seen_messages.append(messages)
            from langchain_core.messages import AIMessage
            if self.calls == 1:
                bad = [dict(PAYLOADS["items"][0], requirement_ids=["R-001"])]
                return AIMessage(content=json.dumps(bad))
            return AIMessage(content=json.dumps(PAYLOADS[self.stage]))

    box = {}
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: box.setdefault(stage, InventingItems(stage) if stage == "items" else FakeModel(stage)))
    g = build_graph(UsageLog("s"))
    r = g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "s"}})
    assert "bundle" in r
    assert box["items"].calls == 2
    retry_feedback = seen_messages[1][-1].content
    assert "R-001" in retry_feedback and "claims unknown requirements" in retry_feedback
    assert "R-1" in retry_feedback  # the valid ids are named


def test_uncovered_requirement_named_in_retry_feedback(monkeypatch):
    class LazyItems(FakeModel):
        def invoke(self, messages):
            self.calls += 1
            from langchain_core.messages import AIMessage
            if self.calls == 1:
                return AIMessage(content=json.dumps([dict(PAYLOADS["items"][0], requirement_ids=[])]))
            return AIMessage(content=json.dumps(PAYLOADS[self.stage]))

    box = {}
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: box.setdefault(stage, LazyItems(stage) if stage == "items" else FakeModel(stage)))
    g = build_graph(UsageLog("s"))
    r = g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "s"}})
    assert box["items"].calls == 2
    # the retry recovered: no coverage surprise survives into the punch list
    assert not any("covered by no item spec" in v for v in r["violations"])


def test_items_validator_units():
    v = _items_validator(["R-1", "R-2"])
    good = [{"slug": "a", "kind": "api_surface", "title": "A", "body_markdown": "b",
             "requirement_ids": ["R-1"]},
            {"slug": "b", "kind": "data_model", "title": "B", "body_markdown": "b",
             "requirement_ids": ["R-2"], "depends_on": ["a"]}]
    out = v(good)
    # code-owned numbering: slugs become SPEC-1..N in plan order, deps translate
    assert [i["spec_id"] for i in out] == ["SPEC-1", "SPEC-2"]
    assert out[1]["depends_on"] == ["SPEC-1"] and out[0]["slug"] == "a"
    with pytest.raises(ModelStructureError, match="covered by no item spec: \\['R-2'\\]"):
        v(good[:1])
    with pytest.raises(ModelStructureError, match="duplicate spec ids"):
        v([good[0], dict(good[0], requirement_ids=["R-2"])])
    with pytest.raises(ModelStructureError, match="depends on unknown specs"):
        v([good[0], dict(good[1], depends_on=["nope"])])
    with pytest.raises(ModelStructureError, match="claims unknown requirements \\['R-99'\\]"):
        v([dict(good[0], requirement_ids=["R-1", "R-99"]), good[1]])
    with pytest.raises(ModelStructureError, match="invalid slug"):
        v([dict(good[0], slug="!!!")])  # unhealable (normalizes to empty) - real drift still fails


def test_items_validator_heals_slug_punctuation():
    """ADR-0022: punctuation drift in a model slug is healed, not retried;
    depends_on references remap with the same table."""
    v = _items_validator(["R-1", "R-2"])
    plans = [{"slug": "user_capabilities", "kind": "api_surface", "title": "Caps", "body_markdown": "b",
              "requirement_ids": ["R-1"]},
             {"slug": "Order API", "kind": "data_model", "title": "B", "body_markdown": "b",
              "requirement_ids": ["R-2"], "depends_on": ["user_capabilities"]}]
    out = v(plans)
    assert [o["slug"] for o in out] == ["user-capabilities", "order-api"]
    assert out[1]["depends_on"] == ["SPEC-1"]  # healed refs still translate to SPEC-ids


def test_items_validator_heal_collisions_get_deterministic_suffixes():
    v = _items_validator(["R-1", "R-2"])
    plans = [{"slug": "Order API", "kind": "api_surface", "title": "A", "body_markdown": "b",
              "requirement_ids": ["R-1"]},
             {"slug": "order-api", "kind": "data_model", "title": "B", "body_markdown": "b",
              "requirement_ids": ["R-2"]}]
    out = v(plans)
    assert [o["slug"] for o in out] == ["order-api", "order-api-2"]


def test_items_plan_validator_heals_slugs():
    from spec_generator.graph import _items_plan_validator

    v = _items_plan_validator(["R-1"])
    out = v([{"slug": "User_Capabilities", "kind": "api_surface", "title": "Caps",
              "requirement_ids": ["R-1"], "depends_on": []}])
    assert out[0]["slug"] == "user-capabilities" and out[0]["spec_id"] == "SPEC-1"


def test_prompt_view_strips_pipeline_owned_recursively():
    state = {"srs": {"session_id": "20260909-094216-e900", "purpose": "p",
                     "functional": [{"requirement_id": "R-1", "title": "t", "contract_version": "0.8.0",
                                     "acceptance": [{"criterion_id": "AC-1", "statement": "s"}]}],
                     "non_functional": []},
             "architecture": {"session_id": "20260909-094216-e900", "stack": {"language": "ts"}}}
    payload = _items_payload(state, ["R-1"])
    assert "20260909-094216-e900" not in payload
    assert "session_id" not in payload and "contract_version" not in payload
    assert '"R-1"' in payload and '"criterion_id": "AC-1"' in payload  # content survives


def test_items_payload_closes_with_the_contract():
    state = {"srs": {"functional": [], "non_functional": []}, "architecture": {}}
    payload = _items_payload(state, ["R-1", "R-2"])
    assert payload.rstrip().index("FORMAT CONTRACT") > payload.index("ARCHITECTURE")
    closing = payload[payload.index("FORMAT CONTRACT"):]
    assert '["R-1", "R-2"]' in closing
    assert "NEVER contain session ids" in closing
    assert "FOUNDER'S PRODUCT" in closing


def test_checkpoint_state_is_plain_json_after_assemble(monkeypatch):
    """No enum instances in state: json.dumps needs no custom encoder, and
    types are exactly str (mode='json' dumps)."""
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: FakeModel(stage))
    g = build_graph(UsageLog("s"))
    r = g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "s"}})
    assert type(r["bundle"]["items"][0]["kind"]) is str
    assert type(r["bundle"]["_srs"]["functional"][0]["priority"]) is str
    json.dumps(r["bundle"])  # would raise on any non-JSON residue


def test_checkpoint_serde_roundtrips_legacy_enum_state_silently(caplog):
    """Pre-fix checkpoints hold enum instances (python-mode model_dump). The
    registered serde reads them without the 'unregistered type' warning that
    spammed his console, and future strict mode stays compatible."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from vmodel_contracts import ItemSpec

    legacy_blob = JsonPlusSerializer().dumps_typed(
        ItemSpec(spec_id="SPEC-1", kind="api_surface", title="t",
                 body_markdown="b", requirement_ids=["R-1"]).model_dump())
    serde = checkpoint_serde()
    with caplog.at_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus"):
        back = serde.loads_typed(legacy_blob)
        serde.loads_typed(serde.dumps_typed(back))
    assert back["kind"] == "api_surface"
    assert "unregistered type" not in caplog.text
