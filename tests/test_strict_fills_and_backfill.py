from conftest import SYSTEM_BASELINE, ModelFactoryLedger, SchemaRoutingArchitectureModel
"""2026-09-09 second live failure, three threads:
1. 'R-5: slot not filled' slipped through the generic fills grammar - the
   grammars are now per-skeleton with every slot id a required property.
2. The model's purpose echoed the V-model methodology - the format contract
   now CLOSES the prompt and the domain frame is explicit.
3. Stages completed before persistence shipped never hit the bundle dir on
   resume - _backfill_persist writes them from checkpoint state."""
import json
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from spec_generator import model_config
from spec_generator.cli import _backfill_persist, _make_persist
from spec_generator.graph import build_graph, _srs_fill_call
from spec_generator.schemas import srs_fill_schema, test_fill_schema as _test_slots_grammar
from spec_generator.skeleton import build_srs_skeleton, merge_srs_fills
from spec_generator.telemetry import UsageLog
from test_progress import INTENT, PAYLOADS, FakeModel

SLOTS = [{"requirement_id": "R-1"}, {"requirement_id": "R-5"}]


def test_srs_grammar_requires_every_slot_id():
    sch = srs_fill_schema(SLOTS)
    fills = sch["properties"]["fills"]
    assert fills["required"] == ["R-1", "R-5"]
    assert set(fills["properties"]) == {"R-1", "R-5"}
    assert fills["additionalProperties"] is False
    for f in fills["properties"].values():
        assert f["required"] == ["title", "statement", "priority", "acceptance"]


def test_tests_grammar_requires_every_slot_id():
    sch = _test_slots_grammar([{"test_id": "T-1"}, {"test_id": "T-22"}])
    fills = sch["properties"]["fills"]
    assert fills["required"] == ["T-1", "T-22"]
    assert fills["additionalProperties"] is False


def test_strict_schema_reaches_the_model_build(monkeypatch, tmp_path):
    """End-to-end: the per-skeleton grammar is what build_chat_model gets."""
    seen = {}

    def build(stage, structured=True, schema=None):
        seen[stage] = schema
        return SchemaRoutingArchitectureModel(stage, schema or {}) if stage == "architecture" else FakeModel(stage)

    monkeypatch.setattr(model_config, "build_chat_model", build)
    db = str(tmp_path / "cp.sqlite")
    g = build_graph(UsageLog("s"), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "s"}})
    assert seen["srs"]["properties"]["fills"]["required"] == ["R-1"]
    assert seen["tests"]["properties"]["fills"]["required"] == ["T-1", "T-2"]


def test_fill_prompt_closes_with_the_domain_framed_format_contract():
    state = {"needs": PAYLOADS["needs"], "intent_model": dict(INTENT),
             "review_fixes": [{"fix": "x"}], "srs": {"functional": [], "non_functional": []}}
    payload, _, schema = _srs_fill_call(state, "\n\nREVIEW FIXES:\n[]\n\nPREVIOUS SRS:\n{}")
    assert payload.rstrip().index("FORMAT CONTRACT") > payload.index("PREVIOUS SRS")
    closing = payload[payload.index("FORMAT CONTRACT"):]
    assert '["R-1"]' in closing and "FOUNDER'S PRODUCT" in closing
    assert "never the V model, this pipeline, or these instructions" in closing
    assert schema["properties"]["fills"]["required"] == ["R-1"]


def test_merge_tolerates_free_mode_array_fills():
    """Without the grammar (structured off / non-Ollama), the array form still merges."""
    slots = build_srs_skeleton(PAYLOADS["needs"])
    array_body = {"purpose": "p", "scope_summary": "sc",
                  "fills": [{"requirement_id": "R-1", "title": "t", "statement": "s",
                             "priority": "must", "acceptance": ["a"]}]}
    merged = merge_srs_fills(slots, array_body)
    assert merged["functional"][0]["title"] == "t"


def test_backfill_writes_completed_stages_without_rerunning(monkeypatch, tmp_path):
    """His hole: stages completed before persistence shipped, bundle dir empty
    on resume. Backfill from checkpoint state; zero model calls."""
    box = {}
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: SchemaRoutingArchitectureModel(stage, kw.get("schema") or {}) if stage == "architecture" else box.setdefault(stage, FakeModel(stage)))
    db = str(tmp_path / "cp.sqlite")
    cfg = {"configurable": {"thread_id": "s"}}
    g = build_graph(UsageLog("s"), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, cfg)
    calls_at_completion = {k: v.calls for k, v in box.items()}

    outdir = tmp_path / "bundle"
    outdir.mkdir()
    g2 = build_graph(UsageLog("s"), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    prior = g2.get_state(cfg).values
    written = _backfill_persist(_make_persist(dict(INTENT), outdir), prior)
    assert written == ["needs", "srs", "architecture", "items", "tests"]
    for f in ("needs.json", "needs.md", "srs.json", "srs.md",
              "architecture.json", "architecture.md", "items.json", "tests.json"):
        assert (outdir / f).exists(), f
    assert (outdir / "specs" / "SPEC-1.md").exists()
    assert {k: v.calls for k, v in box.items()} == calls_at_completion  # nothing re-ran


def test_backfill_refuses_pre_identity_architecture(tmp_path):
    legacy_arch = {"session_id":"s","stack":{"language":"ts"},"components":[],"decisions":[]}
    import pytest
    with pytest.raises(Exception):
        _backfill_persist(_make_persist(dict(INTENT), tmp_path), {"architecture":legacy_arch})
