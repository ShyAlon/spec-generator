from conftest import SYSTEM_BASELINE, SchemaRoutingArchitectureModel, schema_routing_factory
"""ADR-0020 skeletons: code owns ids, kinds, ordering, and trace links; the
model's prose fills are merged onto them, and unusable structure is a named,
retryable error (ModelStructureError), never a silent remap."""
import json

import pytest

from spec_generator import model_config
from spec_generator.graph import build_graph
from spec_generator.telemetry import UsageLog

from spec_generator.skeleton import (ModelStructureError, assign_architecture_ids,
                                     build_srs_skeleton, build_test_skeleton,
                                     merge_srs_fills, merge_test_fills)
from test_progress import CONTRACT_SRS, INTENT, PAYLOADS, FakeModel


def test_srs_skeleton_one_slot_per_need_in_order():
    needs = {"needs": [
        {"need_id": "N-1", "kind": "functional", "intent_fields": ["problem_statement"]},
        {"need_id": "N-2", "kind": "non_functional", "intent_fields": []}]}
    slots = build_srs_skeleton(needs)
    assert [s["requirement_id"] for s in slots] == ["R-1", "R-2"]
    assert slots[0]["kind"] == "functional" and slots[0]["need_ids"] == ["N-1"]
    assert slots[1]["kind"] == "non_functional" and slots[1]["need_ids"] == ["N-2"]


def test_merge_assigns_ac_ids_and_sections_by_skeleton_kind():
    needs = {"needs": [
        {"need_id": "N-1", "kind": "non_functional", "intent_fields": []},
        {"need_id": "N-2", "kind": "functional", "intent_fields": ["f"]}]}
    body = {"purpose": "p", "scope_summary": "s", "fills": [
        {"requirement_id": "R-2", "title": "t2", "statement": "s2", "priority": "must",
         "acceptance": ["a2", "a3"]},
        {"requirement_id": "R-1", "title": "t1", "statement": "s1", "priority": "could",
         "acceptance": ["a1"]}]}
    merged = merge_srs_fills(build_srs_skeleton(needs), body)
    # sections follow the SKELETON kind, ids follow slot order - fill order is irrelevant
    assert [r["requirement_id"] for r in merged["functional"]] == ["R-2"]
    assert [r["requirement_id"] for r in merged["non_functional"]] == ["R-1"]
    assert [a["criterion_id"] for a in merged["non_functional"][0]["acceptance"]] == ["AC-1"]
    assert [a["criterion_id"] for a in merged["functional"][0]["acceptance"]] == ["AC-2", "AC-3"]
    assert merged["functional"][0]["kind"] == "functional"  # model prose cannot change kind


def test_merge_rejects_missing_unknown_and_duplicate_fills():
    slots = build_srs_skeleton(PAYLOADS["needs"])
    with pytest.raises(ModelStructureError, match="not filled"):
        merge_srs_fills(slots, {"purpose": "p", "scope_summary": "s", "fills": []})
    with pytest.raises(ModelStructureError, match="unknown"):
        merge_srs_fills(slots, {"fills": [{"requirement_id": "R-99", "title": "t", "statement": "s",
                                           "priority": "must", "acceptance": ["a"]},
                                          {"requirement_id": "R-1", "title": "t", "statement": "s",
                                           "priority": "must", "acceptance": ["a"]}]})
    with pytest.raises(ModelStructureError, match="twice"):
        merge_srs_fills(slots, {"fills": [{"requirement_id": "R-1", "title": "t", "statement": "s",
                                           "priority": "must", "acceptance": ["a"]}] * 2})


def test_test_skeleton_links_and_continuous_ids():
    assigned = [dict(PAYLOADS["items"][0], spec_id="SPEC-1")]  # post-numbering shape
    slots = build_test_skeleton(CONTRACT_SRS, assigned)
    assert slots == [
        {"test_id": "T-1", "level": "acceptance", "criterion_id": "AC-1", "spec_id": None},
        {"test_id": "T-2", "level": "unit", "criterion_id": None, "spec_id": "SPEC-1"}]
    merged = merge_test_fills(slots, {"fills": [
        {"test_id": "T-2", "description": "d2", "components": ["api"], "runner": "pytest", "definition": {"preconditions": [], "steps": ["Run x"], "expected_outcomes": ["passes"], "cleanup": [], "cases": [], "execution": {}}},
        {"test_id": "T-1", "description": "d1", "components": ["api"], "runner": "http", "definition": {"preconditions": [], "steps": ["GET /"], "expected_outcomes": ["HTTP 200"], "cleanup": [], "cases": [], "execution": {}}}]})
    assert [(t["test_id"], t["description"]) for t in merged] == [("T-1", "d1"), ("T-2", "d2")]


def test_architecture_ids_assigned_and_duplicates_rejected():
    out = assign_architecture_ids({"components": [{"name": "a", "responsibility": "r"}],
                                   "decisions": [{"title": "x", "decision": "d", "rationale": "r",
                                                  "decision_id": "ADR-99"},  # model ids ignored
                                                 {"title": "y", "decision": "d2", "rationale": "r2"}]})
    assert [d["decision_id"] for d in out["decisions"]] == ["ADR-1", "ADR-2"]
    with pytest.raises(ModelStructureError, match="duplicate"):
        assign_architecture_ids({"components": [{"name": "a", "responsibility": "r"},
                                                {"name": "a", "responsibility": "r2"}], "decisions": []})


def _run_until_stage_error(monkeypatch, bad_stage, bad_payload):
    """Full graph with one stage's model emitting a contract-broken draft on
    EVERY attempt: the failure must surface at THAT stage (retryable contract
    feedback, then a friendly stage error) - never downstream at a gate or in
    assembly. The stage validator owns the MERGED body (ADR-0020)."""
    import pytest
    from langchain_core.messages import AIMessage

    from spec_generator.errors import UserFacingError

    class FM(FakeModel):
        def invoke(self, messages):
            self.calls += 1
            if self.stage == bad_stage:
                return AIMessage(content=json.dumps(bad_payload))
            return super().invoke(messages)

    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: schema_routing_factory(stage, kw.get("schema"), lambda: FM(stage)))
    g = build_graph(UsageLog("s"))
    with pytest.raises(UserFacingError) as ei:
        g.invoke({"intent_model": INTENT, "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "merged-" + bad_stage}})
    return ei.value


def test_architecture_merged_body_validated_at_stage(monkeypatch):
    """A contract-broken ADR body fails at the architecture micro-call."""
    from langchain_core.messages import AIMessage
    from spec_generator.errors import UserFacingError
    class BrokenAdr(SchemaRoutingArchitectureModel):
        def stream(self, messages):
            if "driver" in self.schema.get("properties", {}):
                yield AIMessage(content=json.dumps({"title": "t", "decision": "x", "rationale": "",
                    "driver": "simplicity", "options_considered": [], "baseline_refs": [],
                    "evidence": [], "affected_repositories": [], "executes_concept_ids": []}))
                return
            yield from super().stream(messages)
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw:
        BrokenAdr(stage, kw.get("schema") or {}) if stage == "architecture" else FakeModel(stage))
    with pytest.raises(UserFacingError) as ei:
        build_graph(UsageLog("s")).invoke({"intent_model": INTENT, "system_baseline": SYSTEM_BASELINE},
                                           {"configurable": {"thread_id": "broken-adr"}})
    assert "architecture" in ei.value.render()

def test_tests_merged_body_validated_at_stage(monkeypatch):
    """Tests: model fills prose per slot, code merges and validates EVERY
    merged AcceptanceTest at the stage. A definition that is not an object
    breaks the contract and must fail AT the tests stage, not in assembly."""
    bad = {"fills": [{"test_id": "T-1", "description": "ok", "components": ["api"], "runner": "http",
                      "definition": {"preconditions": [], "steps": ["GET /"], "expected_outcomes": ["HTTP 200"], "cleanup": [], "cases": [], "execution": {}}},
                     {"test_id": "T-2", "description": "broken", "components": ["api"], "runner": "pytest",
                      "definition": "not-an-object"}]}
    e = _run_until_stage_error(monkeypatch, "tests", bad)
    assert "tests" in e.render()


def test_duplicate_needs_merge_before_they_create_duplicate_requirements():
    from spec_generator.skeleton import merge_duplicate_needs
    data={"session_id":"s","needs":[
      {"need_id":"N-1","stakeholder":"manager","statement":"Predict shipment damage and delay risks before arrival","kind":"functional","intent_fields":["problem_statement"],"intent_need_ids":["SN-1"]},
      {"need_id":"N-6","stakeholder":"manager","statement":"Predict shipment damage and delay risks before arrival","kind":"functional","intent_fields":["core_value_proposition"],"intent_need_ids":["SN-4"]},
      {"need_id":"N-9","stakeholder":"operator","statement":"Scale horizontally for growing data volume","kind":"non_functional","intent_fields":[],"intent_need_ids":["SN-5"]}]}
    merged=merge_duplicate_needs(data)
    assert [n["need_id"] for n in merged["needs"]]==["N-1","N-2"]
    assert merged["needs"][0]["intent_need_ids"]==["SN-1","SN-4"]
    assert merged["needs"][0]["intent_fields"]==["problem_statement","core_value_proposition"]


def test_borderline_or_cross_kind_needs_are_not_silently_merged():
    from spec_generator.skeleton import merge_duplicate_needs
    data={"needs":[
      {"need_id":"N-1","statement":"Scale horizontally for growing daily data volume","kind":"non_functional","intent_fields":[],"intent_need_ids":[]},
      {"need_id":"N-2","statement":"Process a few dozen MB of data each day","kind":"non_functional","intent_fields":[],"intent_need_ids":[]},
      {"need_id":"N-3","statement":"Scale horizontally for growing daily data volume","kind":"functional","intent_fields":[],"intent_need_ids":[]}]}
    assert len(merge_duplicate_needs(data)["needs"])==3


def test_approved_intent_needs_are_carried_one_to_one_without_implied_duplicates():
    from spec_generator.skeleton import needs_from_intent
    intent={"session_id":"s","stakeholder_needs":[
      {"need_id":"SN-1","statement":"Identify damaged or delayed shipments beforehand","kind":"functional"},
      {"need_id":"SN-5","statement":"Handle dozens of MB daily and scale horizontally","kind":"non_functional"}]}
    out=needs_from_intent(intent)
    assert [(n["need_id"],n["intent_need_ids"],n["statement"]) for n in out["needs"]]==[
      ("N-1",["SN-1"],intent["stakeholder_needs"][0]["statement"]),
      ("N-2",["SN-5"],intent["stakeholder_needs"][1]["statement"])]
