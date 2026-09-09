"""ADR-0020 skeletons: code owns ids, kinds, ordering, and trace links; the
model's prose fills are merged onto them, and unusable structure is a named,
retryable error (ModelStructureError), never a silent remap."""
import pytest

from spec_generator.skeleton import (ModelStructureError, assign_architecture_ids,
                                     build_srs_skeleton, build_test_skeleton,
                                     merge_srs_fills, merge_test_fills)
from test_progress import CONTRACT_SRS, PAYLOADS


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
    slots = build_test_skeleton(CONTRACT_SRS, PAYLOADS["items"])
    assert slots == [
        {"test_id": "T-1", "level": "acceptance", "criterion_id": "AC-1", "spec_id": None},
        {"test_id": "T-2", "level": "unit", "criterion_id": None, "spec_id": "SPEC-api"}]
    merged = merge_test_fills(slots, {"fills": [
        {"test_id": "T-2", "description": "d2", "runner": "pytest", "definition": {"path": "x"}},
        {"test_id": "T-1", "description": "d1", "runner": "http", "definition": {"url": "/"}}]})
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
