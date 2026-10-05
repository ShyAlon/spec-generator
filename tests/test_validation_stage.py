"""Validation stage: the human-judged half of the V-model, apart from tests.json."""
import pytest
from conftest import SYSTEM_BASELINE

from spec_generator.findings_ledger import Ledger
from spec_generator.graph import build_graph
from spec_generator.skeleton import (ModelStructureError, check_validation_items,
                                     merge_validation_fills, uncovered_needs, validation_slots)
from spec_generator.telemetry import UsageLog
from test_progress import INTENT, PAYLOADS, fake_models  # noqa: F401

NEEDS = {"needs": [{"need_id": "N-1", "statement": "kids enjoy it"}, {"need_id": "N-2", "statement": "it is fast"}]}
SRS = {"functional": [{"requirement_id": "R-1", "need_ids": ["N-1"],
                       "acceptance": [{"criterion_id": "AC-1", "statement": "score shows"}]}],
       "non_functional": []}
TESTS = [{"test_id": "T-1", "criterion_id": "AC-1"}, {"test_id": "T-2", "spec_id": "SPEC-1"}]
ARCH = {"components": [{"name": "api"}, {"name": "ui"}]}


def item(**kw):
    base = {"criterion_id": None, "components": ["ui"], "what_the_human_does": "Play a round with a child",
            "what_to_observe": "Does the child smile", "pass_when": "The child asks to play again",
            "evidence": "Notes in the review log", "environment": "A real tablet"}
    return {**base, **kw}


def test_slots_show_machine_coverage_per_need():
    slots = validation_slots(NEEDS, SRS, TESTS)
    assert [s["need_id"] for s in slots] == ["N-1", "N-2"]
    assert slots[0]["machine_covered"] and slots[0]["criteria"][0]["test_ids"] == ["T-1"]
    assert not slots[1]["machine_covered"] and slots[1]["criteria"] == []


def test_checks_name_the_problem():
    slot = validation_slots(NEEDS, SRS, TESTS)[0]
    check_validation_items(slot, [item(criterion_id="AC-1")], ARCH)  # valid
    for bad, needle in [(item(criterion_id="AC-9"), "does not belong to this need"),
                        (item(components=["ghost"]), "not in the architecture"),
                        (item(pass_when=" "), "pass_when is empty"),
                        (item(what_the_human_does="Tap [data-testid=go]"), "no selectors")]:
        with pytest.raises(ModelStructureError, match=needle):
            check_validation_items(slot, [bad], ARCH)


def test_merge_owns_ids_and_derives_verified_by():
    slots = validation_slots(NEEDS, SRS, TESTS)
    plan = merge_validation_fills(slots, {"N-1": {"items": [item(criterion_id="AC-1"), item()]},
                                          "N-2": {"items": [item()]}})
    ids = [(v["validation_id"], v["need_ids"], v["verified_by"]) for v in plan["items"]]
    assert ids == [("V-1", ["N-1"], ["T-1"]), ("V-2", ["N-1"], []), ("V-3", ["N-2"], [])]


def test_uncovered_need_is_one_without_tests_and_without_items():
    slots = validation_slots(NEEDS, SRS, TESTS)
    assert [s["need_id"] for s in uncovered_needs(slots, {"items": []})] == ["N-2"]
    plan = merge_validation_fills(slots, {"N-2": {"items": [item()]}})
    assert uncovered_needs(slots, plan) == []


def test_rerun_retires_findings_not_raised_again():
    led = Ledger()
    kw = dict(rule_version="1", source="validation", kind="semantic", artifact_revision="r", scope="need")
    keep = led.add(rule_id="validation.uncovered_need", targets=["N-1"], summary="a", **kw)
    gone = led.add(rule_id="validation.uncovered_need", targets=["N-2"], summary="b", **kw)
    other = led.add(rule_id="review.llm", targets=["N-2"], summary="c", **kw)
    assert led.retire_rule("validation.uncovered_need", {keep}, "rerun") == [gone]
    assert led.findings[gone]["state"] == "SUPERSEDED" and led.findings[keep]["state"] == "OPEN"
    assert led.findings[other]["state"] == "OPEN"


def test_pipeline_runs_validation_after_tests_and_bundles_it(fake_models, monkeypatch):
    monkeypatch.setitem(PAYLOADS, "validation", {"items": [item(criterion_id=None, components=["api"])]})
    r = build_graph(UsageLog("s")).invoke({"intent_model": INTENT, "system_baseline": SYSTEM_BASELINE},
                                          {"configurable": {"thread_id": "s"}})
    v = r["validation"]["items"]
    assert [i["validation_id"] for i in v] == ["V-1"] and v[0]["need_ids"] == ["N-1"]
    assert r["bundle"]["_validation"]["items"][0]["validation_id"] == "V-1"
    assert r["bundle"]["_manifest"]["files"]["validation"] == "validation.json"


def test_pipeline_with_no_items_bundles_an_empty_plan(fake_models):
    r = build_graph(UsageLog("s")).invoke({"intent_model": INTENT, "system_baseline": SYSTEM_BASELINE},
                                          {"configurable": {"thread_id": "s"}})
    assert r["validation"] == {"items": []}
    assert r["bundle"]["_validation"]["items"] == []
