from conftest import SYSTEM_BASELINE
"""SRS redesign: standalone artifact, deterministic carry, guards, reference doctrine."""
import json

import pytest
from langchain_core.messages import AIMessage

from spec_generator import model_config
from spec_generator.graph import build_graph
from spec_generator.render import render_srs
from spec_generator.telemetry import UsageLog
from test_progress import INTENT, PAYLOADS, FakeModel


def _fills(srs_body):
    """ADR-0020: the model returns prose fills keyed by the skeleton's slot
    ids; derive the fills shape from a contract-shaped fixture body."""
    return {"purpose": srs_body["purpose"], "scope_summary": srs_body["scope_summary"],
            "fills": [{"requirement_id": r["requirement_id"], "title": r["title"],
                       "statement": r["statement"], "priority": r["priority"],
                       "acceptance": [a["statement"] for a in r["acceptance"]]}
                      for r in srs_body["functional"] + srs_body["non_functional"]]}


def _run(monkeypatch, srs_body, thread="srs1"):
    payloads = dict(PAYLOADS, srs=_fills(srs_body))
    class FM(FakeModel):
        def invoke(self, messages):
            self.calls += 1
            return AIMessage(content=json.dumps(payloads[self.stage]))
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: FM(stage))
    g = build_graph(UsageLog("s"))
    return g.invoke({"intent_model": INTENT, "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": thread}})


SRS = {"session_id": "s", "intent_model_ref": "s", "purpose": "p", "scope_summary": "sc",
       "functional": [
           {"requirement_id": "R-1", "title": "Spec file", "statement": "system shall emit a spec",
            "priority": "must", "kind": "functional",
            "acceptance": [{"criterion_id": "AC-1", "statement": "spec exists"}],
            "need_ids": ["N-1"], "intent_fields": ["problem_statement"]}],
       "non_functional": []}


def test_srs_artifact_in_bundle(monkeypatch):
    r = _run(monkeypatch, SRS)
    srs = r["bundle"]["_srs"]
    assert srs["purpose"] == "p"
    assert [q["requirement_id"] for q in srs["functional"]] == ["R-1"]


def test_structure_carried_deterministically(monkeypatch):
    intent = dict(INTENT, glossary={"TMS": "transport mgmt"},
                  constraints=["mandated stack"],
                  assumption_items=[{"statement": "feeds stay up", "source_turns": [0]}],
                  open_questions=["who owns the data?"])
    payloads = dict(PAYLOADS, srs=_fills(SRS))
    class FM(FakeModel):
        def invoke(self, messages):
            self.calls += 1
            return AIMessage(content=json.dumps(payloads[self.stage]))
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: FM(stage))
    g = build_graph(UsageLog("s"))
    r = g.invoke({"intent_model": intent, "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "carry"}})
    srs = r["bundle"]["_srs"]
    # the LLM was never asked for these - they come from the intent model
    assert srs["definitions"] == {"TMS": "transport mgmt"}
    assert srs["constraints"] == ["mandated stack"]
    assert srs["assumptions"][0]["statement"] == "feeds stay up"
    assert srs["open_questions"] == ["who owns the data?"]


def test_solution_language_flagged_and_criterion_dropped(monkeypatch):
    bad = dict(SRS, functional=[dict(SRS["functional"][0], acceptance=[
        {"criterion_id": "AC-1", "statement": "GET /spec returns 200"},
        {"criterion_id": "AC-2", "statement": "spec exists"}])])
    r = _run(monkeypatch, bad, thread="guard")
    func = r["bundle"]["_srs"]["functional"][0]
    assert [a["criterion_id"] for a in func["acceptance"]] == ["AC-2"], "solution-language criterion dropped"
    assert any("solution-language" in f or "solution language" in f for f in r["violations"])


def test_manifest_points_at_srs_and_bundle_version_bumped(monkeypatch):
    r = _run(monkeypatch, SRS)
    m = r["bundle"]["_manifest"]
    assert m["bundle_version"] == "0.2.0"
    assert m["files"]["srs"] == "srs.md"
    assert "requirements" not in m["files"]


def test_render_srs_document():
    from vmodel_contracts import SRSDocument
    srs = SRSDocument.model_validate(dict(SRS, definitions={"TMS": "transport mgmt"},
                                          constraints=["c"], open_questions=["q"]))
    md = render_srs(srs)
    assert "# Software Requirements Specification" in md
    assert "## 3. Definitions" in md and "**TMS**" in md
    assert "## 4. Functional requirements" in md
    assert "R-1 [must] Spec file" in md
    assert "executable_hint" not in md, "no solution mechanics in the SRS render"


def test_tests_stage_owns_ids_and_links(monkeypatch):
    """ADR-0020: the tests skeleton assigns T-ids (acceptance per criterion,
    then unit per item spec) and the criterion/spec links; the model's fills
    carry prose only - emitted ids/levels are irrelevant."""
    r = _run(monkeypatch, SRS, thread="skeleton")
    tests = r["bundle"]["_tests"]
    assert [(t["test_id"], t["level"], t.get("criterion_id"), t.get("spec_id")) for t in tests] == [
        ("T-1", "acceptance", "AC-1", None),
        ("T-2", "unit", None, "SPEC-1"),
    ]

