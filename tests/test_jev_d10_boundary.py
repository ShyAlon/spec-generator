"""JEV-D10 at the pipeline boundary: control plane, acceptance, quarantine, evidence."""
import json


import pytest
from langchain_core.messages import AIMessage
from reference_checkpoint import reference_checkpoint
from conftest import SYSTEM_BASELINE

from spec_generator import model_config
from spec_generator.cli import _make_partials, _make_persist
from spec_generator.content_acceptance import DIRECTIVE_CATALOGUE
from spec_generator.errors import AssemblyBlocked, ContentRejected
from spec_generator.findings_ledger import Ledger
from spec_generator.graph import build_graph
from spec_generator.telemetry import UsageLog
from test_per_item_resume import PLAN, _arch_answer
from test_progress import INTENT, PAYLOADS

GOOD = ["# API\n\nExposes saveProgress(record) and loadProgress(). Errors: storage unavailable returns a typed error.",
        "# Web\n\nRenders the lesson screen and keyboard. Errors: audio unavailable shows a notice and continues."]


def _model(bodies, seen_messages=None):
    queue = list(bodies)

    class M:
        def __init__(self, stage, schema):
            self.stage, self.schema = stage, schema

        def stream(self, messages):
            if seen_messages is not None:
                seen_messages.append((self.stage, [str(m.content) for m in messages]))
            if self.stage == "srs":
                if set(self.schema.get("properties", {})) == {"purpose", "scope_summary"}:
                    out = {"purpose": "p", "scope_summary": "s"}
                else:
                    out = {"title": "t", "statement": "system shall x", "priority": "must", "acceptance": ["a"]}
            elif self.stage == "items":
                out = PLAN if self.schema.get("type") == "array" else {"body_markdown": queue.pop(0)}
            elif self.stage == "tests":
                wanted = self.schema["properties"]["fills"]["required"]
                out = {"fills": {t: PAYLOADS["tests"]["fills"].get(t, {
                    "description": f"{t} holds", "components": ["api"], "runner": "pytest",
                    "definition": {"preconditions": [], "steps": ["Run x"], "expected_outcomes": ["passes"],
                                   "cleanup": [], "cases": [], "execution": {}}}) for t in wanted}}
            elif self.stage == "architecture":
                out = _arch_answer(self.schema)
            else:
                out = PAYLOADS[self.stage]
            yield AIMessage(content=json.dumps(out))
    return M


def _run(monkeypatch, tmp_path, bodies, name="r", seen=None):
    monkeypatch.delenv("SPEC_GENERATOR_BULK", raising=False)
    factory = _model(bodies, seen)
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: factory(stage, kw.get("schema") or {}))
    partials = _make_partials(tmp_path)
    persist = _make_persist(dict(INTENT), tmp_path)
    g = build_graph(UsageLog("s"), checkpointer=reference_checkpoint(str(tmp_path / f"{name}.db")),
                    persist=persist, partials=partials)
    return g, partials


def test_watchdog_retry_sends_no_instruction_text(monkeypatch):
    """The cause fix: the old retry sentence entered the product-content channel."""
    from vmodel_contracts.runtime import AdaptiveWatchdogTimeout
    from spec_generator import graph

    sent = []

    def fake_invoke(stage, payload, log, progress=None, feedback=None, **kw):
        sent.append(feedback)
        if len(sent) == 1:
            raise AdaptiveWatchdogTimeout(stage, 2.0, 1.0, None)
        return '{"ok": 1}'

    monkeypatch.setattr(graph, "_invoke", fake_invoke)
    monkeypatch.setattr(model_config, "_console_line", lambda *_: None)
    assert graph._call_json("items", "payload", UsageLog("s")) == {"ok": 1}
    assert sent == [None, None]
    assert all("exceeded its runtime budget" not in str(f) for f in sent)


def test_contract_retry_feedback_is_framed_as_control_not_content():
    from spec_generator.graph import _CONTROL_FRAME
    assert "not product content" in _CONTROL_FRAME and "Never copy" in _CONTROL_FRAME


def test_directive_echo_stops_the_stage_keeps_evidence_and_saves_no_partial(monkeypatch, tmp_path):
    echo = DIRECTIVE_CATALOGUE["watchdog-retry"]
    g, partials = _run(monkeypatch, tmp_path, [GOOD[0], echo])
    with pytest.raises(ContentRejected) as caught:
        g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "t"}})
    assert caught.value.spec_id == "SPEC-2"
    assert [d["spec_id"] for d in partials.load("items")] == ["SPEC-1"]   # SPEC-2 never saved as completed
    evidence = json.loads((tmp_path / "content-rejections.json").read_text())
    assert evidence[0]["spec_id"] == "SPEC-2" and evidence[0]["body"] == echo
    assert evidence[0]["results"][0]["catalogue_id"] == "watchdog-retry"
    assert not (tmp_path / "tests.json").exists()   # nothing derived


def test_wrapper_and_unicode_variants_of_an_echo_also_stop(monkeypatch, tmp_path):
    variant = "## Piano module\n\n**" + DIRECTIVE_CATALOGUE["watchdog-retry"].upper().replace("RETURN", "Ｒｅｔｕｒｎ") + "**"
    g, _ = _run(monkeypatch, tmp_path, [GOOD[0], variant])
    with pytest.raises(ContentRejected):
        g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "t"}})


def test_duplicate_body_stops_the_stage(monkeypatch, tmp_path):
    g, partials = _run(monkeypatch, tmp_path, ["# SPEC-1: API\n" + GOOD[0].split("\n", 1)[1], "# SPEC-2: Web\n" + GOOD[0].split("\n", 1)[1]])
    with pytest.raises(ContentRejected) as caught:
        g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "t"}})
    r = caught.value.results[0]
    assert r["rule"] == "duplicate" and r["sibling_id"] == "SPEC-1" and r["score"] is not None


def test_resumed_partial_goes_through_the_same_boundary(monkeypatch, tmp_path):
    """A pre-existing partial (any earlier run) is not trusted: duplicates stop on resume."""
    g, partials = _run(monkeypatch, tmp_path, [GOOD[0]])
    from spec_generator import items_resume
    original = items_resume.prepare_plan

    def seed_valid_plan(partials, payload, validator, propose):
        # A durable plan from an earlier run, but bodies still need acceptance.
        plan = validator(PLAN)
        partials.save("items_plan", {"version": 1,
            "context_sha256": items_resume.context_digest(payload),
            "plan_sha256": items_resume.digest(plan), "plan": plan})
        partials.save("items", [{**p, "body_markdown": GOOD[0]} for p in plan])
        return original(partials, payload, validator, lambda: pytest.fail("plan replay"))

    monkeypatch.setattr(items_resume, "prepare_plan", seed_valid_plan)
    with pytest.raises(ContentRejected):
        g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "t"}})


def test_gray_body_opens_a_finding_quarantines_the_spec_and_derives_no_tests(monkeypatch, tmp_path):
    # a real section quoting the directive inside useful text: founder's C-07 shape
    quoting = ("Store note events locally.\n\n" + DIRECTIVE_CATALOGUE["watchdog-retry"] + "\n\nReturn the grade.")
    g, _ = _run(monkeypatch, tmp_path, [GOOD[0], quoting])
    cfg = {"configurable": {"thread_id": "t"}}
    with pytest.raises(AssemblyBlocked) as blocked:        # the gate stops assembly while a finding is open
        g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, cfg)
    assert "1 open finding" in blocked.value.reasons[0]
    state = g.get_state(cfg).values
    ledger = Ledger.from_dict(state["findings_ledger"])
    opened = ledger.open_findings()
    assert len(opened) == 1 and opened[0]["targets"] == ["SPEC-2"] and opened[0]["rule_id"] == "echo.directive"
    assert all(t.get("spec_id") != "SPEC-2" for t in state["tests"])   # nothing derives from the quarantined spec
    assert json.loads((tmp_path / "findings-ledger.json").read_text())["findings"]
    assert not (tmp_path / "manifest.json").exists()                   # nothing shipped
