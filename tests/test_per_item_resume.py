from conftest import SYSTEM_BASELINE
"""2026-09-10: Ctrl+C at T-34 of 36 reran the whole stage. Now each completed
item inside a looped authoring stage (srs fills, items bodies, tests fills)
is preserved the moment it closes and skipped on rerun - same doctrine as
completed stages. Plus: the plan phase no longer masquerades as item lines,
and micro-calls print a start line when they begin (+ tokens when reported).
"""
import json
import sqlite3

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from spec_generator import model_config
from spec_generator.cli import _make_partials
from spec_generator.graph import build_graph
from spec_generator.progress import ProgressBus, stderr_printer
from spec_generator.telemetry import UsageLog
from test_progress import INTENT, PAYLOADS

PLAN = [{"slug": "api", "kind": "api_surface", "title": "API",
         "requirement_ids": ["R-1"], "depends_on": []},
        {"slug": "web", "kind": "ui_flow", "title": "Web",
         "requirement_ids": ["R-1"], "depends_on": ["api"]}]




def _arch_answer(schema):
    """0.13.0 micro-pipeline: answer the one small call this schema shapes."""
    props = set(schema.get("properties", {}))
    if props == {"stack_keys"}:
        return {"stack_keys": ["language"]}
    if props == {"technology", "off_list_justification", "baseline_refs"}:
        return {"technology": "python", "off_list_justification": None, "baseline_refs": ["runtime:spec-generator:1"]}
    if props == {"responsibility"}:
        return {"responsibility": "serve"}
    if props == {"position"}:
        return {"position": "fast"}
    if "driver" in props:
        return {"title": "t", "decision": "x", "rationale": "y", "driver": "simplicity",
                "options_considered": [{"option": "z", "reason": "heavier"}],
                "baseline_refs": ["repo:spec-generator"], "evidence": [],
                "affected_repositories": [], "executes_concept_ids": []}
    if schema.get("type") == "array":
        item_props = set(schema["items"].get("properties", {}))
        if "slug" in item_props:
            return [{"slug": "t"}]
        return [{"name": "api", "requirement_ids": ["R-1"],
                 "baseline_refs": ["repo:spec-generator"], "migration": False}]
    raise AssertionError(f"unexpected architecture schema: {schema}")

def _routing(calls, fail_after_bodies=None, bus_events=None):
    class RoutingModel:
        def __init__(self, stage, schema):
            self.stage, self.schema = stage, schema

        def stream(self, messages):
            bodies = sum(1 for st, sc in calls if st == "items" and sc.get("type") != "array")
            calls.append((self.stage, self.schema))
            if self.stage == "srs":
                if set(self.schema.get("properties", {})) == {"purpose", "scope_summary"}:
                    out = {"purpose": "p", "scope_summary": "sc"}
                else:
                    out = {"title": "t", "statement": "system shall x", "priority": "must",
                           "acceptance": ["a"]}
            elif self.stage == "items":
                if self.schema.get("type") == "array":
                    out = PLAN
                else:
                    if fail_after_bodies is not None and bodies >= fail_after_bodies:
                        raise RuntimeError("simulated crash mid-stage")
                    out = {"body_markdown": "# body"}
            elif self.stage == "tests":
                wanted = self.schema["properties"]["fills"]["required"]
                out = {"fills": {tid: PAYLOADS["tests"]["fills"].get(
                    tid, {"description": f"{tid} holds", "runner": "pytest",
                          "definition": {"preconditions": [], "steps": ["Run x"], "expected_outcomes": ["passes"], "cleanup": [], "cases": [], "execution": {}}}) for tid in wanted}}
            elif self.stage == "architecture":
                out = _arch_answer(self.schema)
            else:
                out = PAYLOADS[self.stage]
            yield AIMessage(content=json.dumps(out))

    return RoutingModel


def _build(monkeypatch, calls, tmp_path, name, fail_after_bodies=None, partials=None):
    monkeypatch.delenv("SPEC_GENERATOR_BULK", raising=False)
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: _routing(calls, fail_after_bodies)(stage, kw.get("schema") or {}))
    bus = ProgressBus()
    events = []
    bus.subscribe(events.append)
    db = str(tmp_path / f"{name}.sqlite")
    g = build_graph(UsageLog("s"), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)),
                    progress=bus, partials=partials)
    return g, events


def test_crash_mid_items_reruns_only_the_missing_bodies(monkeypatch, tmp_path):
    partials = _make_partials(tmp_path)
    calls1 = []
    g, _ = _build(monkeypatch, calls1, tmp_path, "run1", fail_after_bodies=1, partials=partials)
    with pytest.raises(RuntimeError):
        g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "s"}})
    saved = partials.load("items")
    assert saved and [d["spec_id"] for d in saved] == ["SPEC-1"]  # first body survived the crash

    calls2 = []
    g2, _ = _build(monkeypatch, calls2, tmp_path, "run2", partials=partials)
    r = g2.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": "ignored"}) \
        if False else g2.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "s2"}})
    bodies = [sc for st, sc in calls2 if st == "items" and sc.get("type") != "array"]
    assert len(bodies) == 1  # SPEC-2 only - SPEC-1 was preserved, not re-bought
    assert [i["spec_id"] for i in r["items"]] == ["SPEC-1", "SPEC-2"]
    assert partials.load("items") is None  # partial cleared on stage completion


def test_crash_mid_tests_asks_only_for_missing_fills(monkeypatch, tmp_path):
    partials = _make_partials(tmp_path)
    partials.save("tests_fills", {"T-1": PAYLOADS["tests"]["fills"]["T-1"]})  # T-1 of T-1..T-2 done
    calls = []
    g, _ = _build(monkeypatch, calls, tmp_path, "run", partials=partials)
    r = g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "s"}})
    tests_calls = [sc for st, sc in calls if st == "tests"]
    wanted = [tuple(sc["properties"]["fills"]["required"]) for sc in tests_calls]
    assert sorted(t for (t,) in wanted) == ["T-2", "T-3"]  # one micro-call per missing slot; T-1 preserved
    assert [t["test_id"] for t in r["tests"]] == ["T-1", "T-2", "T-3"]
    assert partials.load("tests_fills") is None


def test_plan_phase_prints_one_plan_line_not_fake_item_lines(monkeypatch, tmp_path):
    calls = []
    _, events = _build(monkeypatch, calls, tmp_path, "run", partials=_make_partials(tmp_path))[0:2] \
        if False else (None, None)
    g, events = _build(monkeypatch, calls, tmp_path, "run", partials=_make_partials(tmp_path))
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "s"}})
    items_events = [e for e in events if e.stage == "items" and e.status == "item"]
    assert any(e.detail == "plan: 2 specs to write" for e in items_events)
    spec_lines = [e.detail for e in items_events if e.detail.startswith("SPEC-")]
    assert len(spec_lines) == 2  # one completion per body - never the plan list again


def test_micro_call_emits_start_and_completion_lines(monkeypatch, tmp_path):
    calls = []
    g, events = _build(monkeypatch, calls, tmp_path, "run", partials=_make_partials(tmp_path))
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "s"}})
    started = [e for e in events if e.status == "item_started"]
    done = [e for e in events if e.status == "item" and e.elapsed_s is not None]
    assert any(e.detail.startswith("R-1") for e in started)      # srs fill
    assert any(e.detail.startswith("SPEC-1 (api)") for e in started)  # items body: numbered + slug
    assert any(e.detail.startswith("R-1") for e in done)
    # a start line always precedes its completion (same stage, same id)
    for s in started:
        later = [e for e in done if e.stage == s.stage and e.ts >= s.ts]
        assert later, f"no completion after start for {s.detail}"


def test_completion_line_carries_tokens_when_reported(monkeypatch, tmp_path):
    from spec_generator.graph import _invoke
    from spec_generator.telemetry import UsageLog

    class Chunk:
        def __init__(self, text, usage=None):
            self.content, self.usage_metadata = text, usage

    class TokModel:
        def stream(self, messages):
            yield Chunk('{"purpose": ', None)
            yield Chunk('"p", "scope_summary": "s"}', {"input_tokens": 10, "output_tokens": 32,
                                                       "total_tokens": 42})

    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: TokModel())
    bus = ProgressBus()
    events = []
    bus.subscribe(events.append)
    _invoke("srs", "{}", UsageLog("s"), progress=bus, stream_items=False, item_label="R-1 do a thing")
    started = [e for e in events if e.status == "item_started"]
    done = [e for e in events if e.status == "item"]
    assert started and started[0].detail == "R-1 do a thing"
    assert done and done[0].detail == "R-1 do a thing · 42 tok"
    assert done[0].elapsed_s is not None


def test_printer_writes_start_then_completion(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)  # redirected log: two lines
    calls = []
    g, _ = _build(monkeypatch, calls, tmp_path, "run", partials=_make_partials(tmp_path))
    bus = ProgressBus()
    bus.subscribe(stderr_printer())
    from spec_generator.graph import _invoke
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: _routing(calls)(stage, kw.get("schema") or {}))
    _invoke("items", "{}", UsageLog("s"), progress=bus, stream_items=False,
            item_label="SPEC-1 (api) API")
    out = capsys.readouterr().err
    import re
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\] SPEC-1 \(api\) API", out)         # start line
    assert re.search(r"\[\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}\] SPEC-1 \(api\) API", out)  # completion
