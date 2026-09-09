"""ADR-0020 typed structural tools: model proposes, code disposes."""
import copy

import pytest

from spec_generator.skeleton import (ModelStructureError, build_srs_skeleton,
                                     build_test_skeleton, merge_test_fills)
from spec_generator.tools import (SrsToolbox, VerificationToolbox, run_tool_loop,
                                  tools_enabled)

def _srs_slots():
    return copy.deepcopy(build_srs_skeleton(NEEDS))


NEEDS = {"needs": [
    {"need_id": "SN-1", "stakeholder": "founder", "statement": "Upload a CSV of signups", "priority": "high"},
    {"need_id": "SN-2", "stakeholder": "founder", "statement": "See conversion per source", "priority": "medium"},
]}
SRS_SLOTS = None  # kept for reference; tests use _srs_slots() deep copies
SPEC = {"spec_id": "SPEC-1", "component": "watcher", "item_id": "I-1",
        "title": "Poller", "description": "polls", "unit_test": "t",
        "depends_on": [], "exposes": ["GET /api/x"], "data": [], "risks": [],
        "requirement_ids": ["R-1"], "estimated_effort": "S"}


def test_opt_in_off_by_default(monkeypatch):
    monkeypatch.delenv("SPEC_GENERATOR_TOOLS", raising=False)
    assert tools_enabled() is False
    monkeypatch.setenv("SPEC_GENERATOR_TOOLS", "1")
    assert tools_enabled() is True
    monkeypatch.setenv("SPEC_GENERATOR_TOOLS", "0")
    assert tools_enabled() is False


class TestSplitRequirement:
    def test_happy_path(self):
        tb = SrsToolbox(_srs_slots(), {"SN-1", "SN-2"})
        out = tb.call("split_requirement", {
            "requirement_id": "R-1",
            "parts": [{"title": "Upload", "statement": "The system shall accept CSV uploads of signup rows.", "priority": "must"},
                      {"title": "Validate", "statement": "The system shall reject malformed rows with row-level errors.", "priority": "must"}]})
        assert out["retired"] == "R-1" and out["new_ids"] == ["R-3", "R-4"]
        assert {s["requirement_id"] for s in tb.slots} == {"R-2", "R-3", "R-4"}
        new = {s["requirement_id"]: s for s in tb.slots}
        assert new["R-3"]["kind"] == _srs_slots()[0]["kind"]  # inherits whatever the retired slot had
        assert new["R-3"]["need_ids"] == ["SN-1"] and new["R-4"]["intent_fields"] == new["R-3"]["intent_fields"]
        assert new["R-3"]["seed"]["priority"] == "must"
        assert tb.applied == ["split R-1 -> R-3, R-4"]

    @pytest.mark.parametrize("args, msg", [
        ({"requirement_id": "R-9", "parts": [{}, {}]}, "no such requirement slot"),
        ({"requirement_id": "R-1", "parts": [{"title": "x", "statement": "y", "priority": "must"}]}, "at least 2 parts"),
        ({"requirement_id": "R-1", "parts": [{"title": "", "statement": "y", "priority": "must"},
                                             {"title": "x", "statement": "y", "priority": "must"}]}, "non-empty title"),
        ({"requirement_id": "R-1", "parts": [{"title": "a", "statement": "b", "priority": "won't"},
                                             {"title": "x", "statement": "y", "priority": "must"}]}, "priority must be"),
    ])
    def test_rejections(self, args, msg):
        tb = SrsToolbox(_srs_slots(), {"SN-1"})
        with pytest.raises(ModelStructureError, match=msg):
            tb.call("split_requirement", args)

    def test_next_split_continues_ids(self):
        tb = SrsToolbox(_srs_slots(), {"SN-1", "SN-2"})
        parts = [{"title": "a", "statement": "s", "priority": "must"},
                 {"title": "b", "statement": "s", "priority": "must"}]
        tb.call("split_requirement", {"requirement_id": "R-1", "parts": parts})
        out = tb.call("split_requirement", {"requirement_id": "R-2", "parts": parts})
        assert out["new_ids"] == ["R-5", "R-6"]


class TestLinkTrace:
    def test_link_and_dedup(self):
        tb = SrsToolbox(_srs_slots(), {"SN-1", "SN-2"})
        tb.call("link_requirement_to_need", {"requirement_id": "R-1", "need_id": "SN-2"})
        out = tb.call("link_requirement_to_need", {"requirement_id": "R-1", "need_id": "SN-2"})
        assert out["need_ids"] == ["SN-1", "SN-2"]
        assert tb.applied == ["linked R-1 -> SN-2"]

    def test_unknown_need_rejected(self):
        tb = SrsToolbox(_srs_slots(), {"SN-1"})
        with pytest.raises(ModelStructureError, match="no such stakeholder need"):
            tb.call("link_requirement_to_need", {"requirement_id": "R-1", "need_id": "SN-9"})


class TestAddTest:
    def _tb(self):
        slots = build_test_skeleton(
            {"functional": [{"requirement_id": "R-1", "kind": "functional",
                             "title": "t", "statement": "s", "priority": "must",
                             "need_ids": ["SN-1"],
                             "acceptance": [{"criterion_id": "AC-R-1-1", "statement": "a", "testable": True}]}],
             "non_functional": [], "out_of_scope": [], "open_questions": []},
            [SPEC])
        return VerificationToolbox(slots, {"AC-R-1-1"}, {"SPEC-1"})

    def test_happy_path(self):
        tb = self._tb()
        out = tb.call("add_test", {"level": "module", "spec_id": "SPEC-1",
                                   "description": "upload then report flow", "runner": "http",
                                   "definition": {"url": "/api/report", "expect_status": 200}})
        assert out["test_id"] == "T-3"
        slot = [s for s in tb.slots if s["test_id"] == "T-3"][0]
        assert slot["level"] == "module" and slot["criterion_id"] is None
        assert slot["seed"]["runner"] == "http"

    @pytest.mark.parametrize("args, msg", [
        ({"level": "e2e", "spec_id": "SPEC-1", "description": "d", "runner": "http", "definition": {}}, "level must be"),
        ({"level": "acceptance", "criterion_id": "AC-9", "description": "d", "runner": "http", "definition": {}}, "existing criterion_id"),
        ({"level": "unit", "spec_id": "SPEC-9", "description": "d", "runner": "http", "definition": {}}, "existing spec_id"),
        ({"level": "unit", "spec_id": "SPEC-1", "description": "", "runner": "http", "definition": {}}, "description, runner"),
    ])
    def test_rejections(self, args, msg):
        with pytest.raises(ModelStructureError, match=msg):
            self._tb().call("add_test", args)

    def test_seed_adopted_as_fill(self):
        tb = self._tb()
        tb.call("add_test", {"level": "system", "spec_id": "SPEC-1", "description": "full flow",
                             "runner": "http", "definition": {"url": "/api/x", "expect_status": 200}})
        doc = merge_test_fills(tb.slots, {"fills": [
            {"test_id": "T-1", "description": "d", "runner": "http", "definition": {"url": "/u", "expect_status": 200}},
            {"test_id": "T-2", "description": "d", "runner": "http", "definition": {"url": "/u", "expect_status": 200}}]})
        assert len(doc) == 3
        added = doc[2]
        assert added["test_id"] == "T-3" and added["description"] == "full flow"
        assert added["spec_id"] == "SPEC-1" and added["level"] == "system"


class _ScriptedModel:
    """bind_tools-able fake: pops scripted invoke() results."""
    def __init__(self, results):
        self._results = list(results)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        r = self._results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class _AI:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.usage_metadata = {"input_tokens": 1, "output_tokens": 1}


class _Log:
    def __init__(self):
        self.records = []

    def record(self, *a, **k):
        self.records.append(a)


def _run(monkeypatch, tb, scripted):
    monkeypatch.setattr("spec_generator.model_config.build_chat_model",
                        lambda stage, structured=True: _ScriptedModel(scripted))
    return run_tool_loop("srs", tb, "payload", _Log())


class TestToolLoop:
    def test_applies_then_stops_clean(self, monkeypatch):
        tb = SrsToolbox(_srs_slots(), {"SN-1", "SN-2"})
        applied, degraded = _run(monkeypatch, tb, [
            _AI([{"name": "split_requirement", "id": "c1",
                  "args": {"requirement_id": "R-1",
                           "parts": [{"title": "a", "statement": "s", "priority": "must"},
                                     {"title": "b", "statement": "s", "priority": "must"}]}}]),
            _AI([])])
        assert degraded is False and applied == ["split R-1 -> R-3, R-4"]
        assert {s["requirement_id"] for s in tb.slots} == {"R-2", "R-3", "R-4"}

    def test_invalid_args_retried_then_stop(self, monkeypatch):
        tb = SrsToolbox(_srs_slots(), {"SN-1"})
        applied, degraded = _run(monkeypatch, tb, [
            _AI([{"name": "split_requirement", "id": "c1",
                  "args": {"requirement_id": "R-9", "parts": [{}, {}]}}]),
            _AI([])])
        assert degraded is False and applied == [] and len(tb.slots) == 2

    def test_three_bad_rounds_degrade(self, monkeypatch):
        tb = SrsToolbox(_srs_slots(), {"SN-1"})
        bad = _AI([{"name": "add_test", "id": "c", "args": {"level": "e2e"}}])
        applied, degraded = _run(monkeypatch, tb, [bad, bad, bad, bad])
        assert degraded is True and applied == []

    def test_model_error_degrades(self, monkeypatch):
        tb = SrsToolbox(_srs_slots(), {"SN-1"})
        applied, degraded = _run(monkeypatch, tb, [ValueError("tool_call parse blew up")])
        assert degraded is True and applied == []

    def test_connection_error_propagates(self, monkeypatch):
        tb = SrsToolbox(_srs_slots(), {"SN-1"})
        with pytest.raises(ConnectionError):
            _run(monkeypatch, tb, [ConnectionError("connection refused to ollama")])


class _ToolScript:
    """The model behind the tool loop: bind_tools-able, pops scripted responses."""
    def __init__(self, responses):
        self._responses = list(responses)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        from langchain_core.messages import AIMessage
        return AIMessage(content="", tool_calls=r)


def _run_graph_with_tools(monkeypatch, scripted):
    from spec_generator import model_config
    from spec_generator.graph import build_graph
    from spec_generator.telemetry import UsageLog
    from test_progress import INTENT, FakeModel

    monkeypatch.setenv("SPEC_GENERATOR_TOOLS", "1")

    def build(stage, structured=True):
        if not structured:
            return _ToolScript(scripted.get(stage, [[]]))
        return FakeModel(stage)

    monkeypatch.setattr(model_config, "build_chat_model", build)
    return build_graph(UsageLog("s")).invoke({"intent_model": INTENT},
                                             {"configurable": {"thread_id": "tools"}})


def test_tools_opt_in_full_pipeline_green(monkeypatch):
    """SPEC_GENERATOR_TOOLS=1: the loop runs before the fill; a valid proposal
    applies; the rest of the V-model pipeline is untouched."""
    r = _run_graph_with_tools(monkeypatch, {
        "srs": [[{"name": "link_requirement_to_need",
                  "args": {"requirement_id": "R-1", "need_id": "N-1"}, "id": "c1"}], []],
        "tests": [[{"name": "add_test",
                    "args": {"level": "module", "spec_id": "SPEC-api",
                             "description": "end-to-end module flow", "runner": "http",
                             "definition": {"url": "/api/x", "expect_status": 200}}, "id": "c2"}], []]})
    srs = r["bundle"]["_srs"]
    assert [q["requirement_id"] for q in srs["functional"]] == ["R-1"]
    assert srs["functional"][0]["need_ids"] == ["N-1"]  # dedup: link was already implied
    tests = r["bundle"]["_tests"]
    assert any(t["test_id"] == "T-3" and t["level"] == "module" and t["spec_id"] == "SPEC-api"
               for t in tests), "add_test proposal adopted as-is"


def test_tools_degrade_to_plain_skeleton(monkeypatch):
    """A model that cannot emit tool calls degrades: the plain 1:1 skeleton
    carries through and the run still completes."""
    r = _run_graph_with_tools(monkeypatch, {"srs": [ValueError("tool calling unsupported")],
                                            "tests": [ValueError("tool calling unsupported")]})
    srs = r["bundle"]["_srs"]
    assert [q["requirement_id"] for q in srs["functional"]] == ["R-1"]
