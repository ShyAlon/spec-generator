"""Progress events + crash-resume for the generation pipeline."""
import json
import sqlite3

import pytest
from langchain_core.messages import AIMessage

from spec_generator import model_config
from spec_generator.graph import build_graph
from spec_generator.progress import ProgressBus
from langgraph.checkpoint.sqlite import SqliteSaver
from spec_generator.telemetry import UsageLog

PAYLOADS = {
    "needs": {"session_id": "s", "needs": [
        {"need_id": "N-1", "stakeholder": "founder", "statement": "needs specs",
         "kind": "functional", "intent_fields": ["problem_statement"], "intent_need_ids": ["SN-1"]}]},
    # ADR-0020: the model returns PROSE FILLS keyed by pipeline-assigned slot
    # ids; code owns ids, kinds, sections, and trace links (see CONTRACT_SRS
    # for the contract-shaped result the node assembles from these).
    "srs": {"purpose": "p", "scope_summary": "sc",
            "fills": {"R-1": {"title": "Spec file",
                              "statement": "system shall emit a spec", "priority": "must",
                              "acceptance": ["spec exists"]}}},
    "architecture": {"session_id": "s", "stack": {"language": "ts"},
                     "components": [{"name": "api", "responsibility": "serve"}],
                     "decisions": [{"title": "t", "decision": "x", "rationale": "y",
                                    "driver": "simplicity",
                                    "options_considered": [{"option": "z", "reason": "heavier"}]}]},
    "items": [{"slug": "api", "kind": "api_surface", "title": "API",
               "body_markdown": "# API", "requirement_ids": ["R-1"]}],  # code numbers it SPEC-1
    "tests": {"fills": {"T-1": {"description": "criterion holds",
                                "runner": "http", "definition": {"preconditions": [], "steps": ["GET /"], "expected_outcomes": ["HTTP 200"], "cleanup": [], "cases": [], "execution": {}}},
                        "T-2": {"description": "spec renders",
                                "runner": "pytest", "definition": {"preconditions": [], "steps": ["Run x"], "expected_outcomes": ["passes"], "cleanup": [], "cases": [], "execution": {}}}}},
    "review": [],
}

# the contract-shaped SRS the srs node assembles from PAYLOADS needs+srs
CONTRACT_SRS = {"session_id": "s", "intent_model_ref": "s", "purpose": "p", "scope_summary": "sc",
                "functional": [
                    {"requirement_id": "R-1", "title": "Spec file", "statement": "system shall emit a spec",
                     "priority": "must", "kind": "functional",
                     "acceptance": [{"criterion_id": "AC-1", "statement": "spec exists"}],
                     "need_ids": ["N-1"], "intent_fields": ["problem_statement"]}],
                "non_functional": []}

INTENT = {"session_id": "s", "problem_statement": "p", "target_users": ["u"],
          "core_value_proposition": "v",
          "stakeholder_needs": [{"need_id": "SN-1", "statement": "founders need specs",
                                 "kind": "functional", "source_turns": [0]}],
          "in_scope": ["a"], "out_of_scope": ["b"],
          "external_integrations": [], "constraints": [], "open_questions": [], "source_turns": {}}


class FakeModel:
    def __init__(self, stage):
        self.stage = stage
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=json.dumps(PAYLOADS[self.stage]))

    def stream(self, messages):
        # the production path streams; delegate to invoke (subclass overrides
        # included) and yield the completion in chunks, like a real model
        text = self.invoke(messages).content
        step = max(1, len(text) // 3)
        for i in range(0, len(text), step):
            yield AIMessage(content=text[i:i + step])


@pytest.fixture
def fake_models(monkeypatch):
    box = {}
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: box.setdefault(stage, FakeModel(stage)))
    return box


def test_full_pipeline_progress_events(fake_models):
    bus = ProgressBus()
    events = []
    bus.subscribe(events.append)
    g = build_graph(UsageLog("s"), progress=bus)
    r = g.invoke({"intent_model": INTENT}, {"configurable": {"thread_id": "s"}})
    assert "bundle" in r
    seq = [(e.stage, e.status) for e in events]
    for stage in ("needs", "srs", "architecture", "items", "tests", "review", "assemble"):
        assert (stage, "started") in seq and (stage, "completed") in seq, stage
    # starts come in pipeline order
    starts = [e.stage for e in events if e.status == "started"]
    assert starts == ["needs", "srs", "architecture", "items", "tests", "review", "assemble"]


def test_crash_mid_pipeline_resumes_without_repaying(tmp_path, fake_models):
    # die at the architecture stage
    class DieAtArch(FakeModel):
        def invoke(self, messages):
            if self.stage == "architecture":
                raise RuntimeError("process died")
            return super().invoke(messages)

    fake_models["architecture"] = DieAtArch("architecture")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: fake_models.setdefault(stage, DieAtArch(stage) if stage == "architecture" else FakeModel(stage)))
    db = str(tmp_path / "cp.sqlite")
    cfg = {"configurable": {"thread_id": "s"}}
    g = build_graph(UsageLog("s"), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    with pytest.raises(RuntimeError):
        g.invoke({"intent_model": INTENT}, cfg)

    # "fix" and rerun: healthy models, same db
    fake_models["architecture"] = FakeModel("architecture")
    g2 = build_graph(UsageLog("s"), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: fake_models.setdefault(stage, FakeModel(stage)))
    r = g2.invoke(None, cfg)
    assert "bundle" in r
    assert fake_models["needs"].calls == 1, "needs stage must NOT be re-paid after resume"
    assert fake_models["srs"].calls == 1
    monkeypatch.undo()


def test_stage_start_lines_carry_input_counts(fake_models):
    """The opening line of each stage states its INPUT cardinality (known
    before the call) - never an output count (not knowable in advance)."""
    bus = ProgressBus()
    events = []
    bus.subscribe(events.append)
    g = build_graph(UsageLog("s"), progress=bus)
    g.invoke({"intent_model": INTENT}, {"configurable": {"thread_id": "s"}})
    started = {e.stage: e for e in events if e.status == "started"}

    n_sn = len(INTENT["stakeholder_needs"])
    assert started["needs"].detail == f"stage 1/6, from {n_sn} interview needs (+ any implied)"

    n_needs = len(PAYLOADS["needs"]["needs"])
    assert started["srs"].detail == f"stage 2/6, from {n_needs} stakeholder needs"

    n_reqs = len(CONTRACT_SRS["functional"]) + len(CONTRACT_SRS["non_functional"])
    assert started["architecture"].detail == f"stage 3/6, from {n_reqs} requirements"
    assert started["items"].detail == f"stage 4/6, covering {n_reqs} requirements"

    n_ac = sum(len(r["acceptance"]) for sec in ("functional", "non_functional") for r in CONTRACT_SRS[sec])
    assert started["tests"].detail == f"stage 5/6, from {n_ac} acceptance criteria + {len(PAYLOADS['items'])} module specs"

    # no detail may promise an output count
    for e in started.values():
        if e.detail:
            assert "will produce" not in e.detail and "expecting" not in e.detail

