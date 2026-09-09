"""ADR-0019 founder review gates (opt-in --review): pause after SRS and
architecture; accept / edit (contract-validated) / feedback (stage re-runs) /
reject. The non-review graph is untouched (test_progress.py covers it)."""
import json

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from spec_generator import model_config
from spec_generator.graph import build_graph, srs_document_data, _tech_constraint_block
from spec_generator.progress import ProgressBus
from spec_generator.telemetry import UsageLog
from test_progress import INTENT, PAYLOADS, FakeModel


def _review_graph(monkeypatch):
    box = {}

    class Recording(FakeModel):
        def invoke(self, messages):
            self.last_human = messages[-1].content
            return super().invoke(messages)

    monkeypatch.setattr(model_config, "build_chat_model", lambda stage: box.setdefault(stage, Recording(stage)))
    g = build_graph(UsageLog("s"), checkpointer=MemorySaver(), progress=ProgressBus(), review=True)
    return g, box


def _interrupt_payload(g, cfg):
    st = g.get_state(cfg)
    for t in st.tasks:
        if t.interrupts:
            return t.interrupts[0].value
    return None


def test_gates_pause_after_srs_and_architecture_then_complete(monkeypatch):
    g, _ = _review_graph(monkeypatch)
    cfg = {"configurable": {"thread_id": "g1"}}
    g.invoke({"intent_model": dict(INTENT)}, cfg)
    payload = _interrupt_payload(g, cfg)
    assert payload["gate"] == "srs" and payload["artifact"]["functional"]
    g.invoke(Command(resume={"action": "accept"}), cfg)
    payload = _interrupt_payload(g, cfg)
    assert payload["gate"] == "architecture"
    result = g.invoke(Command(resume={"action": "accept"}), cfg)
    st = g.get_state(cfg)
    assert not st.next and "bundle" in result


def test_gate_feedback_reruns_the_stage_with_the_feedback(monkeypatch):
    g, box = _review_graph(monkeypatch)
    cfg = {"configurable": {"thread_id": "g2"}}
    g.invoke({"intent_model": dict(INTENT)}, cfg)
    assert _interrupt_payload(g, cfg)["gate"] == "srs"
    before = box["srs"].calls
    g.invoke(Command(resume={"action": "regenerate", "feedback": "the SRS misses offline mode"}), cfg)
    assert box["srs"].calls == before + 1
    assert "the SRS misses offline mode" in box["srs"].last_human
    assert "PREVIOUS DRAFT" in box["srs"].last_human
    # the re-run pauses at the SRS gate again for approval of the new draft
    assert _interrupt_payload(g, cfg)["gate"] == "srs"
    # feedback consumed and cleared from state: it never leaks into other stages
    g.invoke(Command(resume={"action": "accept"}), cfg)
    assert "the SRS misses offline mode" not in (box["architecture"].last_human or "")
    assert not g.get_state(cfg).values.get("gate_feedback")
    result = g.invoke(Command(resume={"action": "accept"}), cfg)  # architecture gate
    assert "bundle" in result


def test_gate_edit_lands_via_update_state(monkeypatch):
    g, _ = _review_graph(monkeypatch)
    cfg = {"configurable": {"thread_id": "g3"}}
    g.invoke({"intent_model": dict(INTENT)}, cfg)
    edited = json.loads(json.dumps(_interrupt_payload(g, cfg)["artifact"]))
    edited["functional"][0]["title"] = "Hand-edited title"
    g.update_state(cfg, {"srs": edited})
    g.invoke(Command(resume={"action": "accept"}), cfg)   # srs gate -> runs to architecture gate
    result = g.invoke(Command(resume={"action": "accept"}), cfg)  # architecture gate -> completes
    assert result["bundle"]["_srs"]["functional"][0]["title"] == "Hand-edited title"


def test_architecture_prompt_carries_tech_constraints():
    intent = dict(INTENT)
    intent["tech_preferences"] = [
        {"area": "language", "preference": "python", "strength": "required", "source_turns": [2]},
        {"area": "cloud", "preference": "AWS", "strength": "preferred", "source_turns": [5]},
    ]
    block = _tech_constraint_block(intent)
    assert "REQUIRED language: python" in block
    assert "preferred cloud: AWS" in block
    assert _tech_constraint_block(INTENT) == ""  # no preferences: prompt unchanged


def test_srs_document_data_injects_structure_from_intent():
    data = srs_document_data(INTENT, dict(PAYLOADS["srs"]))
    assert data["definitions"] == INTENT.get("glossary", {})
    assert "constraints" in data and "open_questions" in data



def test_gate_render_of_contract_invalid_artifact_is_friendly_not_a_traceback():
    """His live crash: render_srs(SRSDocument.model_validate(...)) at the SRS
    gate raised a raw pydantic traceback when the draft lacked 'title'. Gate
    render/validate failures get the friendly three-part treatment instead."""
    from test_progress import INTENT, PAYLOADS
    import copy

    from spec_generator.cli import _render_gate
    from spec_generator.errors import UserFacingError

    bad = copy.deepcopy(PAYLOADS["srs"])
    for r in bad["functional"]:
        del r["title"]
    with pytest.raises(UserFacingError) as ei:
        _render_gate("srs", bad, dict(INTENT))
    rendered = ei.value.render()
    assert "does not pass the contract" in rendered
    assert "title" in rendered
