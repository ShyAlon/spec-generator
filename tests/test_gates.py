from conftest import SYSTEM_BASELINE
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
from test_progress import CONTRACT_SRS, INTENT, PAYLOADS, FakeModel


def _review_graph(monkeypatch):
    box = {}

    class Recording(FakeModel):
        def invoke(self, messages):
            self.last_human = messages[-1].content
            return super().invoke(messages)

    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: box.setdefault(stage, Recording(stage)))
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
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, cfg)
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
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, cfg)
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
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, cfg)
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
    data = srs_document_data(INTENT, dict(CONTRACT_SRS))
    assert data["definitions"] == INTENT.get("glossary", {})
    assert "constraints" in data and "open_questions" in data



def test_gate_render_of_contract_invalid_artifact_is_friendly_not_a_traceback():
    """His live crash: render_srs(SRSDocument.model_validate(...)) at the SRS
    gate raised a raw pydantic traceback when the draft lacked 'title'. Gate
    render/validate failures get the friendly three-part treatment instead."""
    from test_progress import CONTRACT_SRS, INTENT
    import copy

    from spec_generator.cli import _render_gate
    from spec_generator.errors import UserFacingError

    bad = copy.deepcopy(CONTRACT_SRS)
    for r in bad["functional"]:
        del r["title"]
    with pytest.raises(UserFacingError) as ei:
        _render_gate("srs", bad, dict(INTENT))
    rendered = ei.value.render()
    assert "does not pass the contract" in rendered
    assert "title" in rendered


def test_cli_resume_does_not_repay_completed_stages(tmp_path, monkeypatch):
    """The banner promises 'rerun the same command - it resumes'. With a
    checkpoint present, the CLI must invoke(None) - needs/srs are NOT
    re-paid - and the same gate re-asks."""
    from test_progress import INTENT

    from spec_generator.cli import _run_with_gates
    from spec_generator.graph import build_graph
    from spec_generator.progress import ProgressBus
    from spec_generator.telemetry import UsageLog

    g, box = _review_graph(monkeypatch)
    cfg = {"configurable": {"thread_id": "resume1"}}
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, cfg)
    assert _interrupt_payload(g, cfg)["gate"] == "srs"
    assert "needs" not in box and box["srs"].calls == 1

    monkeypatch.setattr("builtins.input", lambda prompt="": "a")
    # the CLI's resume decision (checkpoint found -> resume=True) must not
    # re-pay completed stages; the gate re-asks from the same checkpoint
    result = _run_with_gates(g, cfg, dict(INTENT), tmp_path, resume=True)
    assert "bundle" in result
    assert "needs" not in box, "needs are deterministic and never paid"
    assert box["srs"].calls == 1, "srs must NOT be re-paid on resume"
    # micro-pipeline: plan + 1 entry + component plan + 1 body + ADR plan + 1 body
    assert box["architecture"].calls == 6  # ran once (as 6 slot calls) after the accepted gate


def test_gate_render_failure_on_stale_checkpoint_restarts_once(tmp_path, monkeypatch):
    """A checkpoint written before stage-level contract validation can hold a
    poisoned draft (his live run): the gate render fails friendly, the run
    restarts ONCE so the stage regenerates under the fixed schema, and the
    regenerated draft reaches the gate."""
    from test_progress import INTENT

    from spec_generator.cli import _run_with_gates
    from spec_generator.errors import UserFacingError

    g, box = _review_graph(monkeypatch)
    cfg = {"configurable": {"thread_id": "stale1"}}
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, cfg)
    assert _interrupt_payload(g, cfg)["gate"] == "srs"

    real_render = __import__("spec_generator.cli", fromlist=["_render_gate"])._render_gate
    calls = {"n": 0}

    def poisoned_once(gate, artifact, intent):
        calls["n"] += 1
        if calls["n"] == 1:
            raise UserFacingError("the srs draft does not pass the contract schema.", "validation detail: x", "rerun")
        return real_render(gate, artifact, intent)

    import spec_generator.cli as cli_mod
    monkeypatch.setattr(cli_mod, "_render_gate", poisoned_once)
    answers = iter(["a", "a"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    result = _run_with_gates(g, cfg, dict(INTENT), tmp_path, resume=True)
    assert "bundle" in result
    assert calls["n"] >= 2  # failed once, rendered after restart
    assert box["srs"].calls == 2  # srs regenerated once by the restart


POISONED_SRS = {
    "purpose": "p", "scope_summary": "sc",
    "functional": [{"requirement_id": f"R-{i}", "statement": "s", "priority": "must",
                    "kind": "functional",
                    "acceptance": [{"criterion_id": f"AC-{i}", "statement": "a"}],
                    "need_ids": ["N-1"], "intent_fields": []} for i in range(1, 5)],
    "non_functional": [{"requirement_id": f"R-{i}", "statement": "s", "priority": "must",
                        "kind": "non_functional",
                        "acceptance": [{"criterion_id": f"AC-{i}", "statement": "a"}],
                        "need_ids": ["N-1"], "intent_fields": []} for i in range(5, 11)],
    "session_id": "s", "intent_model_ref": "s",
}  # his live checkpoint shape: 10 requirements (4F+6NF), every title stripped by the old grammar bug


def test_heal_restart_end_to_end_sqlite_skeleton_mode(tmp_path, monkeypatch, capsys):
    """THE live path (his 2:58 paste): a pre-fix checkpoint holding a poisoned
    title-less SRS, resumed under --review with SqliteSaver. The gate render
    fails on the old draft ONCE, the heal restarts calmly (no Error block),
    the SRS stage regenerates in skeleton mode, and the gate then renders the
    merged, title-ful artifact. Proves the old-draft error text is the heal's
    first render, NOT a hole in the skeleton flow."""
    import sqlite3

    import pytest
    from langgraph.checkpoint.sqlite import SqliteSaver
    from test_progress import INTENT, FakeModel

    from spec_generator.cli import _run_with_gates
    from spec_generator.errors import UserFacingError

    box = {}
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: box.setdefault(stage, FakeModel(stage)))
    conn = sqlite3.connect(str(tmp_path / "checkpoints.sqlite"), check_same_thread=False)
    g = build_graph(UsageLog("s"), checkpointer=SqliteSaver(conn),
                    progress=ProgressBus(), review=True)
    cfg = {"configurable": {"thread_id": "heal-e2e"}}
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, cfg)
    assert _interrupt_payload(g, cfg)["gate"] == "srs"
    g.update_state(cfg, {"srs": json.loads(json.dumps(POISONED_SRS))})

    # the poisoned draft fails the gate with EXACTLY his pasted (capped) text
    from spec_generator.cli import _render_gate
    with pytest.raises(UserFacingError) as ei:
        _render_gate("srs", POISONED_SRS, dict(INTENT))
    detail = ei.value.cause
    assert detail.startswith("validation detail: functional.0.title: Field required; "
                             "functional.1.title: Field required")

    capsys.readouterr()  # reset capture
    answers = iter(["a", "a"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    srs_calls_before = box["srs"].calls
    result = _run_with_gates(g, cfg, dict(INTENT), tmp_path, resume=True)
    out = capsys.readouterr().err

    assert "bundle" in result
    assert box["srs"].calls == srs_calls_before + 1, "SRS regenerated exactly once by the heal"
    func = result["bundle"]["_srs"]["functional"]
    assert all(r.get("title") for r in func), "merged skeleton-mode body carries titles"
    assert "expected one-time heal" in out and "do NOT rerun" in out
    assert "Error: the srs draft" not in out, "the expected heal must not look like a failure"


def test_heal_second_failure_reraises_with_full_detail(tmp_path, monkeypatch):
    """If the REGENERATED draft also fails the gate render, the heal re-raises
    with the full validation detail (that would be a real bug, and it must be
    loud)."""
    import pytest
    from test_progress import INTENT

    from spec_generator.cli import _run_with_gates
    from spec_generator.errors import UserFacingError

    g, box = _review_graph(monkeypatch)
    cfg = {"configurable": {"thread_id": "heal-twice"}}
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, cfg)
    g.update_state(cfg, {"srs": json.loads(json.dumps(POISONED_SRS))})

    import spec_generator.cli as cli_mod
    monkeypatch.setattr(cli_mod, "_render_gate", cli_mod._render_gate)  # real render: fails every time on the poison
    # poison survives the restart: make the srs stage re-emit the same title-less body
    monkeypatch.setattr("spec_generator.graph.merge_srs_fills",
                        lambda slots, body: json.loads(json.dumps(POISONED_SRS)))
    with pytest.raises(UserFacingError) as ei:
        _run_with_gates(g, cfg, dict(INTENT), tmp_path, resume=True)
    assert "functional.0.title: Field required" in ei.value.render()


def test_expert_opinion_gate_records_simulated_not_founder(monkeypatch):
    import json
    from langchain_core.messages import AIMessage
    from spec_generator import model_config
    from spec_generator.graph import build_graph
    from spec_generator.telemetry import UsageLog
    from test_progress import FakeModel, INTENT, PAYLOADS
    from conftest import SYSTEM_BASELINE
    class ExpertModel(FakeModel):
        def __init__(self,stage,schema):super().__init__(stage);self.schema=schema
        def stream(self,messages):
            if self.stage=='expert_opinion':
                yield AIMessage(content='Here is my review.\nAPPROVE\nRATIONALE: artifact is coherent for test execution');return
            yield from super().stream(messages)
    monkeypatch.setattr(model_config,'build_chat_model',lambda stage,**k: ExpertModel(stage,k.get('schema') or {}))
    g=build_graph(UsageLog('s'),review=True,expert_opinion=True)
    result=g.invoke({'intent_model':dict(INTENT),'system_baseline':SYSTEM_BASELINE},{'configurable':{'thread_id':'expert'}})
    opinions=result['expert_opinions']
    assert [o['gate'] for o in opinions]==['srs','architecture']
    assert all(o['mode']=='simulated_expert_opinion' and o['human_review_required'] for o in opinions)
    assert all('founder' not in o['mode'] for o in opinions)
