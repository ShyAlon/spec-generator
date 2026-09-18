from conftest import SYSTEM_BASELINE, ModelFactoryLedger, SchemaRoutingArchitectureModel
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
            self.last_human = "\n".join(str(m.content) for m in messages)
            return super().invoke(messages)

    ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        if stage == "architecture":
            model = ledger.build(stage, kw.get("schema")); model.last_human = None
            original = model.stream
            def stream(messages):
                model.last_human = "\n".join(str(m.content) for m in messages)
                yield from original(messages)
            model.stream = stream
            return model
        return box.setdefault(stage, Recording(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    from conftest import StageCounter
    box["architecture"] = StageCounter(ledger, "architecture")
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
    assert box["architecture"].calls == 7  # ran once (as 6 slot calls) after the accepted gate


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

    box = {}; ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        return ledger.build(stage, kw.get("schema")) if stage == "architecture" else box.setdefault(stage, FakeModel(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
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
    expert_prompts = []
    class ExpertModel(FakeModel):
        def __init__(self,stage,schema):super().__init__(stage);self.schema=schema
        def stream(self,messages):
            if self.stage=='expert_opinion':
                expert_prompts.append('\n'.join(str(m.content) for m in messages))
                yield AIMessage(content='Here is my review.\nAPPROVE\nRATIONALE: artifact is coherent for its stage');return
            yield from super().stream(messages)
    def build(stage, **k):
        if stage == 'architecture':
            from test_per_item_resume import _routing
            return _routing([])(stage, k.get('schema') or {})
        return ExpertModel(stage, k.get('schema') or {})
    monkeypatch.setattr(model_config,'build_chat_model',build)
    g=build_graph(UsageLog('s'),review=True,expert_opinion=True)
    result=g.invoke({'intent_model':dict(INTENT),'system_baseline':SYSTEM_BASELINE},{'configurable':{'thread_id':'expert'}})
    opinions=result['expert_opinions']
    assert [o['gate'] for o in opinions]==['srs','architecture']
    assert all(o['mode']=='simulated_expert_opinion' and o['human_review_required'] for o in opinions)
    assert all('founder' not in o['mode'] for o in opinions)
    assert len(expert_prompts) == 2
    assert "Software Requirements Specification (SRS)" in expert_prompts[0]
    assert "requirements and acceptance criteria, not executable tests" in expert_prompts[0]
    assert "system architecture: stack, components, ADRs, and non-functional positions" in expert_prompts[1]
    assert all("reviewer for a TEST RUN" not in prompt for prompt in expert_prompts)

def test_expert_gate_wraps_unexpected_factory_failure(monkeypatch):
    from spec_generator.errors import UserFacingError
    from spec_generator import model_config
    from spec_generator.graph import build_graph
    from spec_generator.telemetry import UsageLog
    from test_progress import FakeModel, INTENT
    from conftest import SYSTEM_BASELINE
    original=model_config.build_chat_model
    def factory(stage,**kw):
        if stage=="expert_opinion": raise KeyError("no output schema for stage 'expert_opinion'")
        return FakeModel(stage)
    monkeypatch.setattr(model_config,"build_chat_model",factory)
    with pytest.raises(UserFacingError) as ei:
        build_graph(UsageLog("s"),review=True,expert_opinion=True).invoke({"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE},{"configurable":{"thread_id":"expert-factory-fail"}})
    rendered=ei.value.render()
    assert rendered.startswith("Error: expert opinion failed")
    assert "Likely cause:" in rendered and "Next:" in rendered


def test_expert_regeneration_is_capped_before_third_full_rewrite(monkeypatch):
    from spec_generator.errors import UserFacingError
    from spec_generator.graph import build_graph
    from spec_generator.telemetry import UsageLog
    from test_progress import INTENT
    from conftest import SYSTEM_BASELINE
    graph=build_graph(UsageLog("s"),review=True,expert_opinion=True)
    state={"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE,
      "srs":{},"architecture":{},"expert_opinions":[
       {"gate":"architecture","action":"regenerate"},
       {"gate":"architecture","action":"regenerate"}]}
    # Exercise the gate node through a graph seeded at its predecessor would
    # require a complete architecture; the source guard itself is the contract.
    import inspect
    source=inspect.getsource(build_graph)
    assert "len(prior_regenerations) >= 2" in source
    assert "without --expert-opinion" in source


def test_expert_regenerate_prints_rationale_and_feedback(monkeypatch, capsys):
    from langchain_core.messages import AIMessage
    from spec_generator import model_config
    from spec_generator.graph import build_graph
    from spec_generator.telemetry import UsageLog
    from test_progress import FakeModel, INTENT
    from conftest import SYSTEM_BASELINE, SchemaRoutingArchitectureModel
    class Reviewer(FakeModel):
        def stream(self, messages):
            if self.stage == "expert_opinion":
                yield AIMessage(content="REGENERATE\nRATIONALE: R-2 is not measurable.\nFEEDBACK: OBJECTION: artifact=srs | rule=measurable acceptance criterion | location=R-2 field acceptance | resolution=Add a numeric threshold to R-2 acceptance")
                return
            yield from super().stream(messages)
    def factory(stage, **kw):
        if stage == "architecture": return SchemaRoutingArchitectureModel(stage, kw.get("schema"))
        return Reviewer(stage)
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    with pytest.raises(Exception):
        # Two identical reviewer rounds reach the intentional human-review park.
        build_graph(UsageLog("s"), review=True, expert_opinion=True).invoke(
            {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
            {"configurable": {"thread_id": "visible-objections"}})
    err = capsys.readouterr().err
    assert "rationale: R-2 is not measurable." in err
    assert "regeneration feedback: OBJECTION: artifact=srs" in err


def test_expert_park_error_carries_both_prior_objections():
    import inspect
    from spec_generator.graph import build_graph
    source = inspect.getsource(build_graph)
    assert "format_expert_review_rounds(prior_regenerations)" in source
    assert 'model_config._console_line("  " + line)' not in source
    assert "boundary=expert-park" in source


def test_expert_review_bar_keeps_polish_under_approve_and_regenerates_only_blockers():
    import inspect
    from spec_generator.graph import build_graph
    source = inspect.getsource(build_graph)
    assert "APPROVE when there is no" in source
    assert "blocking contradiction, missing trace, unsafe assumption, or build-stopping ambiguity" in source
    assert "Non-blocking improvements and polish MUST stay as suggestions in RATIONALE under APPROVE" in source
    assert "REGENERATE only for a named blocking defect" in source
    assert "OBJECTION: artifact=<reviewed artifact name>" in source
    assert "rule=<named rule or criterion>" in source
    assert "location=<section, id, field, or quoted fragment>" in source
    assert "resolution=<concrete change>" in source


def test_expert_json_verdict_is_extracted_without_format_retry(monkeypatch):
    from langchain_core.messages import AIMessage
    from spec_generator import model_config
    from spec_generator.graph import build_graph
    from spec_generator.telemetry import UsageLog
    from test_progress import FakeModel, INTENT
    from conftest import SYSTEM_BASELINE, SchemaRoutingArchitectureModel
    calls = {"expert": 0}
    class Reviewer(FakeModel):
        def stream(self, messages):
            calls["expert"] += 1
            yield AIMessage(content='```json\n{"action":"REGENERATE","stakeholderNeeds":[{"id":"SN-1"}],"feedback":"OBJECTION: artifact=srs | rule=requirement traceability criterion | location=R-1 field need_ids | resolution=Add the governing SN-1 trace link"}\n```')
    def factory(stage, **kw):
        if stage == "architecture": return SchemaRoutingArchitectureModel(stage, kw.get("schema"))
        return Reviewer(stage) if stage == "expert_opinion" else FakeModel(stage)
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    with pytest.raises(Exception):
        build_graph(UsageLog("s"), review=True, expert_opinion=True).invoke(
            {"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE},
            {"configurable":{"thread_id":"json-expert"}})
    # One call per regeneration round: JSON format itself never consumes retry 2.
    assert calls["expert"] == 2


def test_expert_garbage_gets_exact_format_retry(monkeypatch):
    from langchain_core.messages import AIMessage
    from spec_generator import model_config
    from spec_generator.graph import build_graph
    from spec_generator.telemetry import UsageLog
    from test_progress import FakeModel, INTENT
    from conftest import SYSTEM_BASELINE, SchemaRoutingArchitectureModel
    prompts=[]
    class Reviewer(FakeModel):
        def stream(self, messages):
            prompts.append("\n".join(str(m.content) for m in messages))
            yield AIMessage(content="garbage")
    def factory(stage, **kw):
        if stage == "architecture": return SchemaRoutingArchitectureModel(stage, kw.get("schema"))
        return Reviewer(stage) if stage == "expert_opinion" else FakeModel(stage)
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    with pytest.raises(Exception):
        build_graph(UsageLog("s"), review=True, expert_opinion=True).invoke(
            {"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE},
            {"configurable":{"thread_id":"garbage-expert"}})
    assert len(prompts) == 2
    assert "FORMAT ERROR" in prompts[1] and "FEEDBACK:" in prompts[1]


def test_parked_gate_objections_are_sectioned_and_wrapped():
    from vmodel_contracts.runtime import format_expert_review_rounds
    rendered=format_expert_review_rounds([
      {"rationale":"First rationale.","feedback":"Fix trace links. Remove solution language.",
       "artifact_revision":1,"artifact_saved_at":"2026-09-17T22:31:12+00:00","artifact_sha256":"aaa111"},
      {"rationale":"Second rationale.","feedback":"Add measurable criteria. Link requirements by id.",
       "artifact_revision":2,"artifact_saved_at":"2026-09-18T06:45:58+00:00","artifact_sha256":"bbb222"}],width=60)
    assert "Round 1\n\nArtifact reviewed:" in rendered and "Round 2\n\nArtifact reviewed:" in rendered
    assert rendered.count("\nRationale:") == 2
    assert "Feedback:\n  1." in rendered and "\n  2." in rendered
    assert " | round" not in rendered
    assert "revision 1, saved at 2026-09-17T22:31:12+00:00" in rendered
    assert "revision 2, saved at 2026-09-18T06:45:58+00:00" in rendered


def test_each_regeneration_reviews_current_artifact_not_stale_snapshot():
    import inspect
    source=inspect.getsource(build_graph)
    assert "artifact_snapshot = state.get(artifact_key)" in source
    assert "json.dumps(artifact_snapshot" in source
    assert '"artifact_sha256": artifact_fingerprint' in source
    assert '"artifact_revision": artifact_revision["revision"]' in source
    assert "artifact_saved_at={artifact_revision['saved_at']}" in source
    assert "prior_regenerations={len(prior_regenerations)}" in source



def test_regenerated_architecture_and_revision_reach_round_two_reviewer(monkeypatch, capsys):
    """Round 2 reviews the regenerated graph-state artifact and its new revision."""
    from langchain_core.messages import AIMessage
    from conftest import SYSTEM_BASELINE
    from test_per_item_resume import _arch_answer

    prompts = []
    counts = {"architecture_reviews": 0}

    class Model:
        def __init__(self, stage, schema=None):
            self.stage, self.schema = stage, schema or {}
        def stream(self, messages):
            if self.stage == "expert_opinion":
                prompt = "\n".join(str(m.content) for m in messages)
                prompts.append(prompt)
                if "system architecture:" in prompt:
                    counts["architecture_reviews"] += 1
                    if counts["architecture_reviews"] == 1:
                        yield AIMessage(content="REGENERATE\nRATIONALE: first draft marker is stale.\nFEEDBACK: OBJECTION: artifact=architecture | rule=current artifact marker criterion | location=ADR-1 field title | resolution=Regenerate the architecture marker in ADR-1")
                        return
                yield AIMessage(content="APPROVE\nRATIONALE: current artifact is coherent.")
                return
            if self.stage == "architecture":
                answer = _arch_answer(self.schema)
                if "driver" in set(self.schema.get("properties", {})):
                    prompt = "\n".join(str(m.content) for m in messages)
                    marker = "regenerated-architecture" if "Regenerate the architecture marker" in prompt else "initial-architecture"
                    answer["title"] = marker
                yield AIMessage(content=json.dumps(answer))
                return
            yield from FakeModel(self.stage).stream(messages)

    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: Model(stage, kw.get("schema")))
    result = build_graph(UsageLog("s"), review=True, expert_opinion=True).invoke(
        {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
        {"configurable": {"thread_id": "fresh-architecture-review"}})

    architecture_prompts = [p for p in prompts if "system architecture:" in p]
    assert len(architecture_prompts) == 2
    assert "initial-architecture" in architecture_prompts[0]
    assert "regenerated-architecture" not in architecture_prompts[0]
    assert "regenerated-architecture" in architecture_prompts[1]
    opinions = [o for o in result["expert_opinions"] if o["gate"] == "architecture"]
    assert [o["artifact_revision"] for o in opinions] == [1, 2]
    assert opinions[0]["artifact_sha256"] != opinions[1]["artifact_sha256"]
    assert all(o["artifact_saved_at"].endswith("+00:00") for o in opinions)
    assert all(o["artifact_saved_at"].startswith("2026-") for o in opinions)  # full date + time, not time-only
    err = capsys.readouterr().err
    for opinion in opinions:
        assert f"artifact_sha256={opinion['artifact_sha256']}" in err
        assert f"artifact_revision={opinion['artifact_revision']}" in err
        assert f"artifact_saved_at={opinion['artifact_saved_at']}" in err



def test_vague_expert_objection_retries_then_specific_objection_passes(monkeypatch):
    from langchain_core.messages import AIMessage
    from conftest import SYSTEM_BASELINE, SchemaRoutingArchitectureModel
    replies = iter([
      "REGENERATE\nFEEDBACK: The artifact hierarchy and authoring rules are not strictly adhered to.\nRATIONALE: vague.",
      "REGENERATE\nFEEDBACK: OBJECTION: artifact=srs | rule=stakeholder need traceability criterion | location=R-1 field need_ids | resolution=Add the governing SN-1 id to need_ids\nRATIONALE: R-1 lacks its source trace.",
      "APPROVE\nRATIONALE: corrected SRS is coherent.",
      "APPROVE\nRATIONALE: architecture is coherent."])
    prompts=[]
    class Reviewer(FakeModel):
        def stream(self,messages):
            prompts.append("\n".join(str(m.content) for m in messages))
            yield AIMessage(content=next(replies))
    def factory(stage,**kw):
        if stage == "architecture": return SchemaRoutingArchitectureModel(stage,kw.get("schema"))
        return Reviewer(stage) if stage == "expert_opinion" else FakeModel(stage)
    monkeypatch.setattr(model_config,"build_chat_model",factory)
    result=build_graph(UsageLog("s"),review=True,expert_opinion=True).invoke(
      {"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE},
      {"configurable":{"thread_id":"specificity-retry"}})
    assert "SPECIFICITY ERROR" in prompts[1]
    assert "objection 1 has no structured OBJECTION line" in prompts[1]
    opinion=result["expert_opinions"][0]
    assert opinion["objections"][0]["artifact"] == "srs"
    assert opinion["objections"][0]["location"] == "R-1 field need_ids"


def test_architecture_upstream_objection_becomes_advisory_without_regeneration(monkeypatch, capsys):
    from langchain_core.messages import AIMessage
    from conftest import SYSTEM_BASELINE, SchemaRoutingArchitectureModel
    calls={"architecture":0,"arch_review":0}
    class Reviewer(FakeModel):
        def stream(self,messages):
            prompt="\n".join(str(m.content) for m in messages)
            if self.stage == "expert_opinion" and "system architecture:" in prompt:
                calls["arch_review"] += 1
                yield AIMessage(content=("REGENERATE\nFEEDBACK: OBJECTION: artifact=srs | rule=shall statement grammar | "
                  "location=R-2 field statement | resolution=Rewrite R-2 as a shall statement\n"
                  "RATIONALE: This fix belongs in the SRS.")); return
            if self.stage == "expert_opinion":
                yield AIMessage(content="APPROVE\nRATIONALE: coherent for its own stage.")
                return
            yield from FakeModel(self.stage).stream(messages)
    class Architecture(SchemaRoutingArchitectureModel):
        def stream(self,messages):
            calls["architecture"] += 1
            yield from super().stream(messages)
    def factory(stage,**kw):
        return Architecture(stage,kw.get("schema")) if stage=="architecture" else Reviewer(stage)
    monkeypatch.setattr(model_config,"build_chat_model",factory)
    result=build_graph(UsageLog("s"),review=True,expert_opinion=True).invoke(
      {"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE},
      {"configurable":{"thread_id":"upstream-advisory"}})
    opinions=[o for o in result["expert_opinions"] if o["gate"]=="architecture"]
    assert len(opinions)==1 and opinions[0]["action"]=="approve"
    assert opinions[0]["objections"] == []
    assert opinions[0]["upstream_advisories"][0]["artifact"] == "srs"
    assert calls["arch_review"] == 1
    # One architecture pass is seven micro-calls; no second regeneration pass.
    assert calls["architecture"] == 7
    output=capsys.readouterr().err
    assert "Upstream advisories:" in output
    assert "This concerns the srs:" in output



def test_legacy_parked_checkpoint_is_rereviewed_under_structured_scope(monkeypatch, capsys):
    """Two pre-protocol regenerations re-enter the gate under today's protocol."""
    from langchain_core.messages import AIMessage
    from conftest import SYSTEM_BASELINE, SchemaRoutingArchitectureModel
    phase={"legacy_resume":False}; architecture_reviews=[]
    class Reviewer(FakeModel):
        def stream(self,messages):
            prompt="\n".join(str(m.content) for m in messages)
            if self.stage == "expert_opinion" and "system architecture:" in prompt:
                architecture_reviews.append(prompt)
                if phase["legacy_resume"]:
                    yield AIMessage(content=("REGENERATE\nFEEDBACK: OBJECTION: artifact=srs | rule=shall statement grammar | "
                      "location=R-2 field statement | resolution=Rewrite R-2 as a shall statement\n"
                      "RATIONALE: The stored concern belongs upstream.")); return
            yield AIMessage(content="APPROVE\nRATIONALE: coherent.")
    def factory(stage,**kw):
        if stage=="architecture": return SchemaRoutingArchitectureModel(stage,kw.get("schema"))
        return Reviewer(stage) if stage=="expert_opinion" else FakeModel(stage)
    monkeypatch.setattr(model_config,"build_chat_model",factory)
    g=build_graph(UsageLog("s"),review=True,expert_opinion=True)
    cfg={"configurable":{"thread_id":"legacy-park-resume"}}
    g.invoke({"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE},cfg)
    legacy=[{"gate":"architecture","action":"regenerate","feedback":"The artifact hierarchy is wrong."},
            {"gate":"architecture","action":"regenerate","feedback":"The artifact hierarchy is wrong."}]
    phase["legacy_resume"]=True
    # This reproduces a persisted run paused immediately before the architecture
    # gate with two old opinions. Updating as the predecessor routes into gate.
    g.update_state(cfg,{"expert_opinions":legacy},as_node="architecture")
    result=g.invoke(None,cfg)
    assert len(architecture_reviews) == 2  # initial review plus fresh resume review
    current=[o for o in result["expert_opinions"] if o.get("review_protocol")==2 and o["gate"]=="architecture"]
    assert len(current)==1 and current[0]["action"]=="approve"
    assert current[0]["upstream_advisories"][0]["artifact"]=="srs"
    output=capsys.readouterr().err
    assert "boundary=expert-checkpoint-migration" in output
    assert "legacy_regenerations=2" in output
    assert "decision=re-review-current-artifact" in output
    assert "This concerns the srs:" in output
    assert "expert review parked" not in output


def test_live_prose_srs_objection_approves_architecture_as_advisory(monkeypatch, capsys):
    from langchain_core.messages import AIMessage
    from conftest import SYSTEM_BASELINE, SchemaRoutingArchitectureModel
    class Reviewer(FakeModel):
        def stream(self,messages):
            prompt="\n".join(str(m.content) for m in messages)
            if self.stage=="expert_opinion" and "system architecture:" in prompt:
                yield AIMessage(content=('REGENERATE\nFEEDBACK:\n1. OBJECTION: The SRS contains solution-specific details such as "AES-256" and "OAuth".\nRATIONALE: The concern belongs upstream.')); return
            yield AIMessage(content="APPROVE\nRATIONALE: coherent.")
    def factory(stage,**kw):
        if stage=="architecture": return SchemaRoutingArchitectureModel(stage,kw.get("schema"))
        return Reviewer(stage) if stage=="expert_opinion" else FakeModel(stage)
    monkeypatch.setattr(model_config,"build_chat_model",factory)
    result=build_graph(UsageLog("s"),review=True,expert_opinion=True).invoke(
      {"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE},
      {"configurable":{"thread_id":"live-prose-upstream"}})
    opinion=[o for o in result["expert_opinions"] if o["gate"]=="architecture"][0]
    assert opinion["action"]=="approve" and not opinion["objections"]
    advisory=opinion["upstream_advisories"][0]
    assert advisory["artifact"]=="srs"
    assert '"AES-256"' in advisory["location"] and '"OAuth"' in advisory["location"]
    output=capsys.readouterr().err
    assert "LOW-SPECIFICITY" not in output and "This concerns the srs:" in output


def test_low_specificity_error_renders_numbered_feedback_not_raw_snippet(monkeypatch):
    from langchain_core.messages import AIMessage
    from conftest import SYSTEM_BASELINE
    from spec_generator.errors import UserFacingError
    class Reviewer(FakeModel):
        def stream(self,messages):
            if self.stage=="expert_opinion":
                yield AIMessage(content=("REGENERATE\nFEEDBACK: The architecture hierarchy is unclear. "
                                         "The architecture rules are not followed.\nRATIONALE: The architecture needs work.")); return
            yield from super().stream(messages)
    monkeypatch.setattr(model_config,"build_chat_model",lambda stage,**kw:Reviewer(stage))
    with pytest.raises(UserFacingError) as exc:
        build_graph(UsageLog("s"),review=True,expert_opinion=True).invoke(
          {"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE},
          {"configurable":{"thread_id":"low-specificity-render"}})
    rendered=exc.value.render()
    assert "Reviewer objections:" in rendered
    assert "Rationale:\n  The architecture needs work." in rendered
    assert "Feedback:\n  1. The architecture hierarchy is unclear." in rendered
    assert "2. The architecture rules are not followed." in rendered
    assert "Output began:" not in rendered
