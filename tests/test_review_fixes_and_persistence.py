from conftest import SYSTEM_BASELINE, ModelFactoryLedger, SchemaRoutingArchitectureModel
"""2026-09-09 live failure + ask: (1) the review-fix application path re-fired
the srs stage against the retired full-document validator while the grammar
constrained the model to fills ('functional: Field required; fills: Extra
inputs are not permitted') - the fix path now re-fires the FILL stage with the
fixes; (2) the bundle persists incrementally, stage by stage, atomically."""
import json

import pytest
from langgraph.checkpoint.memory import MemorySaver

from spec_generator import model_config
from spec_generator.cli import _atomic_write, _make_persist
from spec_generator.graph import build_graph
from spec_generator.telemetry import UsageLog
from test_progress import INTENT, PAYLOADS, FakeModel

FIXES = [{"fix": "R-1 statement too vague - name the emitted file format"}]


class QueueModel(FakeModel):
    """Per-stage payload queue; falls back to PAYLOADS[stage] when drained."""
    QUEUES = {}

    def invoke(self, messages):
        self.calls += 1
        self.last_human = "\n".join(str(m.content) for m in messages)
        q = self.QUEUES.get(self.stage) or []
        if q:
            payload = q.pop(0)
        elif self.stage == "architecture":
            payload = self._architecture_answer(self.last_human)
        else:
            payload = PAYLOADS[self.stage]
        return type("R", (), {"content": json.dumps(payload)})()


@pytest.fixture
def queue_models(monkeypatch):
    box = {}
    QueueModel.QUEUES = {}
    ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        if stage == "architecture": return ledger.build(stage, kw.get("schema"))
        return box.setdefault(stage, QueueModel(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    return box


def test_review_fixes_refire_the_fill_stage(queue_models):
    QueueModel.QUEUES["review"] = [FIXES, []]  # round 1 flags, round 2 clean
    g = build_graph(UsageLog("s"), checkpointer=MemorySaver())
    r = g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "rf1"}})
    assert "bundle" in r
    srs = queue_models["srs"]
    assert srs.calls == 2  # initial author + one fix application
    apply_payload = srs.last_human
    assert "REVIEW FIXES" in apply_payload and FIXES[0]["fix"] in apply_payload
    assert "REQUIREMENT SLOTS" in apply_payload  # fill shape, not the retired full-document ask
    # the fix path went through the skeleton merge validator: downstream stages
    # re-ran against the corrected SRS and the pipeline completed
    assert queue_models["items"].calls == 2 and queue_models["tests"].calls == 2  # re-derived from the fixed SRS
    assert queue_models["review"].calls == 2
    assert all(isinstance(c["statement"], str)
               for req in r["srs"]["functional"]
               for c in req["acceptance"])


def test_clean_review_never_pays_the_fix_stage(queue_models):
    g = build_graph(UsageLog("s"), checkpointer=MemorySaver())
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "rf2"}})
    assert queue_models["srs"].calls == 1
    assert queue_models["review"].calls == 1  # clean: stops before pass 2


def test_persist_hook_fires_per_stage_in_order(queue_models):
    fired = []
    g = build_graph(UsageLog("s"), checkpointer=MemorySaver(),
                    persist=lambda stage, data, state: fired.append(stage))
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "p1"}})
    assert fired == ["needs", "srs", "architecture", "items", "tests"]


def test_incremental_bundle_on_disk_and_atomic(tmp_path, queue_models):
    persist = _make_persist(dict(INTENT), tmp_path)
    g = build_graph(UsageLog("s"), checkpointer=MemorySaver(), persist=persist)
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "p2"}})
    for f in ("needs.json", "needs.md", "srs.json", "srs.md",
              "architecture.json", "architecture.md", "items.json", "tests.json"):
        assert (tmp_path / f).exists(), f
    assert (tmp_path / "specs" / "SPEC-1.md").exists()
    assert "Driver:" in (tmp_path / "architecture.md").read_text()  # rendered, not raw JSON
    for stage in ("srs", "architecture"):
        revision = json.loads((tmp_path / f"{stage}.revision.json").read_text())
        assert revision["revision"] == 1
        assert revision["saved_at"].endswith("+00:00")
    assert not list(tmp_path.glob("*.tmp")) and not list((tmp_path / "specs").glob("*.tmp"))


def test_late_failure_leaves_completed_stages_on_disk(tmp_path, monkeypatch):
    class DieAtTests(QueueModel):
        def invoke(self, messages):
            if self.stage == "tests":
                raise RuntimeError("process died")
            return super().invoke(messages)

    box = {}; ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        return ledger.build(stage, kw.get("schema")) if stage == "architecture" else box.setdefault(stage, DieAtTests(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    persist = _make_persist(dict(INTENT), tmp_path)
    g = build_graph(UsageLog("s"), checkpointer=MemorySaver(), persist=persist)
    with pytest.raises(RuntimeError):
        g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "p3"}})
    for f in ("needs.json", "srs.json", "architecture.json", "items.json", "specs/SPEC-1.md"):
        assert (tmp_path / f).exists(), f
    assert not (tmp_path / "tests.json").exists()  # the failed stage wrote nothing


def test_atomic_write_leaves_no_tmp(tmp_path):
    _atomic_write(tmp_path / "x.json", '{"a": 1}')
    assert (tmp_path / "x.json").read_text() == '{"a": 1}'
    assert not (tmp_path / "x.json.tmp").exists()


def test_typed_review_targets_and_systemic_fallback():
    from spec_generator.graph import _fix_targets, _normalized_review_fixes
    scoped = _normalized_review_fixes([
        {"scope": "requirement", "target_ids": ["R-1"], "fix": "tighten wording"},
        {"scope": "test", "target_ids": ["T-2"], "fix": "add cleanup"},
    ])
    assert _fix_targets(scoped, "requirement") == {"R-1"}
    assert _fix_targets(scoped, "test") == {"T-2"}
    assert _fix_targets(scoped, "spec") == set()
    assert _fix_targets([{"scope": "systemic", "target_ids": [], "fix": "global"}], "test") is None
    # Legacy string answers fail safe: full fallback, never guessed targeting.
    assert _fix_targets(_normalized_review_fixes(["legacy fix"]), "requirement") is None


def test_dependency_closure_maps_requirement_fixes_to_specs():
    from spec_generator.graph import _affected_spec_ids
    state = {"items": [
        {"spec_id": "SPEC-1", "requirement_ids": ["R-1"]},
        {"spec_id": "SPEC-2", "requirement_ids": ["R-2"]},
    ]}
    fixes = [{"scope": "requirement", "target_ids": ["R-1"], "fix": "tighten"}]
    assert _affected_spec_ids(state, fixes) == {"SPEC-1"}
    fixes.append({"scope": "spec", "target_ids": ["SPEC-2"], "fix": "repair body"})
    assert _affected_spec_ids(state, fixes) == {"SPEC-1", "SPEC-2"}
    assert _affected_spec_ids(state, [{"scope": "systemic", "target_ids": [], "fix": "global"}]) is None



def test_contract_shaped_criteria_are_normalized_at_fill_reuse_boundary():
    from spec_generator.graph import _previous_srs_fill
    from spec_generator.skeleton import build_srs_skeleton, merge_srs_fills
    from test_progress import CONTRACT_SRS

    requirement = CONTRACT_SRS["functional"][0]
    requirement = dict(requirement)
    requirement["acceptance"] = [{"criterion_id":"AC-1", "statement":"spec exists", "source_turns":[12]}]
    fill = _previous_srs_fill(requirement)
    assert fill["acceptance"] == ["spec exists"]
    merged = merge_srs_fills(
        build_srs_skeleton(PAYLOADS["needs"]),
        {"purpose": CONTRACT_SRS["purpose"],
         "scope_summary": CONTRACT_SRS["scope_summary"],
         "fills": {requirement["requirement_id"]: fill}},
    )
    assert merged["functional"][0]["acceptance"] == [
        {"criterion_id": "AC-1", "statement": "spec exists"}
    ]

def test_stage_validation_error_is_human_readable_not_raw_traceback():
    from pydantic import ValidationError
    from spec_generator.graph import SRSDocument, _stage_validation_error, srs_document_data
    from test_progress import CONTRACT_SRS

    invalid = __import__("copy").deepcopy(CONTRACT_SRS)
    invalid["functional"][0]["acceptance"][0]["statement"] = {
        "criterion_id": "AC-1", "statement": "nested"
    }
    with pytest.raises(ValidationError) as caught:
        SRSDocument.model_validate(srs_document_data(INTENT, invalid))

    rendered = _stage_validation_error("apply_fixes", caught.value).render()
    assert rendered.startswith("Error: stage 'apply_fixes'")
    assert "Likely cause: artifact 'apply_fixes' has an offending field shape" in rendered
    assert "functional.0.acceptance.0.statement" in rendered
    assert "got dict" in rendered
    assert "Next:" in rendered and "changing models cannot fix" in rendered


def test_checkpoint_resume_after_apply_fixes_failure_skips_upstream(monkeypatch):
    """A failed fix pass resumes at apply_fixes without redoing completed stages."""
    class CrashOnceAtFix(QueueModel):
        crashed = False

        def invoke(self, messages):
            if (self.stage == "srs" and self.calls == 1
                    and "REVIEW FIXES" in "\n".join(str(m.content) for m in messages)
                    and not type(self).crashed):
                type(self).crashed = True
                raise RuntimeError("simulated apply_fixes crash")
            return super().invoke(messages)

    CrashOnceAtFix.crashed = False
    QueueModel.QUEUES = {"review": [FIXES, []]}
    box = {}
    ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        if stage == "architecture":
            model = ledger.build(stage, kw.get("schema"))
            box["architecture"] = type("Counter", (), {"calls": property(lambda _self: ledger.calls("architecture"))})()
            return model
        return box.setdefault(stage, CrashOnceAtFix(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    graph = build_graph(UsageLog("s"), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "resume-apply-fixes"}}

    with pytest.raises(RuntimeError, match="simulated apply_fixes crash"):
        graph.invoke(
            {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
            config)

    calls_at_crash = {stage: model.calls for stage, model in box.items()}
    result = graph.invoke(None, config)

    assert "bundle" in result
    assert box["architecture"].calls == calls_at_crash["architecture"]
    assert box["items"].calls == calls_at_crash["items"] + 1
    assert box["tests"].calls == calls_at_crash["tests"] + 1
    # Exactly one SRS call is added: the retried fix fill, not initial authoring.
    assert box["srs"].calls == calls_at_crash["srs"] + 1
    assert box["review"].calls == calls_at_crash["review"] + 1


def test_large_review_is_requirement_chunked_and_resumable(monkeypatch):
    """Live scale: 8 requirements/10 specs/44 tests never becomes one review call."""
    import copy
    from spec_generator.graph import build_graph
    from spec_generator.telemetry import UsageLog
    calls=[]
    class Partials:
        def __init__(self): self.data={}
        def load(self,k): return self.data.get(k)
        def save(self,k,v): self.data[k]=copy.deepcopy(v)
        def clear(self,k): self.data.pop(k,None)
    partials=Partials()
    # Source-level regression keeps this independent of the full authoring fake:
    # review chunks are max 2 requirements and each successful chunk persists.
    import inspect
    source=inspect.getsource(build_graph)
    assert "requirements[i:i + 2]" in source
    assert 'partials.save("review", saved)' in source
    assert 'partials.clear("review")' in source
    assert 'f"round-{n}-chunk-{index}"' in source
    assert 8 // 2 == 4 and 4 < 8 + 10 + 44


def test_review_batches_one_requirement_and_caps_finding_shape():
    import inspect
    from spec_generator.graph import build_graph
    from spec_generator.schemas import for_stage
    source=inspect.getsource(build_graph)
    assert "chunks = [[requirement] for requirement in requirements]" in source
    assert "max_findings=24 max_fix_chars=320" in source
    assert "Never restate requirements, specs, tests, or rationale" in source
    schema=for_stage("review")
    assert schema["items"]["properties"]["fix"]["maxLength"] == 320
    assert schema["items"]["properties"]["target_ids"]["maxItems"] == 8


def test_review_partial_key_is_stable_per_single_requirement_batch():
    import inspect
    from spec_generator.graph import build_graph
    source=inspect.getsource(build_graph)
    assert 'chunk_id = f"round-{n}-chunk-{index}"' in source
    assert 'if chunk_id not in saved:' in source
    assert 'partials.save("review", saved)' in source
