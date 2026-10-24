"""Resume hydration: settled state is admitted only from lineage-bound artifacts."""
import json
from argparse import Namespace
from pathlib import Path

import pytest

from spec_generator import lineage
from spec_generator.cli import _hydrate_from_lineage
from spec_generator.errors import UserFacingError


class FakeGraph:
    def __init__(self):
        self.updated = None
    def update_state(self, cfg, values, as_node=None):
        self.updated = (values, as_node)


def _args(tmp_path, to_stage, intent=True, conops=None, baseline=None):
    intent_path = tmp_path / "interview" / "intent_model.json"
    if intent:
        intent_path.parent.mkdir(parents=True, exist_ok=True)
        intent_path.write_text(json.dumps({"session_id": "s1"}))
    return Namespace(intent_model=intent_path, conops=conops, system_baseline=baseline,
                     to_stage=to_stage)


def _bind_artifact(root, phase, name, body, source="restore", settled=True):
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(body))
    lineage.bind(root, task_id="t1", session_id="s1", generation="7",
                 phase=phase, name=name, sha256=lineage.sha256_file(root / name),
                 source=source, settled=settled)


NEEDS = {"contract_version": "1", "session_id": "s1",
         "needs": [{"need_id": "N-1", "stakeholder": "founder", "statement": "run the business",
                    "kind": "functional", "intent_fields": ["goal"], "intent_need_ids": []}]}


def test_fresh_host_hydrates_settled_stages_without_regeneration(tmp_path, monkeypatch):
    monkeypatch.setenv("VMODEL_PROGRESS_TASK_ID", "t1")
    monkeypatch.setenv("VMODEL_PROGRESS_GENERATION", "7")
    lineage.record_inputs(tmp_path, task_id="t1", session_id="s1", generation="7",
                          input_hashes={"intent_model.json": lineage.sha256_file(_args(tmp_path, "srs").intent_model)})
    _bind_artifact(tmp_path, "needs", "needs.json", NEEDS)
    graph = FakeGraph()
    prior = _hydrate_from_lineage(graph, {"configurable": {"thread_id": "s1"}}, {},
                                  tmp_path, _args(tmp_path, "srs"), "s1", False)
    assert prior["needs"]["needs"][0]["need_id"] == "N-1", "needs state comes from the bound artifact"
    assert "contract_version" not in prior["needs"], "stamps never enter graph state"
    values, as_node = graph.updated
    assert as_node == "needs", "resume re-points at the stage after the hydrated one"


def test_missing_prerequisite_refuses_with_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("VMODEL_PROGRESS_TASK_ID", "t1")
    monkeypatch.setenv("VMODEL_PROGRESS_GENERATION", "7")
    # srs is bound as settled while its prerequisite needs is absent: the record is
    # inconsistent and resuming would regenerate settled downstream work.
    _bind_artifact(tmp_path, "srs", "srs.json", {"contract_version": "1", "session_id": "s1",
                                                 "requirements": []})
    with pytest.raises(UserFacingError, match="refused this resume"):
        _hydrate_from_lineage(FakeGraph(), {}, {}, tmp_path, _args(tmp_path, "architecture"), "s1", False)


def test_pending_stage_after_settled_ones_runs_normally(tmp_path, monkeypatch):
    monkeypatch.setenv("VMODEL_PROGRESS_TASK_ID", "t1")
    monkeypatch.setenv("VMODEL_PROGRESS_GENERATION", "7")
    # needs settled and bound; srs never ran. Pickup hydrates needs and lets srs run.
    _bind_artifact(tmp_path, "needs", "needs.json", NEEDS)
    graph = FakeGraph()
    prior = _hydrate_from_lineage(graph, {"configurable": {"thread_id": "s1"}}, {},
                                  tmp_path, _args(tmp_path, "architecture"), "s1", False)
    assert "srs" not in prior and prior["needs"]["needs"][0]["need_id"] == "N-1"


def test_changed_settled_bytes_refuse_even_when_checkpoint_covers(tmp_path, monkeypatch):
    """Restored authority outranks local checkpoint claims: a returning host with a
    stale checkpoint must not have substituted bytes blessed."""
    monkeypatch.setenv("VMODEL_PROGRESS_TASK_ID", "t1")
    monkeypatch.setenv("VMODEL_PROGRESS_GENERATION", "7")
    _bind_artifact(tmp_path, "needs", "needs.json", NEEDS)
    (tmp_path / "needs.json").write_text(json.dumps({"needs": [{"tampered": True}]}))
    prior = {"needs": {"needs": []}}  # checkpoint covers the stage; disk must STILL match
    with pytest.raises(UserFacingError, match="refused this resume"):
        _hydrate_from_lineage(FakeGraph(), {}, prior, tmp_path, _args(tmp_path, "srs"), "s1", False)


def test_authorized_rewrite_rebind_carries_settled_and_verifies(tmp_path, monkeypatch):
    """Same-host legitimate rewrite: the mutation point (assembly) re-binds the new
    bytes, carrying settledness forward; resume then verifies against the NEW binding.
    A rewrite WITHOUT a re-bind must still refuse."""
    monkeypatch.setenv("VMODEL_PROGRESS_TASK_ID", "t1")
    monkeypatch.setenv("VMODEL_PROGRESS_GENERATION", "7")
    _bind_artifact(tmp_path, "needs", "needs.json", NEEDS)
    rewritten = {**NEEDS, "needs": NEEDS["needs"] + [
        {"need_id": "N-2", "stakeholder": "ops", "statement": "keep the lights on",
         "kind": "functional", "intent_fields": ["goal"], "intent_need_ids": []}]}
    # unsanctioned rewrite (no re-bind): refuses even though bytes are contract-valid
    (tmp_path / "needs.json").write_text(json.dumps(rewritten))
    with pytest.raises(UserFacingError, match="refused this resume"):
        _hydrate_from_lineage(FakeGraph(), {}, {}, tmp_path, _args(tmp_path, "srs"), "s1", False)
    # authorized rewrite: the mutation point re-binds the new bytes
    lineage.bind(tmp_path, task_id="t1", session_id="s1", generation="7",
                 phase="needs", name="needs.json",
                 sha256=lineage.sha256_file(tmp_path / "needs.json"),
                 source="assembly", settled=True)
    graph = FakeGraph()
    prior = _hydrate_from_lineage(graph, {"configurable": {"thread_id": "s1"}}, {},
                                  tmp_path, _args(tmp_path, "srs"), "s1", False)
    assert [n["need_id"] for n in prior["needs"]["needs"]] == ["N-1", "N-2"]
    assert lineage.lookup(lineage.load(tmp_path), "needs", "needs.json")["settled"] is True


def test_input_hash_mismatch_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("VMODEL_PROGRESS_TASK_ID", "t1")
    monkeypatch.setenv("VMODEL_PROGRESS_GENERATION", "7")
    args = _args(tmp_path, "srs")
    lineage.record_inputs(tmp_path, task_id="t1", session_id="s1", generation="7",
                          input_hashes={"intent_model.json": "0" * 64})
    _bind_artifact(tmp_path, "needs", "needs.json", NEEDS)
    with pytest.raises(UserFacingError, match="refused this resume"):
        _hydrate_from_lineage(FakeGraph(), {}, {}, tmp_path, args, "s1", False)


def test_hosted_disk_artifacts_without_record_refuse(tmp_path, monkeypatch):
    monkeypatch.setenv("VMODEL_PROGRESS_TASK_ID", "t1")
    (tmp_path / "srs.json").write_text(json.dumps({"contract_version": "1"}))
    with pytest.raises(UserFacingError, match="cannot be tied to their approved record"):
        _hydrate_from_lineage(FakeGraph(), {}, {}, tmp_path, _args(tmp_path, "items"), "s1", False)


def test_standalone_without_record_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("VMODEL_PROGRESS_TASK_ID", raising=False)
    (tmp_path / "srs.json").write_text(json.dumps({"contract_version": "1"}))
    # no hosted env, no record: today's behavior - checkpoint is the sole authority
    assert _hydrate_from_lineage(FakeGraph(), {}, {}, tmp_path, _args(tmp_path, "items"), "s1", False) == {}


def test_companion_binding_mismatch_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("VMODEL_PROGRESS_TASK_ID", "t1")
    monkeypatch.setenv("VMODEL_PROGRESS_GENERATION", "7")
    _bind_artifact(tmp_path, "needs", "needs.json", NEEDS)
    _bind_artifact(tmp_path, "srs", "srs.semantic-review.json", {"decisions": []})
    (tmp_path / "srs.semantic-review.json").write_text(json.dumps({"decisions": ["swapped"]}))
    with pytest.raises(UserFacingError, match="refused this resume"):
        _hydrate_from_lineage(FakeGraph(), {}, {}, tmp_path, _args(tmp_path, "architecture"), "s1", False)


def _bound_file(tmp_path, phase, name, body, settled=True):
    (tmp_path / name).write_text(json.dumps(body))
    lineage.bind(tmp_path, task_id="t1", session_id="s1", generation="7",
                 phase=phase, name=name, sha256=lineage.sha256_file(tmp_path / name),
                 source="restore", settled=settled)
    return (tmp_path / name).read_bytes()


def test_assembly_rebind_admits_only_serialization_difference(tmp_path, monkeypatch):
    """The terminal render may re-serialize (contract stamping, model null defaults);
    the binding then tracks the new bytes with settledness carried forward."""
    from spec_generator.cli import _assembly_rebind_lineage, _stamped
    monkeypatch.setenv("VMODEL_PROGRESS_TASK_ID", "t1")
    monkeypatch.setenv("VMODEL_PROGRESS_GENERATION", "7")
    tests = [{"test_id": "T-1", "criterion_id": "AC-1", "level": "acceptance", "runner": "http",
              "description": "criterion holds", "spec_id": None, "components": ["api"],
              "definition": {"steps": ["GET /"], "expected_outcomes": ["HTTP 200"],
                             "preconditions": [], "cleanup": [], "cases": [],
                             "execution": {"method": "GET", "url": "/", "expected_status": 200}}}]
    prior = _bound_file(tmp_path, "tests", "tests.json", _stamped(tests))
    # the render drops the stamp and adds model null defaults: admitted
    rendered = [{**t, "definition": {**t["definition"],
                                     "execution": {"body_json": None, "command": None, "expected_exit": None,
                                                   "expected_status": 200, "expected_stdout_contains": None,
                                                   "method": "GET", "url": "/"}}} for t in tests]
    (tmp_path / "tests.json").write_text(json.dumps(rendered))
    _assembly_rebind_lineage(tmp_path, "s1", {"tests.json": prior})
    entry = lineage.lookup(lineage.load(tmp_path), "tests", "tests.json")
    assert entry["sha256"] == lineage.sha256_file(tmp_path / "tests.json")
    assert entry["settled"] is True, "settledness carries forward"
    assert entry["source"] == "assembly"


def test_assembly_rebind_refuses_substantive_drift(tmp_path, monkeypatch):
    """A render that changes semantic content - including dropping an approved
    founder decision - fails closed and leaves the settled binding untouched."""
    from spec_generator.cli import _assembly_rebind_lineage, _stamped
    monkeypatch.setenv("VMODEL_PROGRESS_TASK_ID", "t1")
    monkeypatch.setenv("VMODEL_PROGRESS_GENERATION", "7")
    tests = [{"test_id": "T-1", "criterion_id": "AC-1", "level": "acceptance", "runner": "http",
              "description": "criterion holds", "spec_id": None, "components": ["api"],
              "definition": {"steps": ["GET /"], "expected_outcomes": ["HTTP 200"],
                             "preconditions": [], "cleanup": [], "cases": [],
                             "execution": {"method": "GET", "url": "/", "expected_status": 200}}}]
    prior = _bound_file(tmp_path, "tests", "tests.json", _stamped(tests))
    settled_sha = lineage.lookup(lineage.load(tmp_path), "tests", "tests.json")["sha256"]
    drifted = [dict(tests[0], description="changed content")]  # substantive
    (tmp_path / "tests.json").write_text(json.dumps(drifted))
    with pytest.raises(UserFacingError, match="beyond stamping/serialization"):
        _assembly_rebind_lineage(tmp_path, "s1", {"tests.json": prior})
    entry = lineage.lookup(lineage.load(tmp_path), "tests", "tests.json")
    assert entry["sha256"] == settled_sha, "a refused render never re-binds"


def test_assembly_rebind_preserves_founder_decisions(tmp_path, monkeypatch):
    """Architecture normalized through the contract model: an identical semantic
    payload (with founder decisions) is admitted; dropping the decision refuses."""
    from spec_generator.cli import _assembly_rebind_lineage, _stamped
    from tests.test_architecture_decisions import _arch, DECISION
    monkeypatch.setenv("VMODEL_PROGRESS_TASK_ID", "t1")
    monkeypatch.setenv("VMODEL_PROGRESS_GENERATION", "7")
    arch = _arch(DECISION)
    prior = _bound_file(tmp_path, "architecture", "architecture.json", _stamped(arch))
    (tmp_path / "architecture.json").write_text(json.dumps(arch))  # stamp dropped only
    _assembly_rebind_lineage(tmp_path, "s1", {"architecture.json": prior})
    assert lineage.lookup(lineage.load(tmp_path), "architecture", "architecture.json")["settled"] is True
    stripped = _arch()  # same architecture WITHOUT the founder decision: substantive drift
    (tmp_path / "architecture.json").write_text(json.dumps(stripped))
    with pytest.raises(UserFacingError, match="beyond stamping/serialization"):
        _assembly_rebind_lineage(tmp_path, "s1", {"architecture.json": prior})
