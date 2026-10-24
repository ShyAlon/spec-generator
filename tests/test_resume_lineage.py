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


def test_changed_settled_bytes_refuse_without_checkpoint_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("VMODEL_PROGRESS_TASK_ID", "t1")
    monkeypatch.setenv("VMODEL_PROGRESS_GENERATION", "7")
    _bind_artifact(tmp_path, "needs", "needs.json", NEEDS)
    (tmp_path / "needs.json").write_text(json.dumps({"needs": [{"tampered": True}]}))
    # no checkpoint coverage (the cross-host pickup case): disk must match the binding
    with pytest.raises(UserFacingError, match="refused this resume"):
        _hydrate_from_lineage(FakeGraph(), {}, {}, tmp_path, _args(tmp_path, "srs"), "s1", False)


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
