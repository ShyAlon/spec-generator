"""Version stamps are load-time gates, not changes to resume constraints."""
import copy
import json
import sys
from types import SimpleNamespace

import pytest
from vmodel_contracts import CONTRACT_VERSION, ConOpsDocument, SystemBaseline

from spec_generator import cli
from spec_generator.artifact_lifecycle import stable_hash
from spec_generator.errors import UserFacingError
from test_progress import INTENT, PAYLOADS
from test_conops_consumption import conops


def restamp(value, stamp):
    value = copy.deepcopy(value)
    if isinstance(value, dict):
        for key in value:
            value[key] = stamp if key == "contract_version" else restamp(value[key], stamp)
    elif isinstance(value, list):
        value = [restamp(item, stamp) for item in value]
    return value


def test_constraint_normalization_is_recursive_non_mutating_and_strict():
    old = {"contract_version": "0.19.0", "nested": [
        {"contract_version": "0.19.0", "id": "concept:pwa", "evidence": [1, 2]}]}
    before = copy.deepcopy(old)
    current = restamp(old, "0.20.0")
    assert cli._resume_constraint_content(old) == cli._resume_constraint_content(current)
    assert old == before
    for changed in (
        {**current, "extra": None},
        {**current, "nested": [{"id": "concept:native-app", "evidence": [1, 2]}]},
        {**current, "nested": [{"id": "concept:pwa", "evidence": [2, 1]}]},
    ):
        assert cli._resume_constraint_content(old) != cli._resume_constraint_content(changed)
    assert cli._resume_constraint_content(None) != cli._resume_constraint_content({})


class ReachedExecution(RuntimeError):
    """Stop before model calls, after exercising the real CLI guards."""


@pytest.fixture
def resume_case(tmp_path, monkeypatch):
    current_conops = conops()
    current_baseline = SystemBaseline.model_validate({
        "baseline_id": "b", "system_name": "Test", "mode": "greenfield",
        "repositories": [], "immutable_decisions": [],
    }).model_dump(mode="json")
    prior = {
        "conops": restamp(current_conops, "0.19.0"),
        "system_baseline": restamp(current_baseline, "0.19.0"),
        "architecture": copy.deepcopy(PAYLOADS["architecture"]), "needs": {"completed": True},
        "srs": {"completed": True},
    }
    updates = []
    graph = SimpleNamespace(
        get_state=lambda cfg: SimpleNamespace(values=prior, next=("items",)),
        update_state=lambda cfg, value, **kw: updates.append(value),
    )
    def invoke(*args, **kwargs):
        raise ReachedExecution("guards passed")
    graph.invoke = invoke
    monkeypatch.setattr(cli, "build_graph", lambda *args, **kwargs: graph)
    monkeypatch.setattr(cli, "_resume_after_bounded_stop", lambda *args: None)
    monkeypatch.setattr(cli, "_make_persist", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_backfill_persist", lambda *args: [])
    monkeypatch.setattr(cli, "_make_partials", lambda *args: SimpleNamespace(
        load=lambda key: {}, clear=lambda key: None))
    from vmodel_contracts import runtime
    monkeypatch.setattr(runtime, "install_plain_log", lambda path, stream: stream)
    from spec_generator import structured_progress
    monkeypatch.setattr(structured_progress, "StructuredProgress", lambda *args: SimpleNamespace(
        subscriber=lambda *args: None))
    from spec_generator import artifact_lifecycle
    adapters = []
    class Lifecycle:
        def __init__(self, outdir, session_id, baseline):
            self.anchor_hash = stable_hash(baseline)
            self.reanchors = []
            adapters.append(self)
        def bootstrap_complete(self, *args): pass
        state = SimpleNamespace(value="complete")
        def resume(self, *args, **kwargs): pass
        def reanchor(self, invalidates): self.reanchors.append(invalidates)
    monkeypatch.setattr(artifact_lifecycle, "ArchitectureLifecycle", Lifecycle)

    outdir = tmp_path / "spec"
    outdir.mkdir()
    (outdir / "checkpoints.sqlite").touch()
    intent_file = tmp_path / "intent_model.json"
    intent_file.write_text(json.dumps(dict(INTENT, contract_version=CONTRACT_VERSION)))
    conops_file = tmp_path / "conops.json"
    conops_file.write_text(json.dumps(current_conops))
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(json.dumps(current_baseline))
    argv = ["cli", str(intent_file), "--conops", str(conops_file),
            "--system-baseline", str(baseline_file), "--out", str(outdir)]
    def run(preflight=False, bounded=False):
        if bounded:
            monkeypatch.setattr(cli, "_resume_after_bounded_stop", lambda *args: "architecture")
        monkeypatch.setattr(sys, "argv", argv + (["--preflight"] if preflight else []))
        cli.main()
    return SimpleNamespace(run=run, prior=prior, updates=updates, adapters=adapters,
                           current_conops=current_conops, conops_file=conops_file,
                           current_baseline=current_baseline, baseline_file=baseline_file,
                           outdir=outdir)


def test_preflight_accepts_stamp_only_conops_and_baseline_drift(resume_case, capsys):
    resume_case.run(preflight=True)
    assert "Preflight OK" in capsys.readouterr().out
    assert not resume_case.updates


@pytest.mark.parametrize("bounded", [False, True])
def test_resume_accepts_stamp_only_drift_without_reanchor_or_artifact_deletion(resume_case, bounded):
    marker = resume_case.outdir / "architecture.json"
    marker.write_text("completed architecture")
    with pytest.raises(ReachedExecution):
        resume_case.run(bounded=bounded)
    assert not resume_case.updates
    assert not any(adapter.reanchors for adapter in resume_case.adapters)
    assert marker.read_text() == "completed architecture"
    assert resume_case.adapters[0].anchor_hash == stable_hash(resume_case.prior["system_baseline"])


@pytest.mark.parametrize("preflight", [True, False])
def test_real_conops_change_is_rejected(resume_case, preflight):
    changed = copy.deepcopy(resume_case.current_conops)
    changed["decision"]["rationale"] = "a different founder decision"
    resume_case.conops_file.write_text(json.dumps(changed))
    with pytest.raises(UserFacingError, match="Concept of Operations"):
        resume_case.run(preflight)
    assert not resume_case.updates


def test_preflight_rejects_real_baseline_change(resume_case):
    changed = {**resume_case.current_baseline, "baseline_id": "different"}
    resume_case.baseline_file.write_text(json.dumps(changed))
    with pytest.raises(UserFacingError, match="system baseline is incompatible"):
        resume_case.run(preflight=True)


def test_resume_real_baseline_change_still_invalidates_architecture(resume_case):
    changed = {**resume_case.current_baseline, "baseline_id": "different"}
    resume_case.baseline_file.write_text(json.dumps(changed))
    marker = resume_case.outdir / "architecture.json"
    marker.write_text("old architecture")
    with pytest.raises(ReachedExecution):
        resume_case.run()
    assert resume_case.adapters[0].reanchors
    assert resume_case.updates[0]["system_baseline"]["baseline_id"] == "different"
    assert resume_case.updates[0]["architecture"] is None
    assert not marker.exists()


@pytest.mark.parametrize("source", ["intent", "baseline"])
def test_explicit_load_time_version_gates_remain(resume_case, source):
    if source == "baseline":
        resume_case.baseline_file.write_text(json.dumps(restamp(resume_case.current_baseline, "0.19.0")))
    else:
        intent_file = resume_case.baseline_file.parent / "intent_model.json"
        intent_file.write_text(json.dumps(dict(INTENT, contract_version="0.19.0")))
    with pytest.raises(UserFacingError, match="0.19.0"):
        resume_case.run(preflight=True)


def test_conops_current_dump_is_content_identical_after_restamp():
    stored = restamp(conops(), "0.19.0")
    migrated = restamp(ConOpsDocument.model_validate(stored).model_dump(mode="json"), CONTRACT_VERSION)
    loaded = ConOpsDocument.model_validate_json(json.dumps(migrated)).model_dump(mode="json")
    assert cli._resume_constraint_content(loaded) == cli._resume_constraint_content(stored)


# Retained behavior executes through an explicit host-owned checkpoint factory.
import pytest
pytestmark=pytest.mark.usefixtures("supported_host_runtime")
