"""lineage.json authority record: binding, verification, substitution refusal,
--from invalidation, and standalone absence semantics."""
import json

import pytest

from spec_generator import lineage
from spec_generator.lineage import LineageRefusal


def _write(path, payload=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return lineage.sha256_file(path)


def test_restore_binding_admits_exact_bytes(tmp_path):
    sha = _write(tmp_path / "srs.json", b'{"a": 1}')
    lineage.bind(tmp_path, task_id="t1", session_id="s1", generation="3",
                 phase="srs", name="srs.json", sha256=sha, source="restore", revision="r9")
    entry = lineage.verify_file(lineage.load(tmp_path), phase="srs", name="srs.json",
                                path=tmp_path / "srs.json", task_id="t1", session_id="s1", generation="3")
    assert entry["revision"] == "r9" and entry["source"] == "restore"


def test_changed_bytes_refuse_with_recovery(tmp_path):
    sha = _write(tmp_path / "architecture.json", b"approved")
    lineage.bind(tmp_path, task_id="t", session_id="s", generation="1",
                 phase="architecture", name="architecture.json", sha256=sha, source="restore")
    _write(tmp_path / "architecture.json", b"regenerated")
    with pytest.raises(LineageRefusal, match="changed on disk"):
        lineage.verify_file(lineage.load(tmp_path), phase="architecture",
                            name="architecture.json", path=tmp_path / "architecture.json")


def test_missing_file_and_missing_binding_refuse(tmp_path):
    record = lineage.load(tmp_path) or {"version": 1, "inputs": {}, "entries": []}
    with pytest.raises(LineageRefusal, match="no settled binding"):
        lineage.verify_file(record, phase="srs", name="srs.json", path=tmp_path / "srs.json")
    sha = _write(tmp_path / "srs.json")
    lineage.bind(tmp_path, task_id="t", session_id="s", generation="1",
                 phase="srs", name="srs.json", sha256=sha, source="restore")
    (tmp_path / "srs.json").unlink()
    with pytest.raises(LineageRefusal, match="missing on disk"):
        lineage.verify_file(lineage.load(tmp_path), phase="srs", name="srs.json",
                            path=tmp_path / "srs.json")


def test_stage_commit_never_replaces_restore_binding_silently(tmp_path):
    approved = _write(tmp_path / "architecture.json", b"approved")
    lineage.bind(tmp_path, task_id="t", session_id="s", generation="1",
                 phase="architecture", name="architecture.json", sha256=approved, source="restore")
    with pytest.raises(LineageRefusal, match="explicit rewind"):
        lineage.bind(tmp_path, task_id="t", session_id="s", generation="1",
                     phase="architecture", name="architecture.json",
                     sha256=lineage.sha256_file(tmp_path / "architecture.json") + "0",
                     source="stage-commit")
    # identical re-commit is a no-op
    lineage.bind(tmp_path, task_id="t", session_id="s", generation="1",
                 phase="architecture", name="architecture.json", sha256=approved, source="stage-commit")
    assert len(lineage.load(tmp_path)["entries"]) == 1


def test_founder_decision_advances_binding(tmp_path):
    approved = _write(tmp_path / "architecture.json", b"approved")
    lineage.bind(tmp_path, task_id="t", session_id="s", generation="1",
                 phase="architecture", name="architecture.json", sha256=approved, source="restore")
    decided = _write(tmp_path / "architecture.json", b"approved+decided")
    lineage.bind(tmp_path, task_id="t", session_id="s", generation="1",
                 phase="architecture", name="architecture.json", sha256=decided,
                 source="founder-decision", revision="r2")
    entry = lineage.verify_file(lineage.load(tmp_path), phase="architecture",
                                name="architecture.json", path=tmp_path / "architecture.json")
    assert entry["source"] == "founder-decision" and entry["revision"] == "r2"


def test_wrong_run_identity_refuses(tmp_path):
    sha = _write(tmp_path / "items.json")
    lineage.bind(tmp_path, task_id="t1", session_id="s1", generation="1",
                 phase="items", name="items.json", sha256=sha, source="restore")
    with pytest.raises(LineageRefusal, match="another run's output"):
        lineage.verify_file(lineage.load(tmp_path), phase="items", name="items.json",
                            path=tmp_path / "items.json", task_id="OTHER", session_id="s1")


def test_input_hash_mismatch_refuses(tmp_path):
    intent = _write(tmp_path / "intent_model.json", b"intent-v1")
    lineage.record_inputs(tmp_path, task_id="t", session_id="s", generation="1",
                          input_hashes={"intent_model.json": intent})
    _write(tmp_path / "intent_model.json", b"intent-v2")
    with pytest.raises(LineageRefusal, match="differs from the settled record"):
        lineage.verify_inputs(lineage.load(tmp_path), {"intent_model.json": tmp_path / "intent_model.json"})


def test_invalidate_from_drops_rewound_and_downstream(tmp_path):
    for phase in ("needs", "srs", "architecture", "items"):
        sha = _write(tmp_path / f"{phase}.json", phase.encode())
        lineage.bind(tmp_path, task_id="t", session_id="s", generation="1",
                     phase=phase, name=f"{phase}.json", sha256=sha, source="restore")
    lineage.invalidate_from(tmp_path, {"architecture", "items", "tests", "validation"})
    record = lineage.load(tmp_path)
    assert sorted(e["phase"] for e in record["entries"]) == ["needs", "srs"]


def test_no_record_is_standalone_not_an_error(tmp_path):
    assert lineage.load(tmp_path) is None
    # binding creates the record on demand (stage commits in a fresh session)
    sha = _write(tmp_path / "needs.json")
    lineage.bind(tmp_path, task_id="", session_id="s", generation="standalone",
                 phase="needs", name="needs.json", sha256=sha, source="stage-commit")
    assert lineage.lookup(lineage.load(tmp_path), "needs", "needs.json")["source"] == "stage-commit"


def test_corrupt_record_refuses_with_recovery(tmp_path):
    (tmp_path / "lineage.json").write_text("{not json")
    with pytest.raises(LineageRefusal, match="unreadable"):
        lineage.load(tmp_path)


def test_provisional_restore_binding_allows_gate_feedback_regeneration(tmp_path):
    # awaiting_approval at restore: the open gate may regenerate the phase.
    old = _write(tmp_path / "srs.json", b"draft-v1")
    lineage.bind(tmp_path, task_id="t", session_id="s", generation="1",
                 phase="srs", name="srs.json", sha256=old, source="restore", settled=False)
    new = _write(tmp_path / "srs.json", b"draft-v2-after-feedback")
    lineage.bind(tmp_path, task_id="t", session_id="s", generation="1",
                 phase="srs", name="srs.json", sha256=new, source="stage-commit")
    assert lineage.verify_file(lineage.load(tmp_path), phase="srs", name="srs.json",
                               path=tmp_path / "srs.json")["sha256"] == new


def test_mark_settled_flips_only_the_named_phase(tmp_path):
    lineage.bind(tmp_path, task_id="t", session_id="s", generation="g", phase="srs",
                 name="srs.json", sha256="a" * 64, source="restore", settled=False)
    lineage.bind(tmp_path, task_id="t", session_id="s", generation="g", phase="architecture",
                 name="architecture.json", sha256="b" * 64, source="restore", settled=False)
    lineage.mark_settled(tmp_path, "srs")
    record = lineage.load(tmp_path)
    assert lineage.lookup(record, "srs", "srs.json")["settled"] is True
    assert lineage.lookup(record, "architecture", "architecture.json")["settled"] is False
    lineage.mark_settled(tmp_path, "srs")  # idempotent
    assert lineage.mark_settled(tmp_path / "missing", "srs") is None
