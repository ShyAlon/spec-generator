"""Repaired findings close when the re-review sees a changed target and does not raise them again."""
import pytest
from spec_generator.findings_ledger import Ledger, OPEN, SUPERSEDED, REJECTED
from spec_generator.graph import _ledger_review_findings, _target_revision


def srs(text):
    return {"functional": [{"requirement_id": "R-1", "statement": text, "acceptance": []},
                           {"requirement_id": "R-2", "statement": "stable", "acceptance": []}]}


def st(text):
    return {"srs": srs(text), "items": [], "tests": []}


def f(scope, ids, text):
    return {"scope": scope, "target_ids": ids, "fix": text}


def setup():
    ledger = Ledger()
    s1 = st("The app shall replicate the UI")
    raised = set()
    _ledger_review_findings(ledger, s1, [f("requirement", ["R-1"], "Clarify 'replicate the UI' to specify observable criteria"),
                                         f("requirement", ["R-2"], "R-2 needs a unit for latency")], 1, raised_out=raised)
    ids = {x["targets"][0]: x["finding_id"] for x in ledger.findings.values()}
    return ledger, ids


def rev_fn(state):
    return lambda fnd: _target_revision(state, {"scope": fnd["scope"], "target_ids": fnd["detail"].get("target_ids")})


def test_repaired_finding_closes_when_target_changed_and_not_reraised():
    ledger, ids = setup()
    ledger.history.append({"event": "repair", "finding_ids": [ids["R-1"]], "requested_by": "founder"})
    s2 = st("The app shall replicate the UI: lesson list, exercise screen and score screen")
    closed = ledger.close_repaired("review:round-2", set(), rev_fn(s2), [(["R-9"], "unrelated finding")])
    assert closed == [ids["R-1"]]
    f1 = ledger.findings[ids["R-1"]]
    assert f1["state"] == SUPERSEDED and "repaired" in f1["closure"]["reason"] and f1["closure"]["after_revision"] != f1["closure"]["before_revision"]
    assert any(h["event"] == "repair_closed" for h in ledger.history)
    assert ledger.findings[ids["R-2"]]["state"] == OPEN                      # never named in a repair


def test_stays_open_when_unchanged_reraised_or_not_repaired():
    ledger, ids = setup()
    ledger.history.append({"event": "repair", "finding_ids": [ids["R-1"]]})
    same = st("The app shall replicate the UI")
    assert ledger.close_repaired("review:round-2", set(), rev_fn(same), []) == []                  # no-op repair
    changed = st("The app shall replicate the UI: lesson list")
    again = [(["R-1"], "Clarify 'replicate the UI' to specify observable criteria for the lesson list")]
    assert ledger.close_repaired("review:round-2", set(), rev_fn(changed), again) == []            # re-raised, reworded
    assert ledger.findings[ids["R-1"]]["state"] == OPEN
    ledger2, ids2 = setup()                                                                       # never sent to repair
    assert ledger2.close_repaired("review:round-2", set(), rev_fn(changed), []) == []


def test_founder_dispositions_are_never_reclosed():
    ledger, ids = setup()
    f1 = ledger.findings[ids["R-1"]]
    ledger.founder_dispose(f1["finding_id"], actor="founder", source="founder-script", rationale="r",
                           artifact_revision=f1["artifact_revision"])
    ledger.history.append({"event": "repair", "finding_ids": [ids["R-1"]]})
    assert ledger.close_repaired("review:round-2", set(), rev_fn(st("changed")), []) == []
    assert ledger.findings[ids["R-1"]]["state"] == REJECTED
