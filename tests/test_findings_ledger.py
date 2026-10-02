import pytest

from spec_generator.content_acceptance import check_echo, DIRECTIVE_CATALOGUE
from spec_generator.findings_ledger import (ADDRESSED, OPEN, REJECTED, SUPERSEDED, Ledger, LedgerError,
                                            finding_from_check, revision_of)

REV = revision_of("body one")


def add(ledger, rule="review.llm", targets=("SPEC-1",), rev=REV, summary="R-1 is vague", ev=None, kind="semantic", source="review"):
    return ledger.add(rule_id=rule, rule_version="1", source=source, kind=kind, targets=list(targets),
                      artifact_revision=rev, summary=summary, evidence=ev)


def test_more_than_24_findings_survive_and_dedup_runs_first():
    ledger = Ledger()
    ids = {add(ledger, targets=(f"SPEC-{i}",), summary=f"finding {i}") for i in range(60)}
    assert len(ids) == 60 and len(ledger.findings) == 60   # no display cap in storage
    dup = add(ledger, targets=("SPEC-3",), summary="finding 3", source="jev")
    assert len(ledger.findings) == 60
    assert sorted(ledger.findings[dup]["sources"]) == ["jev", "review"]   # provenance kept on dedup


def test_different_evidence_is_a_different_finding_and_wording_becomes_an_alias():
    ledger = Ledger()
    a = add(ledger, ev=[{"id": "B1", "sha256": "aa"}])
    b = add(ledger, ev=[{"id": "B2", "sha256": "bb"}])
    assert a != b
    same = ledger.add(rule_id="review.llm", rule_version="1", source="review", kind="semantic", targets=["SPEC-1"],
                      artifact_revision=REV, summary="R-1 is vague", evidence=[{"id": "B1", "sha256": "aa"}],
                      alias="R-1 lacks detail")
    assert same == a and "R-1 lacks detail" in ledger.findings[a]["aliases"]


def test_semantic_finding_closes_only_by_founder_disposition_bound_to_revision():
    ledger = Ledger(); fid = add(ledger)
    with pytest.raises(LedgerError):   # a model/other actor cannot close it
        ledger.founder_dispose(fid, actor="gpt", source="decision-page", rationale="fixed", artifact_revision=REV)
    with pytest.raises(LedgerError):   # wrong source
        ledger.founder_dispose(fid, actor="founder", source="model", rationale="fixed", artifact_revision=REV)
    with pytest.raises(LedgerError):   # stale revision
        ledger.founder_dispose(fid, actor="founder", source="decision-page", rationale="ok", artifact_revision=revision_of("other"))
    with pytest.raises(LedgerError):   # no rationale
        ledger.founder_dispose(fid, actor="founder", source="decision-page", rationale=" ", artifact_revision=REV)
    ledger.founder_dispose(fid, actor="founder", source="decision-page", rationale="Reviewed the replacement",
                           artifact_revision=REV, subtype="resolved_by_reviewed_replacement")
    f = ledger.findings[fid]
    assert f["state"] == REJECTED and f["closure"]["subtype"] == "resolved_by_reviewed_replacement"
    assert not ledger.gate()["blocked"]


def test_semantic_finding_cannot_be_addressed_by_a_changed_artifact():
    ledger = Ledger(); fid = add(ledger)
    with pytest.raises(LedgerError, match="no reproducible predicate"):
        ledger.address(fid, before_revision=REV, after_revision=revision_of("new"), predicate_id="p",
                       predicate_version="1", predicate_still_fires=False)
    assert ledger.findings[fid]["state"] == OPEN and ledger.gate()["blocked"]


def test_deterministic_closure_needs_change_and_passing_predicate():
    ledger = Ledger()
    echo = check_echo(DIRECTIVE_CATALOGUE["watchdog-retry"])[0].to_dict()
    fid = finding_from_check(ledger, echo, "SPEC-4", DIRECTIVE_CATALOGUE["watchdog-retry"])
    rev = ledger.findings[fid]["artifact_revision"]
    new = revision_of("Store note events locally.")
    with pytest.raises(LedgerError, match="unchanged"):
        ledger.address(fid, before_revision=rev, after_revision=rev, predicate_id="echo", predicate_version="1", predicate_still_fires=False)
    with pytest.raises(LedgerError, match="still fires"):
        ledger.address(fid, before_revision=rev, after_revision=new, predicate_id="echo", predicate_version="1", predicate_still_fires=True)
    ledger.address(fid, before_revision=rev, after_revision=new, predicate_id="echo", predicate_version="1", predicate_still_fires=False)
    assert ledger.findings[fid]["state"] == ADDRESSED and ledger.findings[fid]["closure"]["after_revision"] == new


def test_structural_findings_are_never_waivable():
    ledger = Ledger(); fid = add(ledger, rule="structural", kind="structural")
    with pytest.raises(LedgerError, match="structural"):
        ledger.founder_dispose(fid, actor="founder", source="decision-page", rationale="x", artifact_revision=REV)


def test_changed_dependency_reopens_a_stale_disposition_and_keeps_history():
    ledger = Ledger(); fid = add(ledger)
    ledger.founder_dispose(fid, actor="founder", source="decision-page", rationale="ok", artifact_revision=REV)
    assert not ledger.gate({"SPEC-1": REV})["blocked"]
    gate = ledger.gate({"SPEC-1": revision_of("edited")})
    assert gate["blocked"] and gate["stale"] == [fid]
    reopened = ledger.refresh({"SPEC-1": revision_of("edited")})
    assert len(reopened) == 1
    assert ledger.findings[fid]["state"] == SUPERSEDED and ledger.findings[fid]["superseded_by"] == reopened[0]
    assert ledger.findings[reopened[0]]["state"] == OPEN and ledger.findings[reopened[0]]["supersedes"] == fid


def test_gate_blocks_on_open_incomplete_and_quarantines_targets():
    ledger = Ledger()
    assert not ledger.gate()["blocked"]
    ledger.mark_incomplete("review:chunk-3", "chunk output truncated")
    assert ledger.gate()["blocked"] and "incomplete" in ledger.gate()["reasons"][0]
    ledger.clear_incomplete("review:chunk-3")
    add(ledger, targets=("SPEC-2",))
    assert ledger.quarantined() == {"SPEC-2"} and ledger.gate()["blocked"]


def test_script_replays_exact_founder_dispositions():
    ledger = Ledger(); fid = add(ledger, rule="echo.directive")
    script = {"version": 1, "authored_by": "founder", "dispositions": [
        {"selector": {"rule_id": "echo.directive", "target_ids": ["SPEC-1"]}, "expected_artifact_sha256": REV,
         "action": "reject", "rationale": "Quoted evidence, not an echo", "reason_subtype": "false_positive"}]}
    assert ledger.apply_script(script) == [fid]
    assert ledger.findings[fid]["closure"]["source"] == "founder-script"


@pytest.mark.parametrize("mutate,why", [
    (lambda s: s.update(authored_by="model"), "authored_by"),
    (lambda s: s["dispositions"][0]["selector"].update(rule_id="*"), "wildcard"),
    (lambda s: s["dispositions"][0]["selector"].update(target_ids=["SPEC-9"]), "no open finding"),
    (lambda s: s["dispositions"][0].update(expected_artifact_sha256=revision_of("old")), "stale"),
    (lambda s: s.update(dispositions=[]), "no dispositions"),
])
def test_script_unknown_stale_wildcard_or_model_authored_stops_the_run(mutate, why):
    ledger = Ledger(); fid = add(ledger, rule="echo.directive")
    script = {"version": 1, "authored_by": "founder", "dispositions": [
        {"selector": {"rule_id": "echo.directive", "target_ids": ["SPEC-1"]}, "expected_artifact_sha256": REV,
         "action": "reject", "rationale": "ok"}]}
    mutate(script)
    with pytest.raises(LedgerError):
        ledger.apply_script(script)
    assert ledger.findings[fid]["state"] == OPEN   # nothing half-applied


def test_ambiguous_selector_is_refused_and_replay_is_identical():
    ledger = Ledger()
    add(ledger, rule="echo.directive", ev=[{"id": "a", "sha256": "1"}])
    add(ledger, rule="echo.directive", ev=[{"id": "b", "sha256": "2"}])
    script = {"version": 1, "authored_by": "founder", "dispositions": [
        {"selector": {"rule_id": "echo.directive", "target_ids": ["SPEC-1"]}, "expected_artifact_sha256": REV,
         "action": "reject", "rationale": "ok"}]}
    with pytest.raises(LedgerError, match="ambiguous"):
        ledger.apply_script(script)


def test_roundtrip_and_old_checkpoint_without_ledger_is_empty_not_clean():
    ledger = Ledger(); add(ledger); ledger.mark_incomplete("x", "y")
    again = Ledger.from_dict(ledger.to_dict())
    assert again.to_dict() == ledger.to_dict()
    assert Ledger.from_dict(None).findings == {}
    with pytest.raises(LedgerError):
        Ledger.from_dict({"version": 99})


def _by_id(fid, **over):
    d = {"selector": {"finding_id": fid}, "expected_artifact_sha256": REV, "action": "reject",
         "rationale": "ok", "reason_subtype": "false_positive"}
    d.update(over)
    return {"version": 1, "authored_by": "founder", "dispositions": [d]}


def test_finding_id_selector_disposes_exactly_one_of_several_same_rule_same_target():
    ledger = Ledger()
    a = add(ledger, rule="echo.directive", ev=[{"id": "a", "sha256": "1"}])
    b = add(ledger, rule="echo.directive", ev=[{"id": "b", "sha256": "2"}])
    assert ledger.apply_script(_by_id(a)) == [a]
    assert ledger.findings[a]["state"] == "REJECTED" and ledger.findings[b]["state"] == OPEN
    assert ledger.findings[a]["closure"]["source"] == "founder-script"


@pytest.mark.parametrize("sel,why", [
    ({"finding_id": "F-*"}, "exact id"),
    ({"finding_id": "F-0123456789abcdef"}, "no open finding"),
    ({"finding_id": "F-0123456789abcdef", "rule_id": "echo.directive"}, "stands alone"),
])
def test_finding_id_selector_refuses_wildcards_unknown_and_mixed_forms(sel, why):
    ledger = Ledger(); add(ledger, rule="echo.directive")
    script = _by_id("x"); script["dispositions"][0]["selector"] = sel
    with pytest.raises(LedgerError, match=why):
        ledger.apply_script(script)


def test_finding_id_selector_is_still_hash_bound_and_atomic():
    ledger = Ledger(); a = add(ledger, rule="echo.directive")
    with pytest.raises(LedgerError, match="stale"):
        ledger.apply_script(_by_id(a, expected_artifact_sha256=revision_of("old")))
    assert ledger.findings[a]["state"] == OPEN
