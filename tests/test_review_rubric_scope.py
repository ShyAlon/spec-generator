"""Review rubric per scope (2026-10-02 run: 116 findings, most asked specs/tests to drop the stack the ADRs chose)."""
from spec_generator.graph import (_decided_architecture_block, _decided_technologies, _filter_decided_stack_findings,
                                  _ledger_review_findings)
from spec_generator.findings_ledger import Ledger
from spec_generator.prompts import STAGE_PROMPTS

STATE = {"architecture": {
    "stack_entries": [{"key": "ci_cd", "value": "GitHub Actions"}, {"key": "hosting", "value": "GitHub Pages"},
                      {"key": "offline", "value": "Service Workers"}, {"key": "x", "value": "ab"}],
    "decisions": [{"title": "Host on GitHub Pages", "technology_selections": [{"display_name": "Workbox"}]}]},
    "items": [], "srs": {}}


def fix(scope, text, ids=("SPEC-1",)):
    return {"scope": scope, "target_ids": list(ids), "fix": text}


def test_prompt_scopes_the_solution_language_rule_to_requirements_only():
    p = STAGE_PROMPTS["review"].format(n=1)
    assert "requirement (the SRS): flag solution language" in p
    assert "spec and test:" in p and "REQUIRED" in p and "never ask to remove, replace, abstract or avoid" in p
    assert "solution language in the SRS" not in p            # the old one-line, scope-blind rule is gone
    assert "{scope, target_ids, fix}" in p                    # output contract intact after .format


def test_decided_architecture_block_lists_stack_adr_titles_and_selected_technologies():
    assert _decided_technologies(STATE) == ["GitHub Actions", "GitHub Pages", "Service Workers", "Workbox"]   # 'ab' too short
    block = _decided_architecture_block(STATE)
    assert "binding" in block and "GitHub Actions" in block and "Host on GitHub Pages" in block and "Workbox" in block


def test_decided_stack_findings_are_archived_not_opened():
    fixes = [fix("spec", "Remove 'GitHub Actions' and 'GitHub Pages' from SPEC-1; keep it solution-agnostic."),
             fix("spec", "Replace 'Service Workers' with a generic offline mechanism."),
             fix("test", "T-3 should avoid naming Workbox in its steps.", ("T-3",)),
             fix("spec", "SPEC-2 never states what happens when the sync fails.", ("SPEC-2",)),                    # real: missing behavior
             fix("spec", "SPEC-1 uses Firebase but the architecture decided GitHub Pages; remove the contradiction.")]  # not decided tech
    kept, filtered = _filter_decided_stack_findings(fixes, STATE)
    assert [f["fix"][:6] for f in filtered] == ["Remove", "Replac", "T-3 sh"]
    assert all(f["filtered"] == "filtered: decided-stack" and f["matched_technology"] for f in filtered)
    assert len(kept) == 2 and "never states" in kept[0]["fix"]


def test_requirement_and_systemic_findings_are_never_filtered_and_filtered_ones_never_reach_the_ledger():
    fixes = [fix("requirement", "R-4 names GitHub Pages; remove the solution language.", ("R-4",)),
             fix("systemic", "Replace GitHub Actions everywhere.", ()),
             fix("test", "Replace 'GitHub Pages' in the steps.", ("T-1",))]
    kept, filtered = _filter_decided_stack_findings(fixes, STATE)
    assert [f["scope"] for f in kept] == ["requirement", "systemic"] and len(filtered) == 1
    ledger = Ledger()
    _ledger_review_findings(ledger, {"srs": {}, "items": [], "tests": []}, kept, 1)
    assert len(ledger.open_findings()) == 2 and not any("T-1" in f["targets"] for f in ledger.open_findings())


def test_review_rerun_replaces_only_its_own_rounds_open_output():
    st = {"srs": {}, "items": [], "tests": []}
    ledger = Ledger()
    old = [fix("requirement", "R-1 is vague.", ("R-1",)), fix("requirement", "R-2 is vague.", ("R-2",))]
    _ledger_review_findings(ledger, st, old, 1)
    r2 = _ledger_review_findings(ledger, st, [fix("requirement", "R-9 is vague.", ("R-9",))], 2)   # other round: kept
    ids1 = [f["finding_id"] for f in ledger.open_findings() if f["sources"] == ["review:round-1"]]
    assert len(ids1) == 2 and r2 == []
    # founder already rejected one of round 1's findings: that decision is not touched
    ledger.founder_dispose(ids1[0], actor="founder", source="founder-script", rationale="ok",
                           artifact_revision=ledger.findings[ids1[0]]["artifact_revision"])
    # the re-run raises R-2 again plus a new one; R-1 is no longer raised
    retired = _ledger_review_findings(ledger, st, [fix("requirement", "R-2 is vague.", ("R-2",)),
                                                   fix("requirement", "R-3 is vague.", ("R-3",))], 1)
    states = {f["summary"]: f["state"] for f in ledger.findings.values()}
    assert states["R-1 is vague."] == "REJECTED" and states["R-2 is vague."] == "OPEN"
    assert states["R-3 is vague."] == "OPEN" and states["R-9 is vague."] == "OPEN" and retired == []
    # now a finding that is only open and not re-raised is replaced, with history
    retired = _ledger_review_findings(ledger, st, [fix("requirement", "R-2 is vague.", ("R-2",))], 1)
    assert len(retired) == 1 and ledger.findings[retired[0]]["state"] == "SUPERSEDED"
    assert ledger.history[-1]["event"] == "review_round_replaced"
