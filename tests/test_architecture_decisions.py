import json

import pytest
from vmodel_contracts import ArchitectureOverview

from spec_generator import architecture_decisions as AD
from spec_generator.cli import _write_decisions_companion
from spec_generator.errors import UserFacingError
from spec_generator.findings_ledger import Ledger
from spec_generator.graph import _assemble_architecture, _founder_stack_block
from spec_generator.render import render_architecture

AT = "2026-10-06T12:00:00Z"


def _arch(decided=None, options=None):
    entries = {
        "language": {"key": "language", "value": "Python", "technology_id": "tech:python", "baseline_refs": []},
        "database": {"key": "database", "status": "research_required", "research_brief": "Which store fits the data?"},
        "hardware": {"key": "hardware", "status": "not_applicable", "reason": "Browser only."},
    }
    if options:
        entries["database"]["candidate_options"] = options
    arch = _assemble_architecture(
        session_id="s", baseline_id="b", entries=entries, comp_slots=[], comp_bodies={},
        adr_slots=[{"decision_id": "ADR-1", "authorizes": ["stack:language", "stack:database", "stack:hardware"], "migration": False}],
        adr_bodies={"ADR-1": {"title": "Runtime", "decision": "Python", "rationale": "fit", "driver": "simplicity",
                              "options_considered": [], "executes_concept_ids": []}}, nfr={})
    return AD.apply_decisions(arch, decided) if decided else arch


DECISION = {"database": {"text": "SQLite", "reason": "Single user, no server", "actor": "founder", "at": AT}}


def test_only_research_required_slots_are_open_and_decisions_close_them():
    assert [d["key"] for d in AD.open_slots(_arch())] == ["database"]
    assert AD.open_slots(_arch(DECISION)) == []


def test_apply_refuses_blank_unknown_and_missing_reason():
    base = _arch()
    for bad in ({"database": {"text": " ", "reason": "r", "at": AT}},
                {"database": {"text": "SQLite", "reason": "", "at": AT}},
                {"database": {"text": "SQLite", "reason": "r"}},
                {"language": {"text": "x", "reason": "r", "at": AT}}):
        with pytest.raises(ValueError):
            AD.apply_decisions(base, bad)


def test_decided_architecture_validates_with_contracts_and_revision_is_stable():
    base, decided = _arch(), _arch(DECISION)
    ArchitectureOverview.model_validate(decided)
    assert AD.architecture_revision(base) == AD.architecture_revision(decided)


def test_require_decided_names_open_slots():
    with pytest.raises(UserFacingError) as e:
        AD.require_decided(_arch())
    assert "database" in e.value.cause
    AD.require_decided(_arch(DECISION))


def test_findings_raised_for_open_slots_and_retired_when_decided():
    ledger = Ledger()
    raised = AD.raise_findings(ledger, _arch())
    assert len(raised) == 1 and [f["state"] for f in ledger.findings.values()] == ["OPEN"]
    AD.raise_findings(ledger, _arch(DECISION))
    assert [f["state"] for f in ledger.findings.values()] == ["SUPERSEDED"]


def test_merge_companion_applies_decisions_and_rejects_other_revision():
    base = _arch()
    companion = AD.companion_for(base)
    companion["slots"][0]["founder_decision"] = DECISION["database"]
    assert AD.open_slots(AD.merge_companion(base, companion)) == []
    companion["revision"] = "other"
    with pytest.raises(ValueError):
        AD.merge_companion(base, companion)


def test_companion_file_prefill_only_when_key_and_brief_match(tmp_path):
    first = _arch(options=["SQLite", "PostgreSQL"])
    _write_decisions_companion(tmp_path, first)
    saved = json.loads((tmp_path / AD.COMPANION_NAME).read_text())
    assert saved["slots"][0]["options"] == ["SQLite", "PostgreSQL"] and saved["slots"][0]["prefill"] is None
    saved["slots"][0]["founder_decision"] = DECISION["database"]
    (tmp_path / AD.COMPANION_NAME).write_text(json.dumps(saved))
    _write_decisions_companion(tmp_path, first)  # redo, same key and brief
    assert json.loads((tmp_path / AD.COMPANION_NAME).read_text())["slots"][0]["prefill"]["text"] == "SQLite"
    changed = _arch()
    changed["stack_entry_dispositions"][0]["research_brief"] = "A different question"
    _write_decisions_companion(tmp_path, changed)
    assert json.loads((tmp_path / AD.COMPANION_NAME).read_text())["slots"][0]["prefill"] is None


def test_no_open_slot_means_no_companion_file(tmp_path):
    _write_decisions_companion(tmp_path, _arch())
    assert (tmp_path / AD.COMPANION_NAME).is_file()
    only_concrete = _arch(); only_concrete["stack_entry_dispositions"] = []
    _write_decisions_companion(tmp_path, only_concrete)
    assert not (tmp_path / AD.COMPANION_NAME).exists()


def test_render_and_items_prompt_show_the_founder_decision():
    decided = _arch(DECISION)
    md = render_architecture(ArchitectureOverview.model_validate(decided))
    assert "Decided by founder: SQLite" in md and "RESEARCH REQUIRED" not in md
    block = _founder_stack_block({"architecture": decided})
    assert "database: SQLite" in block and "Single user" in block
    assert _founder_stack_block({"architecture": _arch()}) == ""
