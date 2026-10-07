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


FRAMING = {"title": "Storing Your Data", "purpose": "Keeps the product's records between visits.", "default_value": "SQLite", "reason": "A single local user needs a small store.", "requirement_ids": ["R-1"], "options": [{"value": "SQLite", "label": "One Local Store", "implication": "Simple to run; a shared server may be needed later."}, {"value": "PostgreSQL", "label": "A Shared Store", "implication": "Supports shared access but adds a server to manage."}], "provenance": "design_proposal", "research_status": "pending"}

def _default_arch():
    arch = _arch(options=["PostgreSQL", "SQLite"])
    arch["stack_entry_dispositions"][0]["technical_choice"] = FRAMING
    return arch


def test_defaults_do_not_require_founder_homework_or_create_open_findings():
    base = _default_arch()
    AD.require_decided(base)
    assert AD.open_slots(base) == []
    assert AD.raise_findings(Ledger(), base) == set()
    chosen = AD.select_defaults(base)
    slot = chosen["stack_entry_dispositions"][0]
    assert slot["system_selection"] == {"value":"SQLite", "source":"architecture_default"}
    assert not slot.get("founder_decision")
    assert AD.architecture_revision(base) == AD.architecture_revision(chosen)
    md = render_architecture(ArchitectureOverview.model_validate(chosen))
    assert "System design default: SQLite" in md and "Decided by founder" not in md
    assert "external research pending" in md
    block = _founder_stack_block({"architecture":chosen})
    assert "system design default" in block and "SQLite" in block


def test_default_retirement_is_scoped_and_override_wins_without_reason():
    ledger=Ledger(); AD.raise_findings(ledger, _arch())
    other=ledger.add(rule_id="other", rule_version="1", source="test", kind="semantic", targets=["R-1"], artifact_revision="r", summary="Business constraint missing")
    AD.raise_findings(ledger, _default_arch())
    assert ledger.findings[other]["state"] == "OPEN"
    assert next(f for f in ledger.findings.values() if f["rule_id"]==AD.DECISIONS_RULE)["state"] == "SUPERSEDED"
    arch=AD.apply_decisions(_default_arch(), {"database":{"text":"Custom Store", "actor":"founder", "at":AT}})
    assert not AD.select_defaults(arch)["stack_entry_dispositions"][0].get("system_selection")
    assert "Custom Store" in _founder_stack_block({"architecture":arch})


def test_companion_carries_only_matching_system_selection():
    base=_default_arch(); companion=AD.companion_for(base)
    companion["slots"][0]["system_selection"]={"value":"SQLite", "source":"architecture_default"}
    merged=AD.merge_companion(base,companion)
    assert merged["stack_entry_dispositions"][0]["system_selection"]["value"]=="SQLite"
    companion["slots"][0]["system_selection"]["value"]="Redis"
    with pytest.raises(ValueError):AD.merge_companion(base,companion)


def test_targeted_refresh_updates_missing_metadata_only_and_is_resumable(tmp_path, monkeypatch):
    import copy
    from spec_generator.cli import _refresh_technical_choices
    import spec_generator.graph as module
    prior={"architecture":_arch(), "srs":{"functional":[{"requirement_id":"R-1"}]}, "items":[], "findings_ledger":Ledger().to_dict()}
    calls=[]; updates=[]; persisted=[]
    monkeypatch.setattr(module,"_technical_choice_call",lambda *a,**kw:(calls.append(a[1]) or FRAMING))
    class Graph:
        def get_state(self,cfg): return None
        @property
        def checkpointer(self): return self
        def get_tuple(self,cfg):
            from types import SimpleNamespace
            return SimpleNamespace(checkpoint={"versions_seen":{"architecture":{}}})
        def update_state(self,cfg,update,**kwargs): updates.append((copy.deepcopy(update),kwargs))
    result=_refresh_technical_choices(Graph(),{},prior,tmp_path,lambda *a:persisted.append(copy.deepcopy(a)),None)
    assert calls==["database"] and result["items"]==prior["items"]
    assert result["architecture"]["decisions"]==prior["architecture"]["decisions"]
    assert updates[0][1]=={"as_node":"architecture"} and persisted[0][0]=="architecture"
    _refresh_technical_choices(Graph(),{},result,tmp_path,lambda *a:None,None)
    assert calls==["database"]


def test_real_checkpoint_consumes_saved_default_without_requeueing_architecture(tmp_path):
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    from spec_generator.graph import build_graph, checkpoint_serde
    from spec_generator.cli import _apply_saved_architecture_decisions
    from spec_generator.telemetry import UsageLog
    conn=sqlite3.connect(tmp_path/'checkpoints.sqlite',check_same_thread=False)
    graph=build_graph(UsageLog('s'),checkpointer=SqliteSaver(conn,serde=checkpoint_serde()))
    cfg={'configurable':{'thread_id':'s'}}
    prior={'architecture':_default_arch(),'items':[], 'findings_ledger':Ledger().to_dict()}
    graph.update_state(cfg,prior,as_node='architecture')
    before=graph.get_state(cfg).next
    companion=AD.companion_for(AD.select_defaults(prior['architecture']))
    (tmp_path/AD.COMPANION_NAME).write_text(json.dumps(companion))
    writes=[]
    result=_apply_saved_architecture_decisions(graph,cfg,prior,tmp_path,lambda *a:writes.append(a))
    assert graph.get_state(cfg).next==before and 'architecture' not in before
    assert result['architecture']['stack_entry_dispositions'][0]['system_selection']['value']=='SQLite'
    assert writes[0][0]=='architecture'
    again=_apply_saved_architecture_decisions(graph,cfg,result,tmp_path,lambda *a:writes.append(a))
    assert again==result and len(writes)==1
    conn.close()


def test_targeted_refresh_rejects_existing_downstream_before_model_call(tmp_path,monkeypatch):
    from spec_generator.cli import _refresh_technical_choices
    import spec_generator.graph as module
    monkeypatch.setattr(module,'_technical_choice_call',lambda *a:pytest.fail('must not call model'))
    with pytest.raises(UserFacingError,match='downstream'):
        _refresh_technical_choices(None,{}, {'architecture':_arch(),'items':[{'spec_id':'keep'}]},tmp_path,None,None)


def test_proposed_default_is_validated_against_known_required_technology():
    from spec_generator.graph import _check_technical_choice_constraints
    state={'intent_model':{'tech_preferences':[{'area':'database','strength':'required','preference':'PostgreSQL'}]}}
    with pytest.raises(ValueError,match='required'):
        _check_technical_choice_constraints(state,'database',FRAMING)
