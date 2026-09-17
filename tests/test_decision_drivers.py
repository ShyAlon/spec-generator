"""ADR-0021: the architecture stage refuses arbitrary decisions - real driver
required, founder_constraint must cite an existing TechPreference, the grammar
narrows driver to the closed enum, and the gate render labels it unverified."""
import pytest

from vmodel_contracts.architecture import ArchitectureOverview
from spec_generator.graph import _check_decision_drivers
from spec_generator.render import render_architecture
from spec_generator.schemas import for_stage
from spec_generator.skeleton import ModelStructureError, assign_architecture_ids

ARCH = {"session_id": "s", "stack_entries": [{"key": "framework", "value": "React", "technology_id":"tech:react", "decision_id":"ADR-1"}],
        "components": [{"name": "web", "responsibility": "UI", "requirement_ids": ["R-1"]}],
        "decisions": [{"title": "Frontend framework", "decision": "React",
                       "rationale": "Largest ecosystem for this archetype.",
                       "driver": "ecosystem_maturity",
                       "options_considered": [{"option": "Angular", "reason": "More ceremony for a solo build."}], "executes_concept_ids":[], "technology_selections":[{"slot_id":"slot:framework","technology_id":"tech:react","display_name":"React"}]}],
        "non_functionals": {}}
INTENT_NO_PREFS = {"session_id": "s", "tech_preferences": []}
INTENT_WITH_PREFS = {"session_id": "s", "tech_preferences": [
    {"area": "language", "preference": "python", "strength": "required", "source_turns": [1]}]}


def _arch(payload):
    from spec_generator.tech_candidates import resolve_technology
    merged = assign_architecture_ids(payload)
    decisions=merged.setdefault("decisions",[])
    if merged.get("stack_entries") and not decisions:
        decisions.append({"decision_id":"ADR-1","title":"Stack","decision":"Stack","rationale":"test fixture","driver":"simplicity","options_considered":[]})
    for i,d in enumerate(decisions,1):
        d.setdefault("decision_id",f"ADR-{i}");d.setdefault("executes_concept_ids",[]);d.setdefault("technology_selections",[])
    owner=decisions[0] if decisions else None
    for e in merged.get("stack_entries",[]):
        if "technology_id" not in e:
            tid,row=resolve_technology(e["value"]);e["technology_id"]=tid
        e.setdefault("decision_id",owner["decision_id"])
        if owner and not any(x["slot_id"]=="slot:"+e["key"].replace("_","-") for x in owner["technology_selections"]):
            sel={"slot_id":"slot:"+e["key"].replace("_","-"),"technology_id":e["technology_id"],"display_name":e["value"],"baseline_refs":e.get("baseline_refs",[])}
            if e["technology_id"].startswith("off-list:"): sel["off_list_justification"]="test fixture"
            owner["technology_selections"].append(sel)
    return ArchitectureOverview.model_validate({**merged, "session_id": "s"})


def test_valid_decision_passes():
    _check_decision_drivers(_arch(ARCH), INTENT_NO_PREFS)


def test_missing_driver_key_fails_contract_validation():
    # contract-required (his call): the KEY must be present...
    bad = {"session_id": "s", "stack_entries": [], "components": [],
           "decisions": [{"title": "t", "decision": "x", "rationale": "y",
                          "options_considered": []}]}
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _arch(bad)


def test_null_driver_is_a_retryable_model_error():
    # ...and a null VALUE (only legal in migrated legacy artifacts) is a
    # stage-level retryable error for new production.
    bad = {"session_id": "s", "stack_entries": [], "components": [],
           "decisions": [{"title": "t", "decision": "x", "rationale": "y",
                          "driver": None, "options_considered": []}]}
    with pytest.raises(ModelStructureError, match="no driver"):
        _check_decision_drivers(_arch(bad), INTENT_NO_PREFS)


def test_founder_constraint_must_cite_an_existing_preference():
    bad = {"session_id": "s", "stack_entries": [], "components": [],
           "decisions": [{"title": "Language", "decision": "We use python",
                          "rationale": "It is nice.", "driver": "founder_constraint",
                          "options_considered": []}]}
    # 'python' IS the founder's stated preference but the rationale does not
    # cite it as the driver... the decision text names it, so this passes:
    _check_decision_drivers(_arch(bad), INTENT_WITH_PREFS)
    # a founder_constraint decision citing nothing the founder said fails:
    bad2 = {"session_id": "s", "stack_entries": [], "components": [],
            "decisions": [{"title": "Language", "decision": "We use ruby",
                           "rationale": "It is nice.", "driver": "founder_constraint",
                           "options_considered": []}]}
    with pytest.raises(ModelStructureError, match="founder_constraint"):
        _check_decision_drivers(_arch(bad2), INTENT_WITH_PREFS)


def test_grammar_narrows_driver_to_the_closed_enum():
    dec = for_stage("architecture")["properties"]["decisions"]["items"]
    if "$ref" in dec:
        dec = for_stage("architecture")["$defs"]["ArchitectureDecision"]
    assert dec["properties"]["driver"] == {"enum": ["founder_constraint", "ecosystem_maturity",
                                                    "community_adoption", "team_fit", "simplicity"]}
    assert "driver" in dec["required"] and "options_considered" in dec["required"]


def test_gate_render_shows_driver_options_and_matrix_evidence():
    md = render_architecture(_arch(ARCH))
    assert "Driver: ecosystem maturity (matrix: registry weekly downloads" in md
    assert "Considered: Angular - rejected: More ceremony for a solo build." in md


def test_gate_render_marks_legacy_driverless_decisions():
    legacy = _arch({"session_id": "s", "stack_entries": [], "components": [],
                    "decisions": [{"title": "t", "decision": "x", "rationale": "y",
                                   "driver": None, "options_considered": [],
                                   "rejected": ["z - old flat form"]}]})
    md = render_architecture(legacy)
    assert "Driver: (not recorded - pre-0.8.0 artifact)" in md
    assert "Rejected: z - old flat form" in md


def test_legacy_architecture_requires_identity_stage_restart():
    """Pre-identity architecture cannot invent ADR ownership during migration."""
    from spec_generator.graph import _legacy_arch_fill
    checkpointed = {"session_id": "s", "stack": {"language": "ts"},
                    "components": [{"name": "api", "responsibility": "serve"}],
                    "decisions": [{"decision_id": "ADR-1", "title": "t", "decision": "x", "rationale": "y"}]}
    with pytest.raises(Exception):
        ArchitectureOverview.model_validate(_legacy_arch_fill(checkpointed))


def test_legacy_arch_fill_never_overwrites_real_drivers():
    from spec_generator.graph import _legacy_arch_fill
    filled = _legacy_arch_fill(ARCH)
    assert filled["decisions"][0]["driver"] == "ecosystem_maturity"
    assert filled["decisions"][0]["options_considered"][0]["option"] == "Angular"


def test_greenfield_labels_ungrounded_entries_new_design():
    arch = _arch({"session_id": "s", "stack_entries": [{"key": "database", "value": "Firestore", "technology_id":"tech:firestore", "decision_id":"ADR-1", "baseline_refs": []}],
                  "components": [], "decisions": [{"decision_id":"ADR-1","title":"Database","decision":"Firestore","rationale":"fit","driver":"simplicity","options_considered":[],"executes_concept_ids":[],"technology_selections":[{"slot_id":"slot:database","technology_id":"tech:firestore","display_name":"Firestore","off_list_justification":None}]}]})
    md = render_architecture(arch, greenfield=True)
    assert "**database**: Firestore (new design)" in md
    assert "migration" not in md


def test_existing_mode_keeps_legacy_migration_label():
    arch = _arch({"session_id": "s", "stack_entries": [{"key": "database", "value": "X", "technology_id":"off-list:x", "decision_id":"ADR-1", "baseline_refs": []}],
                  "components": [], "decisions": [{"decision_id":"ADR-1","title":"Database","decision":"X","rationale":"fit","driver":"simplicity","options_considered":[],"executes_concept_ids":[],"technology_selections":[{"slot_id":"slot:database","technology_id":"off-list:x","display_name":"X","off_list_justification":"test-only"}]}]})
    assert "(migration: see ADRs)" in render_architecture(arch)


def test_maturity_driver_uses_matrix_service_votes():
    md = render_architecture(_arch({"session_id": "s", "stack_entries": [], "components": [],
        "decisions": [{"title": "Adopt PostgreSQL", "decision": "Use PostgreSQL",
                       "rationale": "Its ecosystem is mature.", "driver": "ecosystem_maturity",
                       "options_considered": []}], "non_functionals": {}}))
    assert "Driver: ecosystem maturity (matrix: 2 sources, score 2, consensus mixed/weak)" in md
    assert "model judgment - unverified" not in md


def test_maturity_driver_uses_matrix_registry_downloads():
    md = render_architecture(_arch({"session_id": "s", "stack_entries": [], "components": [],
        "decisions": [{"title": "Adopt Django", "decision": "Use Django",
                       "rationale": "Its ecosystem is mature.", "driver": "community_adoption",
                       "options_considered": []}], "non_functionals": {}}))
    assert "Driver: community adoption (matrix: registry weekly downloads" in md


def test_unmeasured_or_non_popularity_drivers_stay_unverified():
    base = {"session_id": "s", "stack_entries": [], "components": [], "non_functionals": {}}
    novel = {"title": "Adopt NovelDB", "decision": "Use NovelDB", "rationale": "mature",
             "driver": "ecosystem_maturity", "options_considered": []}
    md = render_architecture(_arch({**base, "decisions": [novel]}))
    assert "model judgment - unverified; challenge here" in md

    measured_but_non_popularity = {
        "title": "Adopt PostgreSQL", "decision": "Use PostgreSQL", "rationale": "easy",
        "driver": "simplicity", "options_considered": []}
    md = render_architecture(_arch({**base, "decisions": [measured_but_non_popularity]}))
    assert "model judgment - unverified" in md
    assert "adoption signal only, not proof of simplicity" in md


def test_alternative_in_rationale_cannot_lend_matrix_evidence():
    md = render_architecture(_arch({"session_id": "s", "stack_entries": [], "components": [],
        "decisions": [{"title": "Adopt NovelDB", "decision": "Use NovelDB",
                       "rationale": "Unlike PostgreSQL, it fits our custom protocol.",
                       "driver": "ecosystem_maturity", "options_considered": []}],
        "non_functionals": {}}))
    assert "model judgment - unverified; challenge here" in md
    assert "matrix: 2 sources" not in md


def test_non_adoption_driver_still_shows_matrix_without_claiming_it_proves_simplicity():
    md = render_architecture(_arch({"session_id": "s", "stack_entries": [], "components": [],
        "decisions": [{"title": "Adopt Supabase", "decision": "Use Supabase",
                       "rationale": "It is simple.", "driver": "simplicity",
                       "options_considered": []}], "non_functionals": {}}))
    assert "Driver: simplicity (model judgment - unverified; matrix: 2 sources" in md
    assert "adoption signal only, not proof of simplicity" in md


def test_technology_selection_must_be_in_owning_adr_substance():
    from spec_generator.graph import _check_adr_substance
    from spec_generator.skeleton import ModelStructureError
    payload = {"session_id":"s", "stack_entries":[{"key":"agent_orchestration","value":"Temporal","technology_id":"tech:temporal","decision_id":"ADR-1"}],
      "components":[], "decisions":[{"decision_id":"ADR-1","title":"Insurer correspondence boundary","decision":"Use per-insurer adapters","rationale":"Normalize claim exchange","driver":"simplicity","options_considered":[],"executes_concept_ids":[],"technology_selections":[{"slot_id":"slot:agent-orchestration","technology_id":"tech:temporal","display_name":"Temporal"}]}], "non_functionals":{}}
    with pytest.raises(ModelStructureError, match="Temporal.*absent"):
        _check_adr_substance(__import__('vmodel_contracts').ArchitectureOverview.model_validate(payload))


def test_repeated_subject_and_alternatives_fail_adr_gate():
    from spec_generator.graph import _check_adr_substance
    from spec_generator.skeleton import ModelStructureError
    options=[{"option":"generic insurer integration","reason":"variance"},{"option":"direct point to point integration","reason":"duplicates"},{"option":"human operated integration","reason":"not autonomous"}]
    def adr(i,title,decision): return {"decision_id":f"ADR-{i}","title":title,"decision":decision,"rationale":"Insurer variance requires normalized correspondence adapters","driver":"simplicity","options_considered":options,"executes_concept_ids":[],"technology_selections":[]}
    payload={"session_id":"s","stack_entries":[],"components":[],"decisions":[adr(1,"Per insurer adapter boundary","Use normalized correspondence adapters"),adr(2,"Normalized insurer correspondence boundary","Use per insurer correspondence adapters")],"non_functionals":{}}
    with pytest.raises(ModelStructureError, match="duplicate one architecture decision"):
        _check_adr_substance(__import__('vmodel_contracts').ArchitectureOverview.model_validate(payload))


def test_duplicate_adr_fallback_merges_losslessly_and_renumbers():
    from spec_generator.graph import _merge_duplicate_adr_pair
    options=[{"option":"A","reason":"one"},{"option":"B","reason":"two"}]
    slots=[
      {"decision_id":"ADR-1","slug":"first","migration":False,"authorizes":["stack:x"],"required_concept_ids":[]},
      {"decision_id":"ADR-2","slug":"distinct","migration":False,"authorizes":["stack:y"],"required_concept_ids":[]},
      {"decision_id":"ADR-3","slug":"duplicate","migration":False,"authorizes":["component:z"],"required_concept_ids":["concept:pwa"]},]
    bodies={
      "ADR-1":{"title":"Choose boundary","decision":"Use normalized boundary","rationale":"short","options_considered":options,"baseline_refs":["repo:a"],"executes_concept_ids":[]},
      "ADR-2":{"title":"Other","decision":"Do other","rationale":"distinct","options_considered":[],"baseline_refs":[],"executes_concept_ids":[]},
      "ADR-3":{"title":"Choose normalized boundary","decision":"Use normalized boundary for all adapters","rationale":"longer complete rationale","options_considered":options,"baseline_refs":["repo:b"],"executes_concept_ids":["concept:pwa"]},}
    _merge_duplicate_adr_pair(slots,bodies,"ADR-1","ADR-3")
    assert [s["decision_id"] for s in slots]==["ADR-1","ADR-2"]
    assert slots[0]["authorizes"]==["stack:x","component:z"]
    assert slots[0]["required_concept_ids"]==["concept:pwa"]
    assert set(bodies)=={"ADR-1","ADR-2"}
    assert bodies["ADR-1"]["baseline_refs"]==["repo:a","repo:b"]
    assert bodies["ADR-1"]["decision"]=="Use normalized boundary for all adapters"
    assert bodies["ADR-1"]["executes_concept_ids"]==["concept:pwa"]


def test_adr_substance_repair_feedback_carries_exact_guilty_pair():
    """The gate detail is embedded verbatim into the next ADR plan/body prompt."""
    detail="ADR-1 and ADR-8 duplicate one architecture decision (subject overlap 0.94, alternatives overlap 1.00)"
    gate_failures=[("ADR-substance",detail)]
    feedback=("\n\nTHE PREVIOUS ARCHITECTURE FAILED THESE ACTIVE DETERMINISTIC GATES. "
              "Fix ALL of them in this ADR set; do not trade one for another:\n- "
              + "\n- ".join(f"{name}: {message}" for name,message in gate_failures))
    assert "ADR-substance: ADR-1 and ADR-8 duplicate" in feedback
    assert "subject overlap 0.94, alternatives overlap 1.00" in feedback


def test_adr_overlap_boundary_logs_scores_threshold_inputs_and_decision(capsys):
    from spec_generator.graph import _duplicate_adr_pairs
    from vmodel_contracts import ArchitectureOverview
    options=[{"option":"Generic insurer adapter","reason":"x"},{"option":"Direct point integration","reason":"y"}]
    def d(i,title): return {"decision_id":f"ADR-{i}","title":title,"decision":"Use normalized adapter boundary","rationale":"r","driver":"simplicity","options_considered":options,"technology_selections":[],"executes_concept_ids":[]}
    arch=ArchitectureOverview.model_validate({"session_id":"s","stack_entries":[],"components":[],"decisions":[d(1,"Normalized adapter"),d(8,"Normalized adapter")],"non_functionals":{}})
    assert _duplicate_adr_pairs(arch)
    line=capsys.readouterr().err
    assert "boundary=adr-substance compared=ADR-1,ADR-8" in line
    assert "subject_overlap=" in line and "subject_threshold=0.30" in line
    assert "alternatives_overlap=" in line and "alternatives_threshold=0.65" in line
    assert "decision=duplicate" in line
