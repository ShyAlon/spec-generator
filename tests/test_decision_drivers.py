"""ADR-0021: the architecture stage refuses arbitrary decisions - real driver
required, founder_constraint must cite an existing TechPreference, the grammar
narrows driver to the closed enum, and the gate render labels it unverified."""
import pytest

from vmodel_contracts.architecture import ArchitectureOverview
from spec_generator.graph import _check_decision_drivers
from spec_generator.render import render_architecture
from spec_generator.schemas import for_stage
from spec_generator.skeleton import ModelStructureError, assign_architecture_ids

ARCH = {"session_id": "s", "stack_entries": [{"key": "framework", "value": "React"}],
        "components": [{"name": "web", "responsibility": "UI", "requirement_ids": ["R-1"]}],
        "decisions": [{"title": "Frontend framework", "decision": "React",
                       "rationale": "Largest ecosystem for this archetype.",
                       "driver": "ecosystem_maturity",
                       "options_considered": [{"option": "Angular", "reason": "More ceremony for a solo build."}]}],
        "non_functionals": {}}
INTENT_NO_PREFS = {"session_id": "s", "tech_preferences": []}
INTENT_WITH_PREFS = {"session_id": "s", "tech_preferences": [
    {"area": "language", "preference": "python", "strength": "required", "source_turns": [1]}]}


def _arch(payload):
    merged = assign_architecture_ids(payload)
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


def test_legacy_arch_fill_makes_pre_080_checkpoints_assemblable():
    """A checkpoint written by pre-0.8.0 code holds decisions without
    driver/options_considered; assembly fills the honest legacy form so a
    resume past the architecture stage cannot crash (ADR-0021)."""
    from spec_generator.graph import _legacy_arch_fill
    checkpointed = {"session_id": "s", "stack": {"language": "ts"},
                    "components": [{"name": "api", "responsibility": "serve"}],
                    "decisions": [{"decision_id": "ADR-1", "title": "t", "decision": "x", "rationale": "y"}]}
    arch = ArchitectureOverview.model_validate(_legacy_arch_fill(checkpointed))
    assert arch.decisions[0].driver is None
    assert arch.decisions[0].options_considered == []
    assert "(not recorded - pre-0.8.0 artifact)" in render_architecture(arch)


def test_legacy_arch_fill_never_overwrites_real_drivers():
    from spec_generator.graph import _legacy_arch_fill
    filled = _legacy_arch_fill(ARCH)
    assert filled["decisions"][0]["driver"] == "ecosystem_maturity"
    assert filled["decisions"][0]["options_considered"][0]["option"] == "Angular"


def test_greenfield_labels_ungrounded_entries_new_design():
    arch = _arch({"session_id": "s", "stack_entries": [{"key": "database", "value": "Firestore",
                                                        "baseline_refs": []}],
                  "components": [], "decisions": []})
    md = render_architecture(arch, greenfield=True)
    assert "**database**: Firestore (new design)" in md
    assert "migration" not in md


def test_existing_mode_keeps_legacy_migration_label():
    arch = _arch({"session_id": "s", "stack_entries": [{"key": "database", "value": "X",
                                                        "baseline_refs": []}],
                  "components": [], "decisions": []})
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
