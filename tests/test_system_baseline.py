import copy
import json
import pytest
from vmodel_contracts import ArchitectureOverview, StackEntry, SystemBaseline
from spec_generator.graph import (_check_baseline_reconciliation, baseline_reference_index)
from spec_generator.skeleton import ModelStructureError, assign_architecture_ids
from spec_generator.schemas import for_stage


def baseline(mode="existing"):
    return SystemBaseline.model_validate({
        "baseline_id": "vmodel-v1", "system_name": "V-model", "mode": mode,
        "repositories": [] if mode == "greenfield" else [{
            "name": "spec-generator", "languages": ["Python"],
            "module_entry_points": ["spec_generator.cli"],
            "existing_artifacts": ["architecture.json"],
            "existing_contracts": ["IntentModel -> SpecBundle"],
            "runtime": ["Python 3.10+", "Ollama"], "persistence": ["SQLite"],
            "integrations": [], "immutable_decisions": ["code owns ids"]}],
        "immutable_decisions": []})


def architecture():
    return ArchitectureOverview.model_validate({**assign_architecture_ids({
        "stack_entries": [{"key": "language", "value": "Python",
                           "baseline_refs": ["repo:spec-generator"], "migration_adr_id": None},
                          {"key": "model_runtime", "value": "Ollama",
                           "baseline_refs": ["runtime:spec-generator:2"], "migration_adr_id": None}],
        "components": [{"name": "spec-generator", "responsibility": "generate artifacts",
                        "requirement_ids": ["R-1"], "baseline_refs": ["repo:spec-generator"],
                        "migration": False, "migration_adr_title": None}],
        "decisions": [{"title": "Preserve local runtime", "decision": "Use Ollama",
                       "rationale": "It is the recorded runtime", "driver": "founder_constraint",
                       "options_considered": [], "baseline_refs": ["runtime:spec-generator:2"],
                       "migration": False, "evidence": [], "affected_repositories": []}],
        "non_functionals": {}}), "session_id": "s", "baseline_id": "vmodel-v1"})


def test_canonical_refs_are_stable_and_cover_each_baseline_field():
    refs = baseline_reference_index(baseline())
    assert {"repo:spec-generator", "module:spec-generator:1", "artifact:spec-generator:1",
            "contract:spec-generator:1", "runtime:spec-generator:2",
            "persistence:spec-generator:1", "decision:spec-generator:1"} <= refs


def test_existing_architecture_must_trace_stack_and_components():
    _check_baseline_reconciliation(architecture(), baseline())
    a = architecture().model_copy(deep=True)
    a.components[0].baseline_refs = []
    with pytest.raises(ModelStructureError, match="requires baseline refs"):
        _check_baseline_reconciliation(a, baseline())
    a = architecture().model_copy(deep=True)
    a.stack_entries.append(StackEntry(key="language", value="Python", baseline_refs=["repo:spec-generator"]))
    with pytest.raises(ModelStructureError, match="unique"):
        _check_baseline_reconciliation(a, baseline())
    a = architecture().model_copy(deep=True)
    a.stack_entries.append(StackEntry(key="database", value="PostgreSQL", baseline_refs=[]))
    with pytest.raises(ModelStructureError, match="ADR tagged migration"):
        _check_baseline_reconciliation(a, baseline())


def test_unknown_generic_archetype_refs_are_rejected():
    a = architecture().model_copy(deep=True)
    a.stack_entries[0].baseline_refs = ["runtime:generic-web-app:1"]
    with pytest.raises(ModelStructureError, match="unknown baseline refs"):
        _check_baseline_reconciliation(a, baseline())


def test_departure_requires_migration_adr_evidence_and_affected_repo():
    a = architecture().model_copy(deep=True)
    a.stack_entries.append(StackEntry(key="database", value="PostgreSQL", baseline_refs=[]))
    with pytest.raises(ModelStructureError, match="ADR tagged migration"):
        _check_baseline_reconciliation(a, baseline())
    d = a.decisions[0]
    d.title = "Database migration"
    d.decision = "Adopt PostgreSQL database"
    d.migration = True; d.evidence = ["R-9 requires shared transactional state"]
    with pytest.raises(ModelStructureError, match="affected_repositories"):
        _check_baseline_reconciliation(a, baseline())
    d.affected_repositories = ["spec-generator"]
    _check_baseline_reconciliation(a, baseline())  # legacy title-linked form heals

def test_departure_authorized_by_pipeline_wired_adr_id():
    a = architecture().model_copy(deep=True)
    d = a.decisions[0]
    d.title = "Database migration"; d.decision = "Adopt PostgreSQL"
    d.migration = True; d.evidence = ["R-9 requires shared transactional state"]
    d.affected_repositories = ["spec-generator"]
    a.stack_entries.append(StackEntry(key="database", value="PostgreSQL",
                                      baseline_refs=[], migration_adr_id=d.decision_id))
    _check_baseline_reconciliation(a, baseline())
    a.stack_entries[2].migration_adr_id = "ADR-99"
    with pytest.raises(ModelStructureError, match="not an ADR tagged migration"):
        _check_baseline_reconciliation(a, baseline())
    a.stack_entries[0].migration_adr_id = d.decision_id  # grounded entry may not cite one
    with pytest.raises(ModelStructureError, match="requires empty baseline_refs"):
        _check_baseline_reconciliation(a, baseline())


def test_greenfield_is_explicit_and_cannot_claim_baseline_or_migration():
    b = baseline("greenfield")
    a = ArchitectureOverview.model_validate({**assign_architecture_ids({
        "stack_entries": [{"key": "language", "value": "Python", "baseline_refs": [], "migration_adr_id": None}],
        "components": [{"name": "app", "responsibility": "serve", "baseline_refs": [], "migration": False, "migration_adr_title": None}],
        "decisions": [{"title": "Language", "decision": "Python", "rationale": "team fit",
                       "driver": "team_fit", "options_considered": [], "baseline_refs": [],
                       "migration": False, "evidence": [], "affected_repositories": []}],
        "non_functionals": {}}), "session_id": "s", "baseline_id": "vmodel-v1"})
    _check_baseline_reconciliation(a, b)
    a.decisions[0].migration = True
    with pytest.raises(ModelStructureError, match="not a migration"):
        _check_baseline_reconciliation(a, b)


def test_architecture_grammar_requires_grounding_fields():
    sch = for_stage("architecture")
    assert "baseline_id" not in sch["properties"]  # pipeline owns it
    entry = sch["properties"]["stack_entries"]["items"]
    assert {"key", "value"} <= set(entry["required"])
    assert "migration_adr_id" not in entry["properties"]  # pipeline wires it at assembly
    comp = sch["properties"]["components"]["items"]
    assert {"baseline_refs", "migration", "migration_adr_title"} <= set(comp["required"])
    dec = sch["properties"]["decisions"]["items"]
    assert {"baseline_refs", "migration", "evidence", "affected_repositories"} <= set(dec["required"])
