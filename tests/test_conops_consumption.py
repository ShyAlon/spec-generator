import pytest
from vmodel_contracts import ArchitectureOverview, ConOpsDocument

from spec_generator.graph import _check_conops_execution
from spec_generator.render import render_architecture
from spec_generator.skeleton import ModelStructureError


def conops(selected="Native app"):
    return ConOpsDocument.model_validate({"session_id": "s",
      "stakeholder_context": [{"stakeholder": "runner", "role": "tracks", "goals": ["speed"],
       "operating_environment": "phone", "source_turns": [1]}],
      "operational_scenarios": [{"scenario_id": "OS-1", "name": "log", "trigger": "done",
       "actors": ["runner"], "operational_flow": ["record"], "success_outcome": "saved", "source_turns": [2]}],
      "concept_options": [{"name": "Native app", "summary": "installed", "advantages": ["APIs"], "disadvantages": ["store"]},
       {"name": "PWA", "summary": "web", "advantages": ["deploy"], "disadvantages": ["limits"]}],
      "decision": {"selected_concept": selected, "rationale": "chosen", "founder_confirmed": True,
                   "source_turns": [5]}}).model_dump(mode="json")


def arch(value="React Native"):
    return ArchitectureOverview.model_validate({"session_id": "s", "stack_entries": [
      {"key": "mobile_framework", "value": value}], "components": [], "decisions": [], "non_functionals": {}})


def test_architecture_must_execute_founder_confirmed_concept():
    _check_conops_execution(arch(), conops())
    with pytest.raises(ModelStructureError, match="does not execute"):
        _check_conops_execution(arch("Progressive Web App"), conops())


def test_legacy_architecture_degrades_loudly_in_review():
    md = render_architecture(arch(), conops_present=False)
    assert "UNVERIFIED: no Concept of Operations artifact" in md
