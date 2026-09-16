import pytest
from vmodel_contracts import ArchitectureOverview, ConOpsDocument
from spec_generator.graph import _check_conops_execution
from spec_generator.skeleton import ModelStructureError

def conops(selected="Native app"):
 return ConOpsDocument.model_validate({"session_id":"s","stakeholder_context":[{"stakeholder":"u","role":"r","goals":["g"],"operating_environment":"e","source_turns":[1]}],"operational_scenarios":[{"scenario_id":"OS-1","name":"n","trigger":"t","actors":["u"],"operational_flow":["x"],"success_outcome":"y","source_turns":[1]}],"concept_options":[{"name":"Native app","summary":"s","advantages":["a"],"disadvantages":["d"]},{"name":"PWA","summary":"s","advantages":["a"],"disadvantages":["d"]}],"decision":{"selected_concept":selected,"rationale":"chosen","founder_confirmed":True,"source_turns":[1]}}).model_dump(mode="json")

def arch(executes=True):
 d={"decision_id":"ADR-1","title":"Mobile","decision":"React Native","rationale":"fit","driver":"simplicity","options_considered":[],"executes_concept_ids":["concept:native-app"] if executes else [],"technology_selections":[{"slot_id":"slot:mobile-framework","technology_id":"tech:react-native","display_name":"React Native"}]}
 return ArchitectureOverview.model_validate({"session_id":"s","stack_entries":[{"key":"mobile-framework","value":"React Native","technology_id":"tech:react-native","decision_id":"ADR-1"}],"components":[],"decisions":[d],"non_functionals":{}})

def test_architecture_must_execute_founder_confirmed_concept():
 _check_conops_execution(arch(),conops())
 with pytest.raises(ModelStructureError,match="concept id"):_check_conops_execution(arch(False),conops())


def test_confirmed_conops_creates_code_owned_execution_slot():
 from spec_generator.graph import _required_concept_adr_slots
 slots=_required_concept_adr_slots(conops())
 assert slots == [{"slug":"execute-native-app","migration":False,"authorizes":[],"required_concept_ids":["concept:native-app"]}]

def test_founder_constraint_can_be_confirmed_conops_not_tech_preference():
 from spec_generator.graph import _check_decision_drivers
 data=arch(True).model_dump(mode="json")
 data["decisions"][0]["driver"]="founder_constraint"
 from vmodel_contracts import ArchitectureOverview
 _check_decision_drivers(ArchitectureOverview.model_validate(data),{"tech_preferences":[]},conops())
