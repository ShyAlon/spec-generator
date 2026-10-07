import json
from pathlib import Path
import pytest
from spec_generator.graph import _check_adr_driver_source,_decided_founder_sources
from spec_generator.schemas import arch_adr_body_schema
from spec_generator.graph import ModelStructureError

SOURCE={'item_id':'FD-missing-error_states','status':'decided','decision':'A clear error message with a "Tell us about it" button.','reason':"This way we'll be notified of bugs and improve."}
INTENT={'tech_preferences':[],'founder_register':{'items':[SOURCE]}}


@pytest.mark.parametrize('file',['adr17-error-surface-a.json','adr17-error-surface-b.json'])
def test_raw_failure_needs_real_register_link(file):
    body=json.loads((Path(__file__).parent/'fixtures'/file).read_text())
    with pytest.raises(ModelStructureError):_check_adr_driver_source(body,INTENT)
    body['founder_decision_refs']=['FD-missing-error_states']
    _check_adr_driver_source(body,INTENT)
    for status in ('open','deferred'):
        with pytest.raises(ModelStructureError):_check_adr_driver_source(body,{'founder_register':{'items':[{**SOURCE,'status':status}]}})
    with pytest.raises(ModelStructureError):_check_adr_driver_source(body,{'founder_register':{'items':[{**SOURCE,'decision':'An unrelated decision.'}]}})


def test_schema_enumerates_decided_refs_and_old_body_defaults():
    schema=arch_adr_body_schema([],[],[],list(_decided_founder_sources(INTENT)))
    assert schema['properties']['founder_decision_refs']['items']['enum']==['FD-missing-error_states']
    from vmodel_contracts import ArchitectureDecision
    body=json.loads((Path(__file__).parent/'fixtures'/'adr17-error-surface-a.json').read_text())
    old=ArchitectureDecision.model_validate({'decision_id':'ADR-17',**body})
    assert old.founder_decision_refs==[]


def test_invalid_and_blank_sources_stay_out_of_schema():
    for changes in ({'decision':''},{'reason':''},{'status':'decide_later'}):
        assert not _decided_founder_sources({'founder_register':{'items':[{**SOURCE,**changes}]}})
