import json,sys
from pathlib import Path
from spec_generator import cli,model_config
from vmodel_contracts import CONTRACT_VERSION


def test_offline_real_cli_replay_and_resume(monkeypatch,tmp_path,capsys):
    intent={"contract_version":CONTRACT_VERSION,"session_id":"supply","problem_statement":"Identify damaged or delayed shipments beforehand","target_users":["Supply chain manager"],"core_value_proposition":"Predict shipment damage and delays","stakeholder_needs":[
      {"need_id":"SN-1","statement":"Identify beforehand which orders and shipments are going to be damaged or delayed en route.","kind":"functional","source_turns":[1]},
      {"need_id":"SN-2","statement":"Integrate real-time disruption data.","kind":"functional","source_turns":[2]},
      {"need_id":"SN-3","statement":"Send risk notifications.","kind":"functional","source_turns":[3]},
      {"need_id":"SN-4","statement":"Calculate event risk levels.","kind":"functional","source_turns":[4]},
      {"need_id":"SN-5","statement":"Handle daily data and scale horizontally.","kind":"non_functional","source_turns":[5]}],"in_scope":["risk"],"out_of_scope":[],"external_integrations":[],"constraints":[],"open_questions":[],"source_turns":{}}
    ip=tmp_path/"intent.json";ip.write_text(json.dumps(intent))
    bp=tmp_path/"baseline.json";bp.write_text(json.dumps({"contract_version":CONTRACT_VERSION,"baseline_id":"greenfield-supply","system_name":"Supply","mode":"greenfield","repositories":[]}))
    fixture=Path(__file__).parent/"fixtures/replay-supply-chain"
    monkeypatch.setenv("VMODEL_REPLAY_DIR",str(fixture));monkeypatch.delenv("SPEC_GENERATOR_BULK",raising=False);model_config._DOTENV=None
    out=tmp_path/"out";args=["prog",str(ip),"--system-baseline",str(bp),"--review","--to","srs","-o",str(out)]
    monkeypatch.setattr(sys,"argv",args);cli.main()
    needs=json.loads((out/"needs.json").read_text())
    assert len(needs["needs"])==5 and all(n["intent_need_ids"] for n in needs["needs"])
    # same command resumes without consuming another recorded response
    monkeypatch.setattr(sys,"argv",args);cli.main()
    assert "Checkpoint found - resuming" in capsys.readouterr().err
