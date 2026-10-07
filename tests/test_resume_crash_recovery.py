import copy
import json
import sqlite3
import sys

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from vmodel_contracts import CONTRACT_VERSION, ConOpsDocument, SystemBaseline

from conftest import ModelFactoryLedger, SchemaRoutingArchitectureModel
from test_progress import INTENT, FakeModel
from test_from_rerun import SYSTEM_BASELINE
from test_conops_consumption import conops
from test_resume_version_stamps import restamp
from spec_generator import cli, model_config
from spec_generator.artifact_lifecycle import ArchitectureLifecycle, stable_hash
from spec_generator.errors import UserFacingError
from spec_generator.graph import build_graph, checkpoint_serde
from spec_generator.telemetry import UsageLog


@pytest.mark.parametrize("recorded_resume", [False, True])
@pytest.mark.parametrize("legacy_anchor", [False, True])
@pytest.mark.parametrize("reconnected", [False, True])
def test_cli_crash_retry_reseals_without_repeating_architecture(tmp_path,monkeypatch,legacy_anchor,reconnected,recorded_resume):
 ledger=ModelFactoryLedger(SchemaRoutingArchitectureModel);box={}
 def factory(stage,**kw):
  if stage=="architecture":return ledger.build(stage,kw.get("schema"))
  return box.setdefault(stage,FakeModel(stage))
 monkeypatch.setattr(model_config,"build_chat_model",factory)
 from vmodel_contracts import runtime
 monkeypatch.setattr(runtime,"install_plain_log",lambda path,stream:stream)
 outdir=tmp_path/"spec";outdir.mkdir()
 current=SystemBaseline.model_validate(SYSTEM_BASELINE).model_dump(mode="json")
 old=restamp(current,"0.19.0");new=restamp(current,CONTRACT_VERSION)
 cfg={"configurable":{"thread_id":"s"}}
 conn=sqlite3.connect(outdir/"checkpoints.sqlite",check_same_thread=False)
 first=build_graph(UsageLog("s"),checkpointer=SqliteSaver(conn,serde=checkpoint_serde()),stop_after="architecture")
 state=first.invoke({"intent_model":dict(INTENT),"system_baseline":old},cfg)
 body={**state["architecture"],"session_id":"s"}
 cli.validated_architecture(body)
 life=ArchitectureLifecycle(outdir,"s",old);life.bootstrap_complete(body,old)
 if legacy_anchor:life.anchor_hash=stable_hash(new)
 if recorded_resume:life.resume("items")
 if reconnected:
  full=build_graph(UsageLog("s"),checkpointer=SqliteSaver(conn,serde=checkpoint_serde()))
  cli._resume_after_bounded_stop(full,cfg,state)
 before=life.path.read_bytes();arch_calls=ledger.calls("architecture")
 intent=tmp_path/"intent.json";intent.write_text(json.dumps(dict(INTENT,contract_version=CONTRACT_VERSION)))
 baseline=tmp_path/"baseline.json";baseline.write_text(json.dumps(new))
 monkeypatch.setattr(sys,"argv",["cli",str(intent),"--system-baseline",str(baseline),"--out",str(outdir)])
 cli.main()
 assert ledger.calls("architecture")==arch_calls
 assert life.path.read_bytes().startswith(before)
 events=[json.loads(line) for line in life.path.read_text().splitlines()]
 assert sum(e["transition"]=="resume" for e in events)==(1 if recorded_resume or not reconnected else 0)
 assert events[-1]["transition"]=="finalize"
 assert events[-1]["anchor_hash"]==stable_hash(old)
 assert json.loads((outdir/"manifest.json").read_text())
 counts={stage:model.calls for stage,model in box.items()}
 cli.main()
 assert ledger.calls("architecture")==arch_calls
 assert {stage:model.calls for stage,model in box.items()}==counts
 conn.close()


def test_incompatible_conops_does_not_reconnect_or_append_lifecycle(tmp_path,monkeypatch):
 from vmodel_contracts import runtime
 monkeypatch.setattr(runtime,"install_plain_log",lambda path,stream:stream)
 outdir=tmp_path/"spec";outdir.mkdir()
 conn=sqlite3.connect(outdir/"checkpoints.sqlite",check_same_thread=False)
 saver=SqliteSaver(conn,serde=checkpoint_serde())
 graph=build_graph(UsageLog("s"),checkpointer=saver)
 cfg={"configurable":{"thread_id":"s"}}
 graph.update_state(cfg,{"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE,
                         "conops":conops(),"architecture":{"completed":True}},as_node="architecture")
 snapshot=graph.get_state(cfg)
 life=ArchitectureLifecycle(outdir,"s",SYSTEM_BASELINE)
 life.bootstrap_complete({"completed":True},SYSTEM_BASELINE)
 log_before=life.path.read_bytes()
 db_before=list(conn.execute("SELECT * FROM checkpoints"))
 writes_before=list(conn.execute("SELECT * FROM writes"))
 intent=tmp_path/"intent.json";intent.write_text(json.dumps(dict(INTENT,contract_version=CONTRACT_VERSION)))
 baseline=tmp_path/"baseline.json";baseline.write_text(json.dumps(SYSTEM_BASELINE))
 changed=conops();changed["decision"]["rationale"]="changed"
 concept=tmp_path/"conops.json";concept.write_text(json.dumps(changed))
 monkeypatch.setattr(sys,"argv",["cli",str(intent),"--conops",str(concept),"--system-baseline",str(baseline),"--out",str(outdir)])
 with pytest.raises(UserFacingError,match="Concept of Operations"):cli.main()
 assert list(conn.execute("SELECT * FROM checkpoints"))==db_before
 assert list(conn.execute("SELECT * FROM writes"))==writes_before
 assert graph.get_state(cfg).config==snapshot.config
 assert life.path.read_bytes()==log_before
 conn.close()
