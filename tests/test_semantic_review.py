import pytest
from vmodel_contracts.semantic_review import make_review
from spec_generator.semantic_review import reviewed_srs
from test_progress import INTENT
from test_srs import SRS

def review():
 candidate={**SRS,'session_id':INTENT['session_id'],'intent_model_ref':INTENT['session_id']}
 return make_review(INTENT['session_id'],'srs',candidate,[{'finding_id':'f1','item_id':'R-1','path':['functional',0],'rule':'solution_language','reason':'Contains a widget','original':candidate['functional'][0]}])

def test_accepted_candidate_survives_resume_without_cleaning():
 r=review();d=r.model_dump();d['decisions']=[{'item_id':'R-1','action':'accept','actor':'founder'}]
 result,record=reviewed_srs(INTENT,r.model_dump(),d)
 assert result==r.candidate
 assert record.decisions[0].actor=='founder'

def test_replace_preserves_structure_and_other_session_is_denied():
 r=review();d=r.model_dump();replacement={**r.candidate['functional'][0],'statement':'The system shall show status.'};d['decisions']=[{'item_id':'R-1','action':'replace','replacement':replacement,'actor':'founder'}]
 assert reviewed_srs(INTENT,r.model_dump(),d)[0]['functional'][0]['statement']=='The system shall show status.'
 with pytest.raises(ValueError):reviewed_srs({**INTENT,'session_id':'other'},r.model_dump(),d)
 d['decisions'][0]['replacement']={'statement':'missing IDs'}
 with pytest.raises(ValueError):reviewed_srs(INTENT,r.model_dump(),d)

def test_reviewed_srs_resumes_a_real_checkpoint_without_authoring(monkeypatch,tmp_path):
 import sqlite3,json
 from langgraph.checkpoint.sqlite import SqliteSaver
 from spec_generator.graph import build_graph,checkpoint_serde
 from spec_generator.cli import _predecessor
 from spec_generator.semantic_review import load_reviewed_srs
 from spec_generator.telemetry import UsageLog
 from spec_generator import model_config
 from test_progress import FakeModel
 from conftest import SYSTEM_BASELINE,schema_routing_factory
 r=review();d=r.model_dump();d['decisions']=[{'item_id':'R-1','action':'accept','actor':'founder'}]
 db=sqlite3.connect(tmp_path/'checkpoint.sqlite',check_same_thread=False)
 saver=SqliteSaver(db,serde=checkpoint_serde());cfg={'configurable':{'thread_id':'s'}}
 g=build_graph(UsageLog('s'),checkpointer=saver,stop_after='architecture')
 g.update_state(cfg,{'intent_model':INTENT,'system_baseline':SYSTEM_BASELINE,'srs':r.candidate,'semantic_reviews':{'srs':r.model_dump()},'needs':{'session_id':'s','needs':[{'need_id':'N-1','stakeholder':'founder','statement':'specs','kind':'functional','intent_fields':['problem_statement'],'intent_need_ids':['SN-1']}]}},as_node='srs')
 (tmp_path/'srs.semantic-review.json').write_text(json.dumps(d))
 candidate,record=load_reviewed_srs(INTENT,tmp_path,g.get_state(cfg).values)
 g2=build_graph(UsageLog('s'),checkpointer=saver,stop_after='architecture')
 g2.update_state(cfg,{'srs':candidate,'semantic_reviews':{'srs':record.model_dump()}},as_node=_predecessor('architecture',False))
 calls=[]
 def factory(stage,**kwargs):
  calls.append(stage);assert stage!='srs'
  return schema_routing_factory(stage,kwargs.get('schema'),lambda:FakeModel(stage))
 monkeypatch.setattr(model_config,'build_chat_model',factory)
 result=g2.invoke(None,cfg)
 assert result['srs']==r.candidate and result['architecture'];assert 'srs' not in calls


def test_founder_answers_reach_architecture_prompts(monkeypatch):
 from spec_generator.graph import build_graph
 from spec_generator.telemetry import UsageLog
 from spec_generator import model_config
 from test_progress import FakeModel
 from conftest import SYSTEM_BASELINE,schema_routing_factory
 from vmodel_contracts.founder_register import FounderRegister
 register=FounderRegister(session_id='s',items=[{'item_id':'FD-style','data_class':'styling','question':'Style?','owner':'founder','status':'decided','scope':'UI','source':'Interview','decision':'Use restrained navy typography','reason':'Owner choice'}])
 prompts=[]
 from conftest import SchemaRoutingArchitectureModel
 class ArchModel(SchemaRoutingArchitectureModel):
  def stream(self,messages):
   prompts.extend(str(m.content) for m in messages)
   yield from super().stream(messages)
 monkeypatch.setattr(model_config,'build_chat_model',lambda stage,**kw:ArchModel(stage,kw.get('schema')) if stage=='architecture' else FakeModel(stage))
 result=build_graph(UsageLog('s'),stop_after='architecture').invoke({'intent_model':{**INTENT,'founder_register':register.model_dump()},'system_baseline':SYSTEM_BASELINE},{'configurable':{'thread_id':'founder-answer'}})
 assert result['architecture'];assert any('Use restrained navy typography' in p and 'decided answers are binding' in p for p in prompts)
