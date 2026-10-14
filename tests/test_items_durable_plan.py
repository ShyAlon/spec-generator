import copy
import pytest
from spec_generator.items_resume import prepare_plan, digest, context_digest
from spec_generator.graph import _items_plan_validator
from spec_generator.cli import _make_partials, _PARTIAL_DOWNSTREAM
RAW=[dict(slug='identity',kind='business_rule',title='Identity',requirement_ids=['R-1'],depends_on=[]),dict(slug='board',kind='ui_flow',title='Board',requirement_ids=['R-2'],depends_on=['identity'])]
V=_items_plan_validator(['R-1','R-2'])
class Memory:
 def __init__(self):self.rows={};self.writes=[]
 def load(self,k):return copy.deepcopy(self.rows.get(k))
 def save(self,k,v):self.rows[k]=copy.deepcopy(v);self.writes.append(k)
def setup():
 p=Memory();plan,done=prepare_plan(p,'payload',V,lambda:V(copy.deepcopy(RAW)))
 assert p.writes==['items_plan'] and done=={}
 p.save('items',[plan[0]|{'body_markdown':'Saved unchanged body'}]);return p,plan

def test_reentry_never_replans_and_preserves_body():
 p,plan=setup()
 def forbidden():pytest.fail('paid plan call replayed')
 got,done=prepare_plan(p,'payload',V,forbidden)
 assert got==plan and done['SPEC-1']['body_markdown']=='Saved unchanged body'
 assert p.writes==['items_plan','items']

def test_plan_saved_before_first_body_even_if_no_bodies_yet():
 p=Memory();plan,_=prepare_plan(p,'payload',V,lambda:V(copy.deepcopy(RAW)))
 assert p.rows['items_plan']['plan']==plan
 prepare_plan(p,'payload',V,lambda:pytest.fail('replan'))

def test_legacy_bodies_without_plan_block_before_provider():
 p=Memory();p.rows['items']=[V(RAW)[0]|{'body_markdown':'body'}]
 with pytest.raises(ValueError,match='no durable plan'):prepare_plan(p,'payload',V,lambda:pytest.fail('provider called'))

@pytest.mark.parametrize('key,value',[('title','changed'),('slug','changed'),('kind','ui_flow'),('depends_on',['SPEC-2']),('requirement_ids',['R-2']),('spec_id','SPEC-99'),('body_markdown','')])
def test_partial_metadata_or_empty_body_refused(key,value):
 p,_=setup();p.rows['items'][0][key]=value
 with pytest.raises(ValueError):prepare_plan(p,'payload',V,lambda:pytest.fail('provider called'))

def test_duplicate_completed_id_refused():
 p,_=setup();p.rows['items']*=2
 with pytest.raises(ValueError,match='duplicate'):prepare_plan(p,'payload',V,lambda:pytest.fail('provider called'))

@pytest.mark.parametrize('change',['context','digest','missing_coverage','numbering','dependencies'])
def test_plan_changes_refused_before_provider(change):
 p,_=setup();e=p.rows['items_plan'];payload='payload'
 if change=='context':payload='other payload'
 elif change=='digest':e['plan_sha256']='wrong'
 elif change=='missing_coverage':e['plan'][1]['requirement_ids']=[];e['plan_sha256']=digest(e['plan'])
 elif change=='numbering':e['plan'][1]['spec_id']='SPEC-99';e['plan_sha256']=digest(e['plan'])
 else:e['plan'][1]['depends_on']=['SPEC-99'];e['plan_sha256']=digest(e['plan'])
 with pytest.raises(ValueError):prepare_plan(p,payload,V,lambda:pytest.fail('provider called'))

def test_plan_partial_disk_roundtrip_and_rewind(tmp_path):
 p=_make_partials(tmp_path);prepare_plan(p,'payload',V,lambda:V(copy.deepcopy(RAW)))
 assert (tmp_path/'items_plan.partial.json').is_file()
 prepare_plan(p,'payload',V,lambda:pytest.fail('provider called'))
 for stage in ('srs','architecture','items'):assert 'items_plan' in _PARTIAL_DOWNSTREAM[stage]
 p.clear('items_plan');assert not (tmp_path/'items_plan.partial.json').exists()
