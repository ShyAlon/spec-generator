from test_review_fixes_and_persistence import queue_models
import json
from pathlib import Path
from spec_generator.docs import refresh_docs, archive_review, render_docs


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True);path.write_text(json.dumps(value))


def sample():
    return {'needs':{'needs':[{'need_id':'N-1','statement':'Learn notes'}]},
        'srs':{'purpose':'Music','functional':[{'requirement_id':'R-1','title':'Notes','need_ids':['N-1'],'acceptance':[{'criterion_id':'AC-1','statement':'Correct notes'}]}]},
        'items':[{'spec_id':'SPEC-1','title':'Music','requirement_ids':['R-1'],'body_markdown':'Unrelated output must remain visible.'}],
        'tests':[{'test_id':'T-1','level':'acceptance','criterion_id':'AC-1','spec_id':None,"components": ["api"], "runner": "static",'description':'Check','definition':{'preconditions':['Ready'],'steps':['Play'],'expected_outcomes':['Correct'],'cleanup':[],'cases':[],'execution':{}}}]}


def test_start_index_and_complete_layout(tmp_path):
    spec=tmp_path/'spec';spec.mkdir()
    refresh_docs(spec,intent={'session_id':'x'},start=True)
    assert 'pending' in (tmp_path/'docs/README.md').read_text()
    assert 'docs/README.md' in (tmp_path/'README.md').read_text()
    for name in ['README.md','intent.md','conops.md','needs.md','requirements.md','architecture.md','tests.md','traceability.md','review/round-1.md','review/round-2.md','changes.md','run.md']:
        assert (tmp_path/'docs'/name).is_file()


def test_links_bad_content_and_test_status():
    files=render_docs(sample())
    assert 'Unrelated output must remain visible.' in files['specs/SPEC-1.md']
    assert '<a id="r-1">' in files['requirements.md']
    assert '<a id="ac-1">' in files['requirements.md']
    assert 'requirements.md#ac-1' in files['tests.md']
    assert 'No execution fields; not run' in files['tests/T-1.md']
    assert 'semantic coverage NOT verified' in files['traceability.md']
    assert 'No comparisons recorded' in files['changes.md']


def test_review_archive_idempotent_full_findings_and_no_state_dump(tmp_path):
    spec=tmp_path/'spec';spec.mkdir()
    record={'round':1,'all_chunk_findings':[{'fix':str(i)} for i in range(40)],'retained_fixes':[], 'state':{'private':'no'}}
    archive_review(spec,record);archive_review(spec,record)
    history=json.loads((spec/'review-history.json').read_text())
    assert len(history)==1 and len(history[0]['all_chunk_findings'])==40
    assert 'state' not in history[0]
    assert '39' in (tmp_path/'docs/review/round-1.md').read_text()


def test_changes_staleness_and_generated_file_cleanup(tmp_path):
    spec=tmp_path/'spec';spec.mkdir();values=sample()
    for k,v in values.items():write(spec/(k+'.json'),v)
    refresh_docs(spec,event='srs');refresh_docs(spec,event='items');refresh_docs(spec,event='tests')
    refresh_docs(spec,event='items',state={'review_round':1})
    assert 'NO-OP' in (tmp_path/'docs/changes.md').read_text()
    values['srs']['purpose']='Changed';write(spec/'srs.json',values['srs']);refresh_docs(spec,event='srs')
    assert 'stale' in (tmp_path/'docs/specs/SPEC-1.md').read_text()
    assert 'before/srs' in (tmp_path/'docs/changes.md').read_text()
    write(spec/'items.json',[]);refresh_docs(spec,event='items')
    assert not (tmp_path/'docs/specs/SPEC-1.md').exists()
    assert (spec/'tests.json').exists() # source not touched


def test_custom_output_does_not_pollute_parent_or_overwrite_readme(tmp_path):
    spec=tmp_path/'custom';spec.mkdir();(spec/'README.md').write_text('Authored')
    refresh_docs(spec,start=True)
    assert (spec/'docs/README.md').exists()
    assert not (tmp_path/'docs').exists()
    assert (spec/'README.md').read_text()=='Authored'


def test_source_files_unchanged_and_status_flags(tmp_path):
    spec=tmp_path/'spec';spec.mkdir();write(spec/'items.json',sample()['items'])
    write(spec/'artifact-status.json',{'artifacts':{'items.json':{'state':'stale'}}})
    before=(spec/'items.json').read_bytes();refresh_docs(spec)
    assert before==(spec/'items.json').read_bytes()
    assert '**stale**' in (tmp_path/'docs/specs/SPEC-1.md').read_text()


def test_per_test_backlinks_and_usage_per_stage(tmp_path):
    spec=tmp_path/'spec';spec.mkdir()
    for k,v in sample().items():write(spec/(k+'.json'),v)
    (spec/'usage.jsonl').write_text(json.dumps({'stage':'tests','prompt_tokens':12,'completion_tokens':4})+'\n')
    refresh_docs(spec,event='tests')
    assert '../requirements.md#ac-1' in (tmp_path/'docs/tests/T-1.md').read_text()
    assert 'per stage' in (tmp_path/'docs/run.md').read_text()
    assert '12' in (tmp_path/'docs/run.md').read_text()


def test_graph_archives_before_review_partial_clear(tmp_path, queue_models):
    from langgraph.checkpoint.memory import MemorySaver
    from spec_generator.cli import _make_persist, _make_partials
    from spec_generator.graph import build_graph
    from spec_generator.telemetry import UsageLog
    from test_progress import INTENT
    from conftest import SYSTEM_BASELINE
    persist=_make_persist(dict(INTENT),tmp_path)
    graph=build_graph(UsageLog('s'),checkpointer=MemorySaver(),persist=persist,partials=_make_partials(tmp_path))
    graph.invoke({'intent_model':dict(INTENT),'system_baseline':SYSTEM_BASELINE},{'configurable':{'thread_id':'docs'}})
    assert (tmp_path/'review-history.json').is_file()
    assert not (tmp_path/'review.partial.json').exists()
    assert 'all chunk findings' in (tmp_path/'docs/review/round-1.md').read_text()


def test_rejects_unsafe_document_paths():
    import pytest
    values=sample();values['items'][0]['spec_id']='../../outside'
    with pytest.raises(ValueError,match='Unsafe document'):render_docs(values)


def test_start_does_not_label_assembled_prior_output_pending(tmp_path):
    spec=tmp_path/'spec';spec.mkdir();write(spec/'traceability.json',[])
    refresh_docs(spec,start=True)
    assert 'assemble | available (prior output' in (tmp_path/'docs/README.md').read_text()
