"""Bound host documents retain exact partial bytes without local files."""
import json
import pytest
from spec_generator.worker_runtime import job_runtime
from test_revision_worker import setup


def test_partials_use_bound_host_document_store_and_clear_without_files(tmp_path,monkeypatch):
    from spec_generator.cli import _make_partials
    cap,root,runtime,cid,state=setup(tmp_path);documents={}
    class Documents:
        def read(self,key):return documents.get(key)
        def write(self,key,raw):documents[key]=raw
        def clear(self,key):documents.pop(key,None)
    runtime.documents=Documents()
    with job_runtime(runtime):
        partials=_make_partials(root/'spec')
        assert partials.load('srs_fills') is None
        partials.save('srs_fills',{'done':['R-1']})
        assert partials.load('srs_fills')=={'done':['R-1']}
        assert json.loads(documents['spec/srs_fills.partial.json'])=={'done':['R-1']}
        assert not (root/'spec/srs_fills.partial.json').exists()
        partials.clear('srs_fills');assert partials.load('srs_fills') is None


def test_document_paths_cannot_escape_bound_runtime(tmp_path):
    from spec_generator.document_runtime import read,write
    cap,root,runtime,cid,state=setup(tmp_path);calls=[]
    runtime.documents=type('Docs',(),{'read':lambda self,key:calls.append(key),'write':lambda *a:calls.append(a)})()
    with job_runtime(runtime):
        with pytest.raises(ValueError):read(root/'spec/../outside')
        with pytest.raises(ValueError):write(tmp_path/'foreign',b'{}')
    assert calls==[]


def test_bound_lifecycle_lineage_and_docs_have_no_file_authority(tmp_path,monkeypatch):
    from spec_generator import lineage
    from spec_generator.cli import _atomic_write
    from spec_generator.artifact_lifecycle import ArchitectureLifecycle
    from spec_generator.docs import refresh_docs,archive_review
    cap,root,runtime,cid,state=setup(tmp_path);documents={}
    class Documents:
        def read(self,key):return documents.get(key)
        def write(self,key,raw):documents[key]=raw
        def clear(self,key):documents.pop(key,None)
        def append(self,key,raw):documents[key]=documents.get(key,b'')+raw
    runtime.documents=Documents();spec=root/'spec';anchor={'id':'baseline'}
    with job_runtime(runtime):
        _atomic_write(spec/'needs.json',json.dumps({'needs':[]}))
        _atomic_write(root/'interview/intent_model.json',json.dumps({'session_id':'s'}))
        life=ArchitectureLifecycle(spec,'s',anchor)
        life.begin_or_checkpoint({'partial':1});event=life.finalize({'complete':1});life.commit_finalize(event)
        assert life.state.value=='complete'
        assert len(documents['spec/artifact-lifecycle.jsonl'].splitlines())==2
        sha=lineage.sha256_file(spec/'needs.json')
        lineage.bind(spec,task_id='t',session_id='s',generation=1,phase='needs',name='needs.json',sha256=sha,source='stage-commit')
        record=lineage.load(spec)
        assert lineage.verify_file(record,phase='needs',name='needs.json',path=spec/'needs.json')['sha256']==sha
        documents['spec/needs.json']=b'{"needs":[1]}'
        with pytest.raises(lineage.LineageRefusal,match='changed'):lineage.verify_file(record,phase='needs',name='needs.json',path=spec/'needs.json')
        documents['spec/needs.json']=b'{"needs":[]}'
        archive_review(spec,{'round':1,'findings':[]})
        files=refresh_docs(spec,event='needs')
        assert documents['docs/needs.md'].decode()==files['needs.md']
        assert json.loads(documents['spec/review-history.json'])[0]['round']==1
        assert not (spec/'needs.json').exists()
        assert not (spec/'lineage.json').exists()
        assert not (spec/'artifact-lifecycle.jsonl').exists()
        assert not (root/'docs').exists()


def test_bound_grounding_cache_and_content_evidence_survive_new_readers(tmp_path):
    from spec_generator.grounding import GroundingCache
    from spec_generator.cli import _make_persist
    cap,root,runtime,cid,state=setup(tmp_path);documents={}
    class Documents:
        def read(self,key):return documents.get(key)
        def write(self,key,raw):documents[key]=raw
        def clear(self,key):documents.pop(key,None)
    runtime.documents=Documents();path=root/'spec/grounding-cache.json'
    with job_runtime(runtime):
        GroundingCache(path).put('first',{'answer':'yes'})
        assert GroundingCache(path).get('first')=={'answer':'yes'}
        persist=_make_persist({'session_id':'s'},root/'spec')
        persist.content_evidence({'rejected':1});persist.content_evidence({'rejected':2})
        persist.grounding('SPEC-1',{'hash':'first'});persist.grounding('SPEC-2',{'hash':'second'})
        assert json.loads(documents['spec/content-rejections.json'])==[{'rejected':1},{'rejected':2}]
        assert set(json.loads(documents['spec/grounding-evidence.json']))=={'SPEC-1','SPEC-2'}
        assert not path.exists()


def test_pg_baseline_self_copy_preserves_exact_approved_bytes(tmp_path):
    from spec_generator.cli import _persist_system_baseline
    cap,root,runtime,cid,state=setup(tmp_path)
    raw=b'{"baseline_id":"approved"}\n';documents={'spec/system-baseline.json':raw}
    class Documents:
        def read(self,key):return documents.get(key)
        def write(self,key,value):documents[key]=value
    runtime.documents=Documents()
    with job_runtime(runtime):
        _persist_system_baseline(root/'spec/system-baseline.json',root/'spec',{'baseline_id':'approved','normalized_default':[]})
    assert documents=={'spec/system-baseline.json':raw}
    assert not (root/'spec/system-baseline.json').exists()


def test_bound_cli_usage_sink_retains_exact_rows_without_local_copy(tmp_path):
    from spec_generator.telemetry import jsonl_sink
    from types import SimpleNamespace
    cap,root,runtime,cid,state=setup(tmp_path);documents={}
    class Documents:
        def append(self,key,raw):documents[key]=documents.get(key,b'')+raw
    runtime.documents=Documents();raw='{"stage":"needs","prompt_tokens":3}'
    with job_runtime(runtime):
        sink=jsonl_sink(root/'spec/usage.jsonl')
        sink(SimpleNamespace(model_dump_json=lambda:raw));sink(SimpleNamespace(model_dump_json=lambda:raw))
    assert documents=={'spec/usage.jsonl':(raw+'\n'+raw+'\n').encode()}
    assert not (root/'spec/usage.jsonl').exists()
