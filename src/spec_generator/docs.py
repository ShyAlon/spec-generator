"""Deterministic human-readable views. Never author or repair source artifacts."""
from __future__ import annotations
import difflib
import hashlib
import json
import os
import re
from pathlib import Path

STAGES = ('needs', 'srs', 'architecture', 'items', 'tests', 'validation', 'review', 'assemble')

def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)

def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()

def _atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text, encoding='utf-8'); os.replace(temp, path)

def _read(path, default=None):
    if not path.is_file(): return default
    try: return json.loads(path.read_text())
    except (ValueError, OSError): return default

def _md(value, depth=2):
    """Lossless readable structure, including unknown future contract fields."""
    if isinstance(value, dict):
        parts=[]
        for k,v in value.items():
            label=str(k).replace('_',' ')
            if isinstance(v,(dict,list)):
                parts.append('#' * min(depth,6)+' '+label+'\n\n'+_md(v,depth+1))
            else:parts.append('- **'+label+':** '+(str(v) if v is not None else '(not supplied)'))
        return '\n\n'.join(parts)
    if isinstance(value, list):
        return '\n\n'.join(_md(x, depth) if isinstance(x, (dict, list)) else '- ' + str(x) for x in value) or '(none)'
    return str(value) if value is not None else '(not supplied)'

def _safe_id(value, prefix):
    if not re.fullmatch(prefix+r'-[1-9][0-9]*', str(value)):
        raise ValueError('Unsafe document identifier: '+str(value))
    return str(value)

def _anchor(identifier): return '<a id="' + str(identifier).lower() + '"></a>\n'
def _cell(value): return str(value).replace('|', '\\|').replace('\n', '<br>')
def _table(head, rows):
    return '| ' + ' | '.join(head) + ' |\n|' + '|'.join('---' for _ in head) + '|\n' + ''.join('| ' + ' | '.join(_cell(x) for x in r) + ' |\n' for r in rows)

def render_docs(sources, status=None, history=None, changes=None, telemetry=None):
    """Return docs-relative Markdown files from canonical values, without I/O."""
    status = status or {}; history = history or []; changes = changes or []; telemetry = telemetry or {}
    files = {}; index = ['# Run documents', '', 'Rendered views, not edit inputs. JSON remains the source of truth. Generation status is not test execution or product-quality approval.', '']
    index += ['## Generation stages', _table(['Stage', 'State'], [(s, status.get(s, 'pending')) for s in STAGES])]
    def emit(name, title, content, stage, source):
        state = status.get(stage, 'available' if sources.get(stage) is not None else 'pending')
        body = '# ' + title + '\n\nState: **' + state + '**. Source: `' + source + '`\n\n'
        value = sources.get(stage)
        if value is not None: body += 'Source SHA-256: `' + _hash(value) + '`\n\n'
        revision=(sources.get('revisions')or{}).get(stage)
        if revision:body+='Revision evidence: '+_json(revision)+'\n\n'
        files[name] = body + content + '\n'
        if not name.startswith(('tests/','specs/')):index.append('- [' + title + '](' + name + ') - ' + state)
    for stage, name, title, source in [('intent','intent.md','Product intent','interview/intent_model.json'),('conops','conops.md','Concept of operations','interview/conops.json'),('needs','needs.md','Stakeholder needs','spec/needs.json'),('architecture','architecture.md','Architecture','spec/architecture.json')]:
        value=sources.get(stage)
        content=_md(value) if value is not None else 'Not available yet.'
        if stage=='needs' and value:
            content='\n\n'.join(_anchor(n['need_id']) + '## ' + n['need_id'] + '\n\n' + _md(n) for n in value.get('needs',[]))
        emit(name,title,content,stage,source)
    srs=sources.get('srs') or {}; reqs=srs.get('functional',[])+srs.get('non_functional',[])
    content=_md({k:v for k,v in srs.items() if k not in ('functional','non_functional')}) if srs else 'Not available yet.'
    for r in reqs:
        content+='\n\n'+_anchor(r['requirement_id'])+'## '+r['requirement_id']+': '+r.get('title','')+'\n\n'+_md({k:v for k,v in r.items() if k!='acceptance'})
        for ac in r.get('acceptance',[]): content+='\n\n'+_anchor(ac['criterion_id'])+'### '+ac['criterion_id']+'\n\n'+_md(ac)
    emit('requirements.md','Requirements (SRS)',content,'srs','spec/srs.json')
    items=sources.get('items') or []; tests=sources.get('tests') or []
    rows=[]
    if items:index+=['','## Item specifications',_table(['Spec','Title','State'],[('[%s](specs/%s.md)'%(i['spec_id'],i['spec_id']),i.get('title',''),status.get('items','available'))for i in items])]
    for item in items:
        sid=_safe_id(item['spec_id'],'SPEC'); name='specs/'+sid+'.md'
        content=_md({k:v for k,v in item.items() if k!='body_markdown'})+'\n\n## Generated body (unchanged)\n\n'+item.get('body_markdown','(missing body)')
        emit(name,sid+': '+item.get('title',''),content,'items','spec/items.json')
    ac_to_req={a['criterion_id']:r['requirement_id'] for r in reqs for a in r.get('acceptance',[])}
    for test in tests:
        tid=_safe_id(test['test_id'],'T'); ac=test.get('criterion_id'); sid=test.get('spec_id'); ex=(test.get('definition')or{}).get('execution')or{}
        readiness='Execution fields supplied; not validated or run' if any(v is not None for k,v in ex.items() if k!='contract_version') else 'No execution fields; not run'
        backlink='[Test index](../tests.md)'
        if ac:backlink+=' | [%s](../requirements.md#%s)'%(ac,ac.lower())
        if sid:backlink+=' | [%s](../specs/%s.md)'%(sid,sid)
        emit('tests/'+tid+'.md',tid+': '+test.get('description',''),backlink+'\n\n'+_md(test)+'\n\n## Execution status\n\n'+readiness,'tests','spec/tests.json')
        links=[]
        if ac: links.append('[%s](requirements.md#%s)'%(ac,ac.lower()))
        if sid: links.append('[%s](specs/%s.md)'%(sid,sid))
        rid=ac_to_req.get(ac)
        if rid: links.insert(0,'[%s](requirements.md#%s)'%(rid,rid.lower()))
        rows.append(('[%s](tests/%s.md)'%(tid,tid),test.get('level',''),', '.join(links),test.get('runner',''),readiness))
    emit('tests.md','Test specifications',_table(['Test','Level','Links','Runner','Readiness'],rows) if rows else 'Not available yet.','tests','spec/tests.json')
    vitems=(sources.get('validation') or {}).get('items') or []
    vrows=[('<a id="%s"></a>%s'%(v['validation_id'].lower(),v['validation_id']),', '.join('[%s](needs.md#%s)'%(n,n.lower()) for n in v.get('need_ids',[])),
            ('[%s](requirements.md#%s)'%(v['criterion_id'],v['criterion_id'].lower())) if v.get('criterion_id') else '',
            ', '.join('[%s](tests/%s.md)'%(t,t) for t in v.get('verified_by',[])),', '.join(v.get('components',[])),
            v.get('what_the_human_does',''),v.get('what_to_observe',''),v.get('pass_when',''),v.get('evidence',''),v.get('environment','')) for v in vitems]
    emit('validation.md','Validation (human judgment)',_table(['Item','Need','Criterion','Machine proxy','Components','What the person does','What to observe','Pass when','Evidence','Environment'],vrows) if vrows else 'Not available yet.','validation','spec/validation.json')
    trace=sources.get('traceability'); trace_rows=[]
    item_ids={i['spec_id'] for i in items}; test_ids={t['test_id'] for t in tests}
    for r in reqs:
        rid=r['requirement_id']; entry=next((t for t in (trace or []) if t['requirement_id']==rid),None)
        specs=entry.get('spec_ids',[]) if entry else [i['spec_id'] for i in items if rid in i.get('requirement_ids',[])]
        acs=[a['criterion_id'] for a in r.get('acceptance',[])]
        tids=entry.get('test_ids',[]) if entry else [t['test_id'] for t in tests if t.get('criterion_id') in acs or t.get('spec_id') in specs]
        gaps=[]
        if not specs:gaps.append('no spec link')
        if set(specs)-item_ids:gaps.append('missing spec target')
        if set(tids)-test_ids:gaps.append('missing test target')
        if any(not any(t.get('criterion_id')==ac for t in tests)for ac in acs):gaps.append('AC without test')
        trace_rows.append((', '.join('[%s](needs.md#%s)'%(n,n.lower()) for n in r.get('need_ids',[])), '[%s](requirements.md#%s)'%(rid,rid.lower()),', '.join('[%s](requirements.md#%s)'%(ac,ac.lower())for ac in acs),', '.join('[%s](specs/%s.md)'%(s,s)for s in specs),', '.join('[%s](tests/%s.md)'%(t,t)for t in tids),'; '.join(gaps)or 'Links resolve; semantic coverage NOT verified'))
    emit('traceability.md','Traceability',('Assembled matrix.' if trace is not None else 'Provisional links derived from current artifacts; not assembled.')+'\n\n'+_table(['Need','Requirement','AC','Spec','Test','Gaps'],trace_rows),'assemble','spec/traceability.json')
    for record in history:
        n=record['round']
        if not isinstance(n,int) or n<1:raise ValueError('Invalid review round')
        content='Findings are proposals, not founder decisions. No automatic claim of resolution.\n\n'+_md(record)
        emit('review/round-%s.md'%n,'Review round %s'%n,content,'review','spec/review-history.json')
    for n in (1,2):
        if 'review/round-%s.md'%n not in files:emit('review/round-%s.md'%n,'Review round %s'%n,'Not recorded yet. Historical findings cannot be reconstructed from missing evidence.','review','spec/review-history.json')
    led=sources.get('findings_ledger')
    if led is not None:
        rows=[(f['finding_id'],f['state'],f['rule_id'],f['kind'],', '.join(f['targets']),f['summary']) for f in (led.get('findings') or {}).values()]
        open_n=sum(1 for r in rows if r[1]=='OPEN')
        emit('findings.md','Findings ledger',('Assembly and handoff are BLOCKED while any finding is OPEN. ' if open_n else 'No open findings. ')+'Semantic findings close only by founder disposition; absence of a finding is not proof of absence of a defect.\n\n'+_table(['Finding','State','Rule','Kind','Targets','Summary'],rows)+'\n\n'+_md({'incomplete_checks':led.get('incomplete') or []}),'review','spec/findings-ledger.json')
    emit('changes.md','Artifact changes',_md(changes) if changes else 'No comparisons recorded yet. Existing files are not historical snapshots.','review','spec/docs-state.json')
    emit('run.md','Run diagnostics',_md(telemetry)+'\n\n## Open issues\n\n'+_md(sources.get('violations',[])),'assemble','spec/usage.jsonl, llm-calls.jsonl, run.log')
    index+=['','## Open findings / warnings',_md(sources.get('violations',[])), '', 'Read review rounds and run diagnostics before treating the bundle as build-ready.']
    files['README.md']='\n\n'.join(index)+'\n'
    return files


def _root(spec):
    return spec.parent if spec.name == 'spec' else spec

def refresh_docs(spec, intent=None, conops=None, event=None, state=None, invalidate=None, start=False):
    from .stop_fence import guarded_publish
    return guarded_publish(lambda:_refresh_docs(spec,intent,conops,event,state,invalidate,start))


def _refresh_docs(spec, intent=None, conops=None, event=None, state=None, invalidate=None, start=False):
    """Single writer for incremental, resume, explicit-stage and final output."""
    spec=Path(spec); root=_root(spec); docs=root/'docs'; ledger=_read(spec/'docs-state.json',{})
    stages=ledger.setdefault('status',{}); snapshots=ledger.setdefault('snapshots',{}); changes=ledger.setdefault('changes',[])
    sources={s:_read(spec/(s+'.json')) for s in ('needs','srs','architecture','items','tests','validation','traceability')}
    sources['findings_ledger']=_read(spec/'findings-ledger.json')
    sources['intent']=intent or _read(root/'interview/intent_model.json');sources['conops']=conops or _read(root/'interview/conops.json') or _read(spec/'conops.json')
    if state is not None:
        for s in ('needs','srs','architecture','items','tests','validation'):
            if sources[s] is None and state.get(s) is not None: sources[s]=state[s]
    if start:
        for s in STAGES:stages[s]='available (prior output; not validated for this invocation)' if sources.get(s) is not None or s=='assemble' and sources.get('traceability') is not None or s=='review' and history_exists(spec) else 'pending'
    if invalidate:
        for s in STAGES[STAGES.index(invalidate):]: stages[s]='stale' if sources.get(s) is not None else 'pending'
    artifacts=_read(spec/'artifact-status.json',{}).get('artifacts',{})
    for s in ('items','tests','assemble'):
        if any(artifacts.get(k,{}).get('state')=='stale' for k in ((s+'.json',) if s!='assemble' else ('manifest.json','traceability.json'))):stages[s]='stale'
    if event:
        stages[event]='generated (not quality-approved)'
        if event=='assemble': stages[event]='assembled (not quality-approved)'
        if event in sources and sources[event] is not None:
            value=sources[event];old=snapshots.get(event)
            if old is not None:
                same=old==value
                if not same and event in STAGES:
                    for downstream in STAGES[STAGES.index(event)+1:]:
                        stages[downstream]='stale' if sources.get(downstream) is not None or downstream=='review' and history_exists(spec) else 'pending'
                if (state or {}).get('review_round') or not same:
                    changes.append({'artifact':event,'after_review_round':(state or {}).get('review_round'), 'result':'NO-OP: content unchanged' if same else 'changed','before_sha256':_hash(old),'after_sha256':_hash(value),'diff':''.join(difflib.unified_diff(_json(old).splitlines(True),_json(value).splitlines(True),fromfile='before/'+event,tofile='after/'+event))})
            snapshots[event]=value
    if state and state.get('violations'):sources['violations']=state['violations']
    elif ledger.get('violations'):sources['violations']=ledger['violations']
    ledger['violations']=sources.get('violations',[])
    history=_read(spec/'review-history.json',[])
    usage=[];calls=[]
    for filename,rows in [('usage.jsonl',usage),('llm-calls.jsonl',calls)]:
        path=spec/filename
        if path.exists():
            for line in path.read_text().splitlines():
                try:rows.append(json.loads(line))
                except ValueError:pass
    telemetry={'stage_status':stages,'usage_rows':len(usage),'call_starts':sum(x.get('event')=='llm_call_start' for x in calls),'call_ends':sum(x.get('event')=='llm_call_end' for x in calls),'call_errors':sum(x.get('event')=='llm_call_error' for x in calls),'prompt_tokens':sum(x.get('prompt_tokens',0)for x in usage),'output_tokens':sum(x.get('completion_tokens',0)for x in usage),'recorded_cost_usd':sum((x.get('prompt_tokens',0)*x.get('price_per_mtok_prompt_usd',0)+x.get('completion_tokens',0)*x.get('price_per_mtok_completion_usd',0))/1e6 for x in usage),'cost_caveat':'Recorded rates only; zero-priced local compute and unmetered failed calls are excluded. Tests have not been run by this renderer.','sources':['../spec/run.log','../spec/rejected-llm-outputs/manifest.jsonl']}
    log=spec/'run.log'
    telemetry['self_heal_messages']=[line for line in log.read_text().splitlines() if 'WARNING self-heal' in line] if log.exists() else []
    telemetry['rejected_outputs']=_read_rejections(spec)
    telemetry['per_stage']={}
    for stage in sorted({x.get('stage','unknown')for x in usage+calls}):
        uu=[x for x in usage if x.get('stage')==stage];cc=[x for x in calls if x.get('stage')==stage]
        telemetry['per_stage'][stage]={'usage_rows':len(uu),'call_starts':sum(x.get('event')=='llm_call_start'for x in cc),'call_ends':sum(x.get('event')=='llm_call_end'for x in cc),'prompt_tokens':sum(x.get('prompt_tokens',0)for x in uu),'output_tokens':sum(x.get('completion_tokens',0)for x in uu)}
    telemetry['artifact_revisions']={s:_read(spec/(s+'.revision.json')) for s in ('srs','architecture')}
    sources['revisions']=telemetry['artifact_revisions']
    files=render_docs(sources,stages,history,changes,telemetry)
    previous=ledger.get('generated_files',[])
    for name in previous:
        if name not in files and name.startswith(('specs/','tests/')) and '..' not in Path(name).parts:
            (docs/name).unlink(missing_ok=True)
    for name,text in files.items():_atomic(docs/name,text)
    ledger['generated_files']=list(files);_atomic(spec/'docs-state.json',_json(ledger))
    # Never overwrite an existing hand-authored root README.
    readme=root/'README.md'
    if not readme.exists():_atomic(readme,'# Run output\n\nStart with [the readable document index](docs/README.md).\n')
    return files

def _read_rejections(spec):
    path=spec/'rejected-llm-outputs/manifest.jsonl';rows=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                x=json.loads(line);rows.append({k:v for k,v in x.items() if k in ('stage','item','attempt','reason','path','ts')})
            except ValueError:pass
    return rows

def history_exists(spec):
    return bool(_read(Path(spec)/'review-history.json', []))

def archive_review(spec, record, intent=None):
    from .stop_fence import guarded_publish
    return guarded_publish(lambda:_archive_review(spec,record,intent))


def _archive_review(spec, record, intent=None):
    state=record.get('state');record={k:v for k,v in record.items() if k!='state'}
    history=_read(Path(spec)/'review-history.json',[])
    # Stable round identity makes a resume retry idempotent.
    history=[x for x in history if x['round']!=record['round']]+[record]
    history.sort(key=lambda x:x['round']);_atomic(Path(spec)/'review-history.json',_json(history))
    refresh_docs(spec,intent=intent,event='review',state=state)
