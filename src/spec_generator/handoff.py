"""Deterministic coding-agent export from completed phase outputs, zero LLM calls."""
from pathlib import Path,PurePosixPath
import hashlib,json
from vmodel_contracts import IntentModel,ConOpsDocument,StakeholderNeedsSet,SRSDocument,ArchitectureOverview,ItemSpec,AcceptanceTest,RequirementSet,SpecManifest,check_compatible
from vmodel_contracts.founder_register import FounderRegister
from vmodel_contracts.semantic_review import SemanticReview
from .assembly import validate_traceability,build_traceability,order_milestones
from .render import render_agents_md,render_spec_file,render_srs,render_needs,render_architecture

def build_handoff(interview:Path,spec:Path)->dict:
    def load(path,model):
        if path.is_symlink() or not path.is_file():raise ValueError(f'Missing or unsafe output: {path.name}')
        if path.stat().st_size>2_000_000:raise ValueError(f'Output exceeds limit: {path.name}')
        raw=json.loads(path.read_text());check_compatible(raw)
        return model.model_validate(raw)
    intent=load(interview/'intent_model.json',IntentModel);conops=load(interview/'conops.json',ConOpsDocument)
    needs=load(spec/'needs.json',StakeholderNeedsSet);srs=load(spec/'srs.json',SRSDocument);arch=load(spec/'architecture.json',ArchitectureOverview)
    def array(name,model):
        path=spec/name
        if not path.is_file() or path.is_symlink() or path.stat().st_size>2_000_000:raise ValueError(f'Unsafe output: {name}')
        values=json.loads(path.read_text())
        for value in values:check_compatible(value)
        return [model.model_validate(x) for x in values]
    items=array('items.json',ItemSpec)
    tests=array('tests.json' if (spec/'tests.json').is_file() else 'test_specifications.json',AcceptanceTest)
    if any(model.session_id!=intent.session_id for model in (conops,needs,srs,arch)):raise ValueError('Phase outputs belong to different sessions')
    validation_path=spec/'validation.json'  # optional: sessions run before the validation stage carry none
    if validation_path.exists():
        from vmodel_contracts.verification import ValidationPlan
        if validation_path.is_symlink() or validation_path.stat().st_size>2_000_000:raise ValueError('Unsafe output: validation.json')
        raw=json.loads(validation_path.read_text());check_compatible(raw)
        if ValidationPlan.model_validate(raw).session_id!=intent.session_id:raise ValueError('Validation plan belongs to another session')
    semantic_records={}
    for path,canonical in ((interview/'semantic-review.json',intent.model_dump(mode='json')),(spec/'srs.semantic-review.json',srs.model_dump(mode='json'))):
        if path.exists():
            if path.is_symlink() or path.stat().st_size>2_000_000:raise ValueError('Unsafe semantic review')
            review=SemanticReview.model_validate_json(path.read_text())
            if review.session_id!=intent.session_id:raise ValueError('Semantic review belongs to another session')
            resolved=review.resolved_candidate(lambda c:None)
            semantic_records[('interview/' if path.parent==interview else 'spec/')+path.name]=review
            # Compare typed contract values, including nested defaults added by validation.
            normalized=(IntentModel if path.parent==interview else SRSDocument).model_validate(resolved).model_dump(mode='json')
            if any(canonical.get(k)!=normalized[k] for k in resolved if k!='founder_register'):raise ValueError('Canonical output differs from reviewed candidate')
    reqs=RequirementSet(session_id=intent.session_id,requirements=srs.requirements())
    violations=validate_traceability(needs,reqs,items,tests)
    if violations:raise ValueError('Traceability is incomplete: '+'; '.join(violations[:6]))
    register=intent.founder_register
    if register is None:raise ValueError('Founder decision register is missing; review interview completeness before packaging')
    if register.session_id!=intent.session_id:raise ValueError('Founder register belongs to another session')
    manifest=SpecManifest(bundle_version='0.2.0',files={'needs':'needs.md','srs':'srs.md','architecture':'architecture.md','agents':'AGENTS.md','traceability':'traceability.json','manifest':'manifest.json',**{i.spec_id:f'specs/{i.spec_id}.md' for i in items}},milestones=order_milestones(items,tests))
    trace=[x.model_dump(mode='json') for x in build_traceability(reqs,items,tests)]
    if srs.founder_register is None or srs.founder_register!=register:
        raise ValueError('SRS founder register differs from the reviewed intent')
    decisions=[{'id':d.decision_id,'choice':d.decision,'why':d.rationale,'source':'spec/architecture.json','owner':'architecture review'} for d in arch.decisions]
    decisions.insert(0,{'id':conops.decision.selected_concept_id,'choice':conops.decision.selected_concept,'why':conops.decision.rationale,'source':'interview/conops.json','owner':'founder'})
    for source,record in semantic_records.items():
        decisions.extend({'id':d.item_id,'choice':d.action,'why':d.rationale,'source':source,'owner':d.actor} for d in record.decisions)
    decisions.extend({'id':x.item_id,'choice':x.decision,'why':x.reason or 'no reason recorded','source':'founder-decisions.json','owner':x.owner} for x in register.items if x.status=='decided')
    digest_lines=['# Coding-agent handoff','','## Decided']+[f"- {d['id']}: {d['choice']} Why: {d['why']}. Source: {d['source']}." for d in decisions]
    digest_lines+=['','## Open founder inputs']+[f"- {x.item_id} [{x.data_class}, {x.status}] Owner: {x.owner}. {x.question} Blocks: {', '.join(x.blocks) or 'none recorded'}. Source: {x.source}." for x in register.items if x.status!='decided']
    digest_lines+=['','## Constraint advisories']+[f'- {x.advisory_id} [{x.status}]: {x.statement} Validation: {x.validation_expectation}' for x in register.advisories]
    digest_lines+=['','Read interview/intent_model.json and interview/conops.json first. Then spec/needs.json, spec/srs.json, spec/architecture.json, spec/items.json and spec/tests.json. Follow spec/AGENTS.md. No build command is inferred.','Producer-relative paths are retained; the spec manifest is relative to spec/.']
    files={'README.md':'\n'.join(digest_lines)+'\n','founder-decisions.json':register.model_dump_json(indent=2),
        'founder-decisions.md':'# Founder decision register\n\n'+'\n'.join(digest_lines[digest_lines.index('## Open founder inputs')+1:]),
        'decisions.json':json.dumps(decisions,indent=2),'spec/AGENTS.md':render_agents_md(arch,items,manifest,[]).replace('Read founder-decisions.json','Read ../founder-decisions.json').replace('from README.md','from ../README.md'),
        'spec/manifest.json':manifest.model_dump_json(indent=2),'spec/traceability.json':json.dumps(trace,indent=2)}
    files.update({name:record.model_dump_json(indent=2) for name,record in semantic_records.items()})
    for item in items:files[f'spec/specs/{item.spec_id}.md']=render_spec_file(item)
    files.update({'spec/srs.md':render_srs(srs),'spec/needs.md':render_needs(needs),'spec/architecture.md':render_architecture(arch)})
    for logical,relative in manifest.files.items():
        if relative.startswith('/') or '..' in PurePosixPath(relative).parts:raise ValueError('Manifest has unsafe dependency')
        name='spec/'+relative
        if name not in files:raise ValueError(f'Missing manifest dependency: {relative}')
    phase_files={'interview/intent_model.json':interview/'intent_model.json','interview/conops.json':interview/'conops.json',
       'spec/needs.json':spec/'needs.json','spec/srs.json':spec/'srs.json','spec/architecture.json':spec/'architecture.json','spec/items.json':spec/'items.json','spec/tests.json':spec/('tests.json' if (spec/'tests.json').is_file() else 'test_specifications.json')}
    if validation_path.exists():phase_files['spec/validation.json']=validation_path
    for name,path in phase_files.items():
        files[name]=path.read_text()
        sibling=path.with_suffix('.md')
        if sibling.is_file() and not sibling.is_symlink() and sibling.stat().st_size<=2_000_000 and str(PurePosixPath(name).with_suffix('.md')) not in files:files[str(PurePosixPath(name).with_suffix('.md'))]=sibling.read_text()
    from .docs import render_docs
    sources={'intent':intent.model_dump(mode='json'),'conops':conops.model_dump(mode='json'),
        'needs':needs.model_dump(mode='json'),'srs':srs.model_dump(mode='json'),
        'architecture':arch.model_dump(mode='json'),'items':[i.model_dump(mode='json') for i in items],
        'tests':[t.model_dump(mode='json') for t in tests],'traceability':trace}
    history_path=spec/'review-history.json'
    history=json.loads(history_path.read_text()) if history_path.is_file() and not history_path.is_symlink() and history_path.stat().st_size<=2_000_000 else []
    # Export documentation without logs/checkpoints/private diagnostic payloads.
    history=[{k:v for k,v in r.items() if k!='state'} for r in history]
    docs_state_path=spec/'docs-state.json'
    docs_state=json.loads(docs_state_path.read_text()) if docs_state_path.is_file() and not docs_state_path.is_symlink() and docs_state_path.stat().st_size<=2_000_000 else {}
    docs=render_docs(sources,{s:'available (not quality-approved)' for s in ('needs','srs','architecture','items','tests','review','assemble')},history,docs_state.get('changes',[]))
    files.update({'docs/'+name:text for name,text in docs.items()})
    files['README.md']+='\nReadable views: [docs/README.md](docs/README.md).\n'
    for name,text in files.items():
        if len(text.encode())>2_000_000:raise ValueError('Handoff file exceeds limit: '+name)
    baseline=spec/'system-baseline.json'
    if baseline.is_file() and not baseline.is_symlink():
        from vmodel_contracts import SystemBaseline
        if baseline.stat().st_size>2_000_000:raise ValueError('Baseline exceeds limit')
        raw=json.loads(baseline.read_text());check_compatible(raw)
        SystemBaseline.model_validate(raw);files['spec/system-baseline.json']=baseline.read_text()
    blockers=[f'{x.item_id}: {x.question}' for x in register.items if x.status!='decided' and x.blocks]
    arch_path=spec/'architecture.json'
    if arch_path.is_file() and not arch_path.is_symlink():
        from .architecture_decisions import open_slots
        blockers+=[f"architecture slot {d['key']}: no founder decision" for d in open_slots(json.loads(arch_path.read_text()))]
    from .findings_ledger import Ledger,revision_of
    ledger_path=spec/'findings-ledger.json'
    if not ledger_path.is_file() or ledger_path.is_symlink():
        blockers.append('findings ledger missing: this bundle was never content-reviewed (unreviewed, not clean)')
    else:
        from .citations import check_snapshots, raise_support_findings
        architecture=arch.model_dump(mode='json')
        for decision in architecture['decisions']:
            check_snapshots(decision,json.loads((interview/'intent_model.json').read_text()))
        ledger=Ledger.from_dict(json.loads(ledger_path.read_text()))
        raise_support_findings(ledger,architecture)
        gate=ledger.gate({i.spec_id:revision_of(i.body_markdown or '') for i in items})
        blockers+=[f'open finding {fid}: blocks handoff' for fid in gate['open']]+[r for r in gate['reasons'] if 'open finding' not in r]
    from vmodel_contracts.handoff import CodingHandoff
    return CodingHandoff.model_validate({'version':1,'session_id':intent.session_id,'files':files,'blocked_reasons':blockers,
            'sha256':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in files.items()}}).model_dump(mode='json')
