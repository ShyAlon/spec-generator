"""Explicit revision-bound producer rework preparation. No auto-accept or writes.

Only selected flagged requirements are passed to the producer. Code preserves
IDs, kind, and trace links, validates the whole document, then creates a fresh
semantic review. The caller must checkpoint/publish with Stop and generation
fences and collect new founder decisions. This is not a route or runner.
"""
from copy import deepcopy
from vmodel_contracts import SRSDocument,text_qa
from vmodel_contracts.semantic_review import SemanticReview,make_review
from .graph import srs_document_data
from .stop_fence import check_execution

LOCKED=('requirement_id','kind','need_ids','intent_fields')


def prepare_rework(*,intent,needs,conops,review,revision,item_ids,guidance,produce):
    check_execution()
    base=SemanticReview.model_validate(review)
    if base.phase!='srs' or base.session_id!=intent.get('session_id') or base.revision!=revision:
        raise ValueError('SRS producer rework revision/session mismatch')
    if not isinstance(item_ids,list) or not item_ids or len(set(item_ids))!=len(item_ids):
        raise ValueError('explicit unique rework item ids required')
    if not isinstance(guidance,str) or len(guidance)>7000:raise ValueError('bounded rework guidance required')
    selected=set(item_ids);flagged={f.item_id for f in base.findings}
    if not selected<=flagged:raise ValueError('rework item is not in this review')
    paths={}
    for f in base.findings:
        if f.item_id in selected:
            if len(f.path)!=2 or f.path[0] not in ('functional','non_functional') or type(f.path[1]) is not int:
                raise ValueError('whole requirement rework only')
            if f.item_id in paths and paths[f.item_id]!=f.path:raise ValueError('ambiguous requirement path')
            paths[f.item_id]=f.path
    candidate=deepcopy(base.candidate);changes=[]
    for item_id in item_ids:
        check_execution();section,index=paths[item_id];old=candidate[section][index]
        replacement=produce(item_id=item_id,original=deepcopy(old),
            findings=[f.model_dump() for f in base.findings if f.item_id==item_id],
            intent=deepcopy(intent),needs=deepcopy(needs),conops=deepcopy(conops),guidance=guidance)
        check_execution()
        if not isinstance(replacement,dict):raise ValueError('producer must return a requirement object')
        if any(replacement.get(k)!=old.get(k) for k in LOCKED):
            raise ValueError('producer rework changed code-owned identity or trace links')
        # Acceptance identifiers are code-owned too. Rework cannot add/drop
        # criteria, although it may repair their wording.
        if [a.get('criterion_id') for a in replacement.get('acceptance',[])]!=[a.get('criterion_id') for a in old.get('acceptance',[])]:
            raise ValueError('producer rework changed acceptance identities')
        candidate[section][index]=deepcopy(replacement)
        changes.append({'item_id':item_id,'before':deepcopy(old),'after':deepcopy(replacement)})
    SRSDocument.model_validate(srs_document_data(intent,candidate))
    fresh=review_reworked_candidate(intent=intent,needs=needs,conops=conops,candidate=candidate)
    return {'review':fresh.model_dump(mode='json'),'changes':changes,
            'source_revision':revision,'founder_accepted':False,
            'requires_new_review':True,'committed':False}


def review_reworked_candidate(*,intent,needs,conops,candidate):
    """Code-owned findings cannot be replaced by the producer callback."""
    import json
    corpus=json.dumps([intent,needs,conops])
    lint=text_qa.srs_draft_findings(candidate,corpus);findings=[]
    from .lint_reasons import describe,group_findings
    for section in ('functional','non_functional'):
        for index,req in enumerate(candidate.get(section) or []):
            rid=str(req['requirement_id'])
            for number,group in enumerate(group_findings(lint,rid)):
                findings.append({'finding_id':f'{rid}:{number}','item_id':rid,'path':[section,index],
                    'rule':'+'.join(dict.fromkeys(g['rule'] for g in group)),
                    'reason':describe(rid,group),'original':req})
    return make_review(intent['session_id'],'srs',candidate,findings)


def prepare_model_rework(*,log,progress=None,**request):
    """Opt-in producer implementation, still no checkpoint or route install."""
    from .graph import _call_json,_prompt_view
    from vmodel_contracts import Requirement
    from .schemas import _grammar_safe
    import json
    def produce(**context):
        payload=("Revise ONLY this flagged SRS requirement. Describe observable behavior grounded in the owner evidence. "
                 "Keep all code-owned requirement, acceptance, kind and trace identities exactly. "
                 "Return one complete requirement JSON object. This is a proposal, never founder approval.\n"
                 +json.dumps(_prompt_view(context),ensure_ascii=False))
        return _call_json('srs',payload,log,progress,
            validate=lambda value:Requirement.model_validate(value).model_dump(mode='json'),
            schema=_grammar_safe(Requirement.model_json_schema()),stream_items=False,
            item_label=context['item_id'])
    return prepare_rework(**request,produce=produce)


def validate_proposal_result(*,result,source_review,revision,item_ids):
    """Check injected producer results at the retention boundary as well."""
    if not isinstance(result,dict):raise ValueError('rework result required')
    base=SemanticReview.model_validate(source_review)
    fresh=SemanticReview.model_validate(result.get('review'))
    if (base.revision!=revision or fresh.session_id!=base.session_id or fresh.phase!='srs'
        or fresh.decisions or result.get('source_revision')!=revision
        or result.get('committed') is not False or result.get('founder_accepted') is not False
        or result.get('requires_new_review') is not True):
        raise ValueError('unapproved revision-bound rework proposal required')
    if not isinstance(item_ids,list) or not item_ids or len(set(item_ids))!=len(item_ids):
        raise ValueError('explicit unique rework item ids required')
    paths={}
    for finding in base.findings:
        if finding.item_id not in item_ids:continue
        if (len(finding.path)!=2 or finding.path[0] not in ('functional','non_functional')
            or type(finding.path[1]) is not int):raise ValueError('whole requirement rework only')
        if finding.item_id in paths and paths[finding.item_id]!=finding.path:
            raise ValueError('ambiguous requirement path')
        paths[finding.item_id]=finding.path
    if set(paths)!=set(item_ids):raise ValueError('rework item is not in this review')
    candidate=deepcopy(base.candidate);expected=[]
    for item_id in item_ids:
        section,index=paths[item_id];old=candidate[section][index]
        try:new=fresh.candidate[section][index]
        except (KeyError,IndexError,TypeError) as exc:raise ValueError('rework requirement missing') from exc
        if not isinstance(new,dict) or any(new.get(k)!=old.get(k) for k in LOCKED):
            raise ValueError('producer rework changed code-owned identity or trace links')
        if [a.get('criterion_id') for a in new.get('acceptance',[])]!=[a.get('criterion_id') for a in old.get('acceptance',[])]:
            raise ValueError('producer rework changed acceptance identities')
        expected.append({'item_id':item_id,'before':deepcopy(old),'after':deepcopy(new)})
        candidate[section][index]=deepcopy(new)
    if candidate!=fresh.candidate:raise ValueError('producer rework changed unselected source evidence')
    if result.get('changes')!=expected:raise ValueError('producer rework change evidence mismatch')
    return fresh


def retain_checkpoint_proposal(*,capability,session_root,checkpoint_id,revision,item_ids,guidance,log,progress=None,prepare=None,validate_publication=None):
    """Retain a new proposal against the exact current source checkpoint.

    Never updates source SRS, semantic decisions, checkpoint cursor or downstream
    artifacts. Installing/applying the proposal is a separate reviewed workflow.
    """
    import json,sqlite3,uuid
    from .cooperative_run import cli_stop_binding
    from .graph import checkpoint_serde
    from langgraph.checkpoint.sqlite import SqliteSaver
    binding=cli_stop_binding()
    if binding is None:raise ValueError('bound worker fence required')
    fence,generation=binding
    if generation!=capability.generation:raise ValueError('rework generation mismatch')
    path=session_root/'spec'/'checkpoints.sqlite'
    from .checkpoint_runtime import configured,checkpoint
    if path.resolve()!=path or (not configured() and not path.is_file()):raise ValueError('owned source checkpoint required')
    config={'configurable':{'thread_id':capability.session_id}}
    def source():
        # Read-only SQLite URI, no source schema creation or journal mutation.
        conn=None
        if configured():saver=checkpoint(path.parent)
        else:
            conn=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,check_same_thread=False)
            saver=SqliteSaver(conn,serde=checkpoint_serde())
        try:
            row=saver.get_tuple(config)
            if row is None or row.config['configurable']['checkpoint_id']!=checkpoint_id:
                raise ValueError('source checkpoint changed')
            values=row.checkpoint['channel_values']
            from vmodel_contracts import check_compatible
            check_compatible(values['intent_model'])
            if values.get('conops') is not None:check_compatible(values['conops'])
            if values.get('system_baseline') is not None:check_compatible(values['system_baseline'])
            review=(values.get('semantic_reviews') or {}).get('srs')
            if not review or review.get('revision')!=revision or review.get('session_id')!=capability.session_id:
                raise ValueError('checkpoint SRS review changed')
            if values.get('srs')!=review.get('candidate'):
                raise ValueError('checkpoint SRS differs from source review')
            if values.get('architecture') or values.get('items'):
                raise ValueError('downstream checkpoint requires explicit invalidation review')
            return values
        finally:
            if conn is not None:conn.close()
    fence.check(generation);values=source()
    result=(prepare or prepare_model_rework)(intent=deepcopy(values['intent_model']),needs=deepcopy(values['needs']),
        conops=deepcopy(values.get('conops')),review=deepcopy(values['semantic_reviews']['srs']),
        revision=revision,item_ids=item_ids,guidance=guidance,log=log,progress=progress)
    fence.check(generation)
    fresh=validate_proposal_result(result=result,source_review=values['semantic_reviews']['srs'],
        revision=revision,item_ids=item_ids)
    SRSDocument.model_validate(srs_document_data(values['intent_model'],fresh.candidate))
    expected_review=review_reworked_candidate(intent=values['intent_model'],needs=values['needs'],
        conops=values.get('conops'),candidate=fresh.candidate)
    if fresh.model_dump()!=expected_review.model_dump():
        raise ValueError('producer rework semantic findings differ from code-owned review')
    # Same-host marker lock closes check/publication race. The worker writer
    # lock serializes checkpoint producers; stale evidence still fails closed.
    with fence.marker_guard():
        fence.check(generation);source()
        if validate_publication is not None:validate_publication()
        proposal_id=uuid.uuid4().hex
        directory=session_root/'spec'/'rework-proposals'
        if directory.resolve()!=directory:raise ValueError('unsafe proposal namespace')
        directory.mkdir(exist_ok=True)
        artifact=directory/(proposal_id+'.json')
        body={'proposal_id':proposal_id,'task_id':capability.task_id,'session_id':capability.session_id,
            'generation':generation,'source_checkpoint_id':checkpoint_id,'source_review_revision':revision,
            'selected_item_ids':list(item_ids),'guidance':guidance,'result':result,
            'committed':False,'founder_accepted':False,'dispatch_started':False}
        temp=artifact.with_suffix('.tmp')
        with temp.open('x') as file:
            json.dump(body,file,ensure_ascii=False,indent=2);file.flush();__import__('os').fsync(file.fileno())
        temp.replace(artifact)
        return body
