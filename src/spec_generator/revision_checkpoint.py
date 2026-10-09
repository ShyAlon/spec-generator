"""Typed phase-specific seed for a new closed-gate proposal checkpoint.

This module never mutates the source checkpoint, runs a model, approves a gate,
creates a generation or dispatches a restart. Its output is not commit authority.
"""
from copy import deepcopy
from vmodel_contracts import IntentModel,StakeholderNeedsSet,SystemBaseline,SRSDocument,check_compatible


def prepare_revision_seed(*,source,phase,change_request,session_id):
    from .cli import _CLEAR_DOWNSTREAM,_predecessor,_rerun_missing_prerequisite
    from .graph import srs_document_data,validated_architecture
    if phase not in ('srs','architecture'):
        raise ValueError('closed phase guidance adapter unavailable')
    if not isinstance(change_request,str) or not change_request.strip() or len(change_request)>20000:
        raise ValueError('bounded explicit founder change request required')
    if not isinstance(source,dict) or not isinstance(session_id,str) or not session_id:
        raise ValueError('explicit source state and session required')
    missing=_rerun_missing_prerequisite(phase,source)
    if missing:raise ValueError('missing prerequisite '+missing)
    seed=deepcopy(source)
    for key,model in (('intent_model',IntentModel),('needs',StakeholderNeedsSet),('system_baseline',SystemBaseline)):
        document=seed.get(key)
        if not isinstance(document,dict):raise ValueError('typed upstream '+key+' required')
        check_compatible(document)
        seed[key]=model.model_validate(document).model_dump(mode='json')
    if seed['intent_model']['session_id']!=session_id or seed['needs']['session_id']!=session_id:
        raise ValueError('revision source session mismatch')
    known={n['need_id'] for n in seed['needs']['needs']}
    intent=seed['intent_model'];intent_ids={n['need_id'] for n in intent['stakeholder_needs']}
    for need in seed['needs']['needs']:
        if any(key not in intent for key in need['intent_fields']) or not set(need['intent_need_ids'])<=intent_ids:
            raise ValueError('upstream need trace mismatch')
    srs=seed.get('srs')
    if not isinstance(srs,dict):raise ValueError('reviewed source SRS required')
    check_compatible(srs)
    typed=SRSDocument.model_validate(srs_document_data(intent,srs))
    if typed.session_id!=session_id or typed.intent_model_ref!=session_id:
        raise ValueError('SRS source session mismatch')
    for requirement in typed.functional+typed.non_functional:
        if not set(requirement.need_ids)<=known or any(key not in intent for key in requirement.intent_fields):
            raise ValueError('SRS trace mismatch')
    old_architecture=deepcopy(seed.get('architecture'))
    if phase=='architecture':
        if not isinstance(old_architecture,dict):raise ValueError('reviewed source architecture required')
        check_compatible(old_architecture)
        architecture=validated_architecture(old_architecture)
        if architecture.session_id!=session_id or architecture.baseline_id!=seed['system_baseline']['baseline_id']:
            raise ValueError('architecture source basis mismatch')
        req_ids={r.requirement_id for r in typed.functional+typed.non_functional}
        if any(not set(c.requirement_ids)<=req_ids for c in architecture.components):
            raise ValueError('architecture source requirement trace mismatch')
    seed.update(deepcopy(_CLEAR_DOWNSTREAM[phase]))
    invalidated={'srs','architecture'} if phase=='srs' else {'architecture'}
    seed['expert_opinions']=[o for o in source.get('expert_opinions',[]) if o.get('gate') not in invalidated]
    seed['gate_feedback']={k:v for k,v in source.get('gate_feedback',{}).items() if k not in invalidated}
    seed['gate_feedback'][phase]=change_request.strip()
    # Architecture's existing feedback prompt needs its reviewed previous draft.
    if phase=='architecture':seed['architecture']=old_architecture
    seed['semantic_reviews']={k:v for k,v in source.get('semantic_reviews',{}).items()
        if k not in invalidated and not (phase=='srs' and k=='srs_applied_sha256')}
    return {'phase':phase,'seed':seed,'as_node':_predecessor(phase,False),
        'fresh_checkpoint_required':True,'partial_reuse_allowed':False,
        'source_mutated':False,'committed':False,'dispatch_started':False}


def seed_fresh_revision_checkpoint(*,source,source_checkpoint_id,expected_checkpoint_id,
                                   phase,change_request,session_id,graph,config):
    """Seed only an empty destination graph after exact source identity matching.

    Caller supplies a Stop-fenced graph and owned fresh output namespace. This
    does not copy a SQLite file or ever write into the original source graph.
    """
    from .cooperative_run import cli_stop_binding
    from .stop_fence import check_execution
    binding=cli_stop_binding()
    if binding is None:raise ValueError('bound cooperative revision execution required')
    if (not isinstance(source_checkpoint_id,str) or not source_checkpoint_id
        or source_checkpoint_id!=expected_checkpoint_id):
        raise ValueError('reviewed source checkpoint changed')
    if config!={'configurable':{'thread_id':session_id}}:
        raise ValueError('exact revision session checkpoint configuration required')
    prepared=prepare_revision_seed(source=source,phase=phase,change_request=change_request,session_id=session_id)
    check_execution()
    existing=graph.get_state(config)
    if existing.values or existing.next:
        raise ValueError('fresh revision destination checkpoint must be empty')
    graph.update_state(config,prepared['seed'],as_node=prepared['as_node'])
    check_execution()
    saved=graph.get_state(config)
    if not saved.values or saved.values.get('gate_feedback',{}).get(phase)!=change_request.strip():
        raise ValueError('revision feedback checkpoint was not durably stored')
    return {**prepared,'destination_checkpoint':saved.config,'checkpoint_seeded':True,
        'producer_invoked':False}
