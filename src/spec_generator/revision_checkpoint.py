"""Typed phase-specific seed for a new closed-gate proposal checkpoint.

This module never mutates the source checkpoint, runs a model, approves a gate,
creates a generation or dispatches a restart. Its output is not commit authority.
"""
from copy import deepcopy
from vmodel_contracts import IntentModel,StakeholderNeedsSet,SystemBaseline,SRSDocument,check_compatible


def prepare_revision_seed(*,source,phase,change_request,session_id):
    from .cli import _CLEAR_DOWNSTREAM,_predecessor,_rerun_missing_prerequisite
    from .graph import srs_document_data,validated_architecture,_check_baseline_reconciliation,_check_decision_drivers
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
        _check_baseline_reconciliation(architecture,SystemBaseline.model_validate(seed['system_baseline']))
        _check_decision_drivers(architecture,intent,seed.get('conops'))
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


def validate_revision_output(*,source,result,phase,change_request,session_id):
    """Reject a proposal that drifts upstream or fails typed/trace boundaries.

    Validation produces no approval and never publishes the result. A following
    CAS must still compare fresh source/checkpoint/artifact evidence.
    """
    import hashlib,json
    from .graph import srs_document_data,validated_architecture,_check_baseline_reconciliation,_check_decision_drivers
    prepared=prepare_revision_seed(source=source,phase=phase,change_request=change_request,session_id=session_id)
    if not isinstance(result,dict):raise ValueError('revision graph output required')
    for key in ('intent_model','needs','system_baseline','conops'):
        if result.get(key)!=prepared['seed'].get(key):
            raise ValueError('revision changed approved upstream '+key)
    if phase=='architecture' and result.get('srs')!=source.get('srs'):
        raise ValueError('architecture revision changed approved SRS')
    if result.get('gate_feedback',{}).get(phase):
        raise ValueError('revision founder guidance was not consumed')
    # Stop at the edited phase. Downstream artifacts must never be mistaken for
    # approved regenerated products from a previous checkpoint/generation.
    downstream=('architecture','items','tests','validation','bundle') if phase=='srs' else ('items','tests','validation','bundle')
    if any(result.get(key) not in (None,[],{}) for key in downstream):
        raise ValueError('revision produced or retained unreviewed downstream output')
    document=result.get('srs' if phase=='srs' else 'architecture')
    if not isinstance(document,dict):raise ValueError('typed regenerated phase artifact required')
    check_compatible(document)
    needs={n['need_id'] for n in prepared['seed']['needs']['needs']}
    intent=prepared['seed']['intent_model']
    if phase=='srs':
        typed=SRSDocument.model_validate(srs_document_data(intent,document))
        if typed.session_id!=session_id or typed.intent_model_ref!=session_id:raise ValueError('regenerated SRS identity mismatch')
        for requirement in typed.functional+typed.non_functional:
            if not set(requirement.need_ids)<=needs or any(key not in intent for key in requirement.intent_fields):
                raise ValueError('regenerated SRS trace mismatch')
        normalized=typed.model_dump(mode='json')
    else:
        typed=validated_architecture(document)
        _check_baseline_reconciliation(typed,SystemBaseline.model_validate(prepared['seed']['system_baseline']))
        _check_decision_drivers(typed,intent,result.get('conops'))
        if typed.session_id!=session_id or typed.baseline_id!=prepared['seed']['system_baseline']['baseline_id']:
            raise ValueError('regenerated architecture identity mismatch')
        srs=SRSDocument.model_validate(srs_document_data(intent,source['srs']))
        ids={r.requirement_id for r in srs.functional+srs.non_functional}
        if any(not set(c.requirement_ids)<=ids for c in typed.components):
            raise ValueError('regenerated architecture trace mismatch')
        normalized=typed.model_dump(mode='json')
    return {'phase':phase,'artifact':normalized,
        'artifact_sha256':hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'typed_and_trace_validated':True,'requires_fresh_review':True,
        'founder_accepted':False,'committed':False,'dispatch_started':False}


def _prepare_lineage_seed(*,source,lineage,change_request,config):
    import re
    required={'kind','semantic_session_id','parent_execution_session_id','parent_generation',
        'execution_session_id','generation','phase','source_checkpoint_id','state','installed','dispatch_started'}
    if not isinstance(lineage,dict) or not required<=lineage.keys():
        raise ValueError('explicit complete semantic/execution lineage required')
    if (lineage['kind']!='closed_revision_execution_lineage' or lineage['state']!='planned'
        or lineage['installed'] is not False or lineage['dispatch_started'] is not False):
        raise ValueError('private uninstalled lineage plan required')
    for key in ('semantic_session_id','parent_execution_session_id','execution_session_id'):
        if not isinstance(lineage[key],str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}',lineage[key]):
            raise ValueError('canonical lineage identity required')
    if lineage['execution_session_id'] in (lineage['semantic_session_id'],lineage['parent_execution_session_id']):
        raise ValueError('fresh execution identity required')
    if (type(lineage['parent_generation']) is not int or lineage['parent_generation']<1
        or type(lineage['generation']) is not int or lineage['generation']!=lineage['parent_generation']+1):
        raise ValueError('exact next lineage generation required')
    if not isinstance(lineage['source_checkpoint_id'],str) or not lineage['source_checkpoint_id']:
        raise ValueError('exact lineage checkpoint required')
    if config!={'configurable':{'thread_id':lineage['execution_session_id']}}:
        raise ValueError('exact fresh execution checkpoint configuration required')
    return prepare_revision_seed(source=source,phase=lineage['phase'],change_request=change_request,
        session_id=lineage['semantic_session_id'])


def seed_lineage_revision_checkpoint(*,source,lineage,change_request,graph,config):
    """Execution seed requires its bound next-generation cooperative fence."""
    from .cooperative_run import cli_stop_binding
    from .stop_fence import check_execution
    prepared=_prepare_lineage_seed(source=source,lineage=lineage,change_request=change_request,config=config)
    binding=cli_stop_binding()
    if binding is None:raise ValueError('bound cooperative revision execution required')
    if binding[1]!=lineage['generation']:raise ValueError('execution fence generation differs from lineage')
    check_execution()
    existing=graph.get_state(config)
    if existing.values or existing.next:raise ValueError('fresh lineage destination checkpoint must be empty')
    graph.update_state(config,prepared['seed'],as_node=prepared['as_node'])
    check_execution();saved=graph.get_state(config)
    if (saved.values.get('gate_feedback',{}).get(lineage['phase'])!=change_request.strip()
        or saved.values.get('intent_model',{}).get('session_id')!=lineage['semantic_session_id']):
        raise ValueError('semantic lineage seed readback mismatch')
    return {'destination_checkpoint':saved.config,'semantic_session_id':lineage['semantic_session_id'],
        'execution_session_id':lineage['execution_session_id'],'checkpoint_seeded':True,
        'source_mutated':False,'installed':False,'dispatch_started':False}


def stage_reviewed_lineage_checkpoint(*,source,proposal_result,lineage,change_request,graph,config):
    """Legacy bound-next-generation staging; no execution or install."""
    class Probe:
        def __init__(self):self.values={}
        def get_state(self,configuration):
            from types import SimpleNamespace
            return SimpleNamespace(values=self.values,next=(),config=configuration)
        def update_state(self,configuration,values,as_node):self.values=deepcopy(values)
    probe=Probe()
    seed_lineage_revision_checkpoint(source=source,lineage=lineage,change_request=change_request,graph=probe,config=config)
    from .stop_fence import check_execution
    return _stage_checkpoint(source=source,proposal_result=proposal_result,lineage=lineage,change_request=change_request,
        graph=graph,config=config,seed=probe.values,check_authority=check_execution)


def stage_private_lineage_checkpoint(*,source,proposal_result,lineage,change_request,graph,config,validate_authority):
    """Preparation under original source fence, not fresh execution authority.

    Callback is supplied by the trusted owner worker, not client options. Caller
    holds original source writer exclusion and private destination containment.
    This helper never invokes graph or models, installs an anchor or binds a run.
    """
    from .cooperative_run import cli_stop_binding
    from .stop_fence import check_execution
    if not callable(validate_authority):raise ValueError('trusted staging authority reader required')
    prepared=_prepare_lineage_seed(source=source,lineage=lineage,change_request=change_request,config=config)
    binding=cli_stop_binding()
    if binding is None or binding[1]!=lineage['parent_generation']:
        raise ValueError('original source fence required for private staging')
    from pathlib import Path
    path=Path(binding[0].path)
    if (path.name!='stop-request.json' or path.parent.name!='interview'
        or path.parent.parent.name!=lineage['parent_execution_session_id']
        or path.parent.parent.parent.name!='sessions'
        or path.parent.parent.parent.parent.name!=lineage.get('owner_id')):
        raise ValueError('original source fence namespace differs from lineage')
    def authority():
        validate_authority();check_execution()
    authority()
    return _stage_checkpoint(source=source,proposal_result=proposal_result,lineage=lineage,change_request=change_request,
        graph=graph,config=config,seed=prepared['seed'],check_authority=authority)


def _stage_checkpoint(*,source,proposal_result,lineage,change_request,graph,config,seed,check_authority):
    validated=validate_revision_output(source=source,result=proposal_result,phase=lineage['phase'],
        change_request=change_request,session_id=lineage['semantic_session_id'])
    if validated['artifact_sha256']!=lineage.get('regenerated_artifact_sha256'):
        raise ValueError('proposal artifact differs from reviewed lineage digest')
    if proposal_result.get('expert_opinions',[])!=seed.get('expert_opinions',[]):
        raise ValueError('proposal retained unreviewed expert decisions')
    gate_node='gate_'+lineage['phase']
    topology=graph.get_graph()
    if not any(edge.source==lineage['phase'] and edge.target==gate_node and not edge.conditional for edge in topology.edges):
        raise ValueError('staged graph must require edited phase gate review')
    check_authority();existing=graph.get_state(config)
    if existing.values or existing.next:raise ValueError('fresh staged destination must be empty')
    # Start from cleared approved seed rather than copying unknown proposal
    # channels, pending writes or old checkpoint metadata into the new thread.
    state=deepcopy(seed);state['gate_feedback'].pop(lineage['phase'],None)
    state[lineage['phase']]=deepcopy(validated['artifact'])
    check_authority()
    graph.update_state(config,state,as_node=lineage['phase'])
    check_authority();saved=graph.get_state(config)
    if (saved.next!=(gate_node,) or saved.values.get(lineage['phase'])!=validated['artifact']
        or saved.values.get('intent_model')!=seed['intent_model']
        or saved.values.get('gate_feedback',{}).get(lineage['phase'])):
        raise ValueError('staged reviewed checkpoint readback mismatch')
    return {'destination_checkpoint':saved.config,'artifact_sha256':validated['artifact_sha256'],
        'semantic_session_id':lineage['semantic_session_id'],'execution_session_id':lineage['execution_session_id'],
        'requires_fresh_gate_review':True,'installed':False,'dispatch_started':False}


def validate_staged_lineage_state(*,source,staged_state,lineage,change_request):
    """Typed/source-preservation readback only, never acceptance or installation."""
    prepared=prepare_revision_seed(source=source,phase=lineage['phase'],change_request=change_request,
        session_id=lineage['semantic_session_id'])
    validated=validate_revision_output(source=source,result=staged_state,phase=lineage['phase'],
        change_request=change_request,session_id=lineage['semantic_session_id'])
    if validated['artifact_sha256']!=lineage.get('regenerated_artifact_sha256'):
        raise ValueError('staged artifact differs from lineage digest')
    if staged_state.get('expert_opinions',[])!=prepared['seed'].get('expert_opinions',[]):
        raise ValueError('staged expert decisions differ from cleared source')
    return {'artifact_sha256':validated['artifact_sha256'],'typed_and_trace_validated':True,
        'approved_upstream_unchanged':True,'installed':False,'dispatch_started':False}
