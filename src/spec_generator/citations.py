"""Attach source data in code; leave semantic support to human review.

No model calls, no prose rewriting, no source selection inferred from prose.
"""
from copy import deepcopy
from vmodel_contracts.citations import FounderDecisionCitation
from vmodel_contracts.semantic_review import digest
from .skeleton import ModelStructureError

SUPPORT_RULE = 'architecture.founder_citation_support'
# Owner policy decision. This changes only this named semantic review rule,
# never source integrity checks or unrelated ledger findings.
SUPPORT_REVIEW_BLOCKS_PACKAGING = True


def decided_sources(intent_model: dict) -> dict[str, dict]:
    items = ((intent_model.get('founder_register') or {}).get('items') or [])
    return {str(item['item_id']): item for item in items
            if item.get('item_id') and item.get('status') == 'decided'
            and str(item.get('decision') or '').strip() and str(item.get('reason') or '').strip()}


def source_snapshots(refs: list[str], intent_model: dict) -> list[dict]:
    if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
        raise ModelStructureError('founder_decision_refs must be a list of exact item ids')
    if len(set(refs)) != len(refs):
        raise ModelStructureError('founder_decision_refs must not contain duplicate ids')
    sources = decided_sources(intent_model)
    unknown = [ref for ref in refs if ref not in sources]
    if unknown:
        raise ModelStructureError(f'unknown or undecided founder decision refs: {unknown}')
    return [FounderDecisionCitation(source_ref=ref, source_revision=digest(sources[ref]),
                                   source=deepcopy(sources[ref])).model_dump() for ref in refs]


def check_snapshots(body: dict, intent_model: dict) -> None:
    expected = source_snapshots(body.get('founder_decision_refs') or [], intent_model)
    actual = body.get('founder_citations') or []
    # Backward read only: never fabricate historical snapshots or approvals.
    if str(body.get('contract_version') or '').startswith('0.28.') and not actual:
        return
    if actual != expected:
        raise ModelStructureError(
            f"{body.get('decision_id', 'ADR')}: missing, stale or altered founder source snapshot; "
            'regenerate this ADR against the current source. Quoting text cannot satisfy this check.')


def prepare_adr_sources(body: dict, slot: dict, intent_model: dict) -> dict:
    """New/legacy partials get snapshots; saved snapshots cannot silently rebind.

    The model body projection never accepts founder_citations. Only code-saved
    partials reach this function with snapshots, and those must still match.
    Mandatory concept links are a slot obligation, not model transcription.
    """
    out = deepcopy(body)
    expected = source_snapshots(out.get('founder_decision_refs') or [], intent_model)
    if out.get('founder_citations') and out['founder_citations'] != expected:
        raise ModelStructureError(
            f"{slot.get('decision_id', 'ADR')}: stale or altered founder source snapshot; "
            'regenerate this ADR against the current source')
    out['founder_citations'] = expected
    out['executes_concept_ids'] = list(dict.fromkeys(
        (slot.get('required_concept_ids') or []) + (out.get('executes_concept_ids') or [])))
    return out


def support_advisories(architecture: dict) -> list[dict]:
    """One visible question per cited ADR, never a synthetic support verdict."""
    return [{'rule': SUPPORT_RULE, 'decision_id': d['decision_id'],
             'status': 'support_not_yet_reviewed',
             'message': f"{d['decision_id']}: source support not yet reviewed. This is a system architecture choice, not a founder-decided architecture.",
             'action': 'Compare each recorded constraint with the ADR explanation. Does it support this choice without extending the founder input? Use the existing finding disposition or request regeneration.',
             'sources': deepcopy(d['founder_citations']),
             'adr': deepcopy(d)}
            for d in architecture.get('decisions') or [] if d.get('founder_citations')]


def raise_support_findings(ledger, architecture: dict) -> set[str]:
    raised = set()
    for advisory in support_advisories(architecture):
        adr, sources = advisory['adr'], advisory['sources']
        revision = digest({'adr': adr, 'sources': sources})
        raised.add(ledger.add(
            rule_id=SUPPORT_RULE, rule_version='1', source='architecture-exit',
            kind='semantic', scope='architecture', targets=[adr['decision_id']],
            artifact_revision=revision, summary=advisory['message'],
            evidence=[{'id': c['source_ref'], 'sha256': c['source_revision'],
                       'excerpt': c['source']['decision']} for c in sources],
            dependency_revisions={c['source_ref']: c['source_revision'] for c in sources},
            detail={'source': deepcopy(sources[0]['source']) if len(sources) == 1 else None,
                    'sources': deepcopy(sources), 'adr': deepcopy(adr),
                    'question': advisory['action']}))
    ledger.retire_rule(SUPPORT_RULE, raised, 'citation or ADR changed or removed; support must be reviewed for the new evidence')
    return raised
