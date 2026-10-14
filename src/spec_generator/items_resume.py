"""Durable items plan and exact partial metadata binding. Never replan saved bodies."""
import hashlib
import json
from .skeleton import ModelStructureError


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def context_digest(payload):
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_saved_plan(envelope, context, validator):
    if not isinstance(envelope, dict) or envelope.get('version') != 1:
        raise ValueError('invalid saved items plan envelope')
    plan = envelope.get('plan')
    if envelope.get('context_sha256') != context or envelope.get('plan_sha256') != digest(plan):
        raise ValueError('saved items plan context or digest changed')
    if not isinstance(plan, list) or not plan:
        raise ValueError('saved items plan must be nonempty')
    ids = [p.get('spec_id') for p in plan]
    if ids != [f'SPEC-{i}' for i in range(1, len(plan) + 1)]:
        raise ValueError('saved items plan numbering changed')
    by_id = {p['spec_id']: p['slug'] for p in plan}
    try:
        raw = [{k: v for k, v in p.items() if k not in ('spec_id', 'depends_on')}
               | {'depends_on': [by_id[d] for d in p.get('depends_on', [])]} for p in plan]
        normalized = validator(raw)
    except (KeyError, TypeError, ModelStructureError) as exc:
        raise ValueError('invalid saved items plan references') from exc
    if normalized != plan:
        raise ValueError('saved items plan does not match canonical validated plan')
    return plan


def prepare_plan(partials, payload, validator, propose):
    context = context_digest(payload)
    completed = (partials.load('items') or []) if partials else []
    saved = partials.load('items_plan') if partials else None
    if saved is None:
        if completed:
            raise ValueError('saved items bodies have no durable plan; explicit recovery review required, no automatic replan')
        plan = propose()
        saved = {'version': 1, 'context_sha256': context, 'plan_sha256': digest(plan), 'plan': plan}
        plan = validate_saved_plan(saved, context, validator)
        if partials:
            partials.save('items_plan', saved)  # before the first body provider call
    else:
        plan = validate_saved_plan(saved, context, validator)
    if not isinstance(completed, list):
        raise ValueError('saved items bodies must be a list')
    planned = {p['spec_id']: p for p in plan}
    done = {}
    for entry in completed:
        if not isinstance(entry, dict) or entry.get('spec_id') in done:
            raise ValueError('invalid or duplicate saved item body')
        key = entry.get('spec_id')
        if key not in planned or {k: v for k, v in entry.items() if k != 'body_markdown'} != planned[key]:
            raise ValueError('saved item metadata differs from durable plan')
        if not isinstance(entry.get('body_markdown'), str) or not entry['body_markdown'].strip():
            raise ValueError('saved item body is empty')
        done[key] = entry
    return plan, done
