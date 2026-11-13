#!/usr/bin/env python3
"""Read existing run exports. No calls, credentials, prompts or source text output.

Usage: python scripts/citation-metrics.py DEBUG_EXPORT_DIR
JSON arrays (debug export) and JSONL (local run) are both supported.
Missing evidence is null, never zero. Durations are observed, not predicted savings.
"""
import argparse
from collections import Counter
import json
import re
from pathlib import Path


def records(path):
    if not path.is_file():
        return None
    text=path.read_text()
    try:
        value=json.loads(text)
        return value if isinstance(value,list) else [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def summarize(root):
    root=Path(root)
    rejected=records(root/'rejected-llm-outputs/manifest.jsonl')
    usage=records(root/'spec/usage.jsonl')
    events=records(root/'task-events.json')
    citation=None if rejected is None else [r for r in rejected if r.get('stage')=='architecture'
              and 'cite the recorded decision text' in str(r.get('reason',''))]
    durations={}
    if events is not None:
        for event in events:
            payload=event.get('payload') or {}
            elapsed=payload.get('elapsed_s')
            if payload.get('status') in ('completed','failed') and isinstance(elapsed,(int,float)):
                durations.setdefault(str(payload.get('stage') or event.get('phase_key')),[]).append(elapsed)
    # Older event exports omit elapsed_s. The existing final stage log is
    # explicit about elapsed time, including failed stages, so use it as such.
    log=root/'spec/run.log'
    log_durations=[]
    if log.is_file():
        for line in log.read_text().splitlines():
            match=re.search(r'(Drafting architecture overview).*?\b(FAILED|completed|COMPLETED).*?after ([0-9.]+)s',line)
            if match:
                log_durations.append({'stage':'architecture','status':match.group(2).lower(),'elapsed_s':float(match.group(3)),'source':'spec/run.log'})
    return {'evidence_kind' :'measured_existing_records','quote_copy_rejections':None if citation is None else len(citation),
            'quote_copy_repair_rejections':None if citation is None else sum(int(r.get('attempt',0))>1 for r in citation),
            'quote_copy_rejected_call_seconds':None if citation is None else [r.get('duration_s') for r in citation],
            'usage_rows_by_stage':None if usage is None else dict(Counter(r.get('stage','unknown') for r in usage)),
            'observed_stage_durations_seconds':durations or None,
            'observed_stage_log_durations':log_durations or None,
            'limits':['Rejection manifest coverage is export-dependent; an absent manifest is not zero failures.',
                      'Usage rows do not uniquely identify retries or total stage duration.',
                      'No per-retry dollars or live before/after savings are inferred.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export_dir')
    print(json.dumps(summarize(parser.parse_args().export_dir),indent=2))
