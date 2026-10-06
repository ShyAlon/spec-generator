"""Provenance lint over the generated artifacts (rules provenance.turn_tag, provenance.reference_name).

The checks themselves live in vmodel_contracts.text_qa; this module walks every founder-facing string in an
artifact and raises ledger findings. Nothing is scrubbed: a hit is an OPEN finding the founder resolves."""
from __future__ import annotations

import json

from vmodel_contracts import text_qa

from .findings_ledger import Ledger, revision_of

RULES = ("provenance.turn_tag", "provenance.reference_name")
# Keys that hold identifiers or machine fields, not founder-facing prose.
_SKIP_KEYS = {"source_turns", "contract_version", "session_id", "intent_model_ref", "criterion_id", "executable_hint"}
_ID_KEYS = ("requirement_id", "spec_id", "test_id", "need_id", "decision_id", "validation_id", "item_id", "id")


def _strings(value, path=""):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, inner in value.items():
            if key in _SKIP_KEYS or str(key).endswith("_ids") or str(key).startswith("_"):
                continue
            yield from _strings(inner, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            yield from _strings(inner, f"{path}[{index}]")


def _records(artifact: str, data):
    """(target id, revision text, record) for each addressable record of an artifact."""
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = [r for value in data.values() if isinstance(value, list) for r in value if isinstance(r, dict)]
        rows = rows or []
    else:
        rows = []
    out = []
    for row in rows:
        rid = next((str(row[k]) for k in _ID_KEYS if row.get(k)), None)
        if rid:
            revision_text = str(row.get("body_markdown")) if row.get("body_markdown") else json.dumps(row, sort_keys=True, default=str)
            out.append((rid, revision_text, row))
    if isinstance(data, dict):  # everything outside the addressable lists belongs to the artifact itself
        rest = {k: v for k, v in data.items() if not (isinstance(v, list) and v and all(isinstance(x, dict) for x in v))}
        out.append((artifact, json.dumps(rest, sort_keys=True, default=str), rest))
    return out


def reference_names(intent_model: dict | None) -> list[str]:
    return [str(r.get("name")) for r in (intent_model or {}).get("reference_names") or [] if r.get("name")]


def raise_findings(ledger: Ledger, artifacts: dict, intent_model: dict | None) -> int:
    """Lint every artifact; OPEN findings are raised per (rule, target). Re-runs retire findings no longer hit.
    Returns the number of findings raised. With no declared reference names the name check is recorded as
    not run in the ledger history (non-blocking: nothing was declared to check against)."""
    names = reference_names(intent_model)
    raised: dict[str, set[str]] = {rule: set() for rule in RULES}
    for artifact, data in artifacts.items():
        if not data:
            continue
        for target, revision_text, record in _records(artifact, data):
            hits: dict[str, list[tuple[str, str]]] = {rule: [] for rule in RULES}
            for path, text in _strings(record):
                for finding in text_qa.turn_tag_findings(text):
                    hits["provenance.turn_tag"].append((path, finding["span"]))
                if names:
                    for finding in text_qa.reference_name_findings(text, names):
                        hits["provenance.reference_name"].append((path, finding["span"]))
            for rule, found in hits.items():
                if not found:
                    continue
                spans = sorted({span for _, span in found})
                revision = revision_of(revision_text)
                label = "turn marker" if rule.endswith("turn_tag") else "reference name"
                fid = ledger.add(
                    rule_id=rule, rule_version="1", source="provenance", kind="semantic", targets=[target],
                    artifact_revision=revision, scope=artifact,
                    evidence=[{"id": f"{artifact}:{path}", "sha256": revision, "excerpt": span} for path, span in found[:20]],
                    summary=f"{target} contains a {label} in founder-facing text: {', '.join(repr(s) for s in spans[:5])}")
                raised[rule].add(fid)
    for rule in RULES:
        ledger.retire_rule(rule, raised[rule], "provenance lint re-run no longer finds this text")
    if not names:
        event = {"event": "check_not_run", "rule_id": "provenance.reference_name",
                 "reason": "no reference names declared in the intent model"}
        if event not in ledger.history:
            ledger.history.append(event)
    return sum(len(v) for v in raised.values())
