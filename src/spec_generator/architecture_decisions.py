"""Founder decisions for research_required stack slots.

Deterministic, no model calls. A research_required slot is open until the
founder records a decision with a reason. Attended runs refuse to start the
items stage while any slot is open; unattended runs raise an OPEN ledger
finding per slot, so the gap is audited and blocks handoff.
"""
from __future__ import annotations

import copy
import hashlib
import json

from .errors import UserFacingError

DECISIONS_RULE = "architecture.undecided_slot"
DECISIONS_RULE_VERSION = "1"
COMPANION_NAME = "architecture.decisions.json"


def _slots(architecture: dict) -> list[dict]:
    return [d for d in (architecture or {}).get("stack_entry_dispositions") or []
            if str(d.get("status")) == "research_required"]


def open_slots(architecture: dict) -> list[dict]:
    return [d for d in _slots(architecture) if not d.get("founder_decision")]


def architecture_revision(architecture: dict) -> str:
    """Hash of the architecture without founder decisions, so saving decisions
    does not change the revision they were recorded against."""
    body = copy.deepcopy(architecture or {})
    for d in body.get("stack_entry_dispositions") or []:
        d.pop("founder_decision", None)
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def companion_for(architecture: dict) -> dict | None:
    """What the gate page needs. None when no slot needs a founder decision."""
    slots = _slots(architecture)
    if not slots:
        return None
    return {
        "revision": architecture_revision(architecture),
        "slots": [{"key": d["key"], "brief": d.get("research_brief") or "",
                   "options": list(d.get("candidate_options") or []),
                   "founder_decision": d.get("founder_decision")} for d in slots],
    }


def apply_decisions(architecture: dict, decisions: dict[str, dict]) -> dict:
    """Return a copy with founder_decision set on the named research_required slots.
    Unknown keys and blank text or reason are refused, never ignored."""
    out = copy.deepcopy(architecture)
    by_key = {d["key"]: d for d in _slots(out)}
    for key, dec in decisions.items():
        if key not in by_key:
            raise ValueError(f"no research_required slot named {key!r}")
        text, reason = str(dec.get("text") or "").strip(), str(dec.get("reason") or "").strip()
        if not text or not reason:
            raise ValueError(f"slot {key!r}: both a decision and a reason are required")
        by_key[key]["founder_decision"] = {"text": text, "reason": reason,
                                           "actor": str(dec.get("actor") or "founder"),
                                           "at": str(dec.get("at") or "")}
        if not by_key[key]["founder_decision"]["at"]:
            raise ValueError(f"slot {key!r}: decision time is required")
    return out


def merge_companion(architecture: dict, companion: dict | None) -> dict:
    """Apply saved decisions from the companion file when it matches this revision."""
    if not companion:
        return architecture
    if companion.get("revision") != architecture_revision(architecture):
        raise ValueError("saved architecture decisions belong to a different revision of the architecture")
    decided = {s["key"]: s["founder_decision"] for s in companion.get("slots") or [] if s.get("founder_decision")}
    return apply_decisions(architecture, decided) if decided else architecture


def require_decided(architecture: dict) -> None:
    """Attended runs: the items stage may not start with an open slot."""
    pending = open_slots(architecture)
    if pending:
        names = ", ".join(d["key"] for d in pending)
        raise UserFacingError(
            "stage 'items': the architecture still has undecided technology slots.",
            f"the founder has not decided: {names}.",
            "decide every open slot on the architecture gate (a decision and a reason), then continue.")


def raise_findings(ledger, architecture: dict, session_revision: str | None = None) -> set[str]:
    """Unattended runs: one OPEN finding per undecided slot. Findings of this rule that no
    longer fire are retired, so deciding a slot later closes its finding."""
    rev = session_revision or architecture_revision(architecture)
    raised = set()
    for d in open_slots(architecture):
        raised.add(ledger.add(
            rule_id=DECISIONS_RULE, rule_version=DECISIONS_RULE_VERSION, source="architecture-exit",
            kind="semantic", targets=[f"stack:{d['key']}"], artifact_revision=rev,
            summary=f"Stack slot {d['key']} is research_required and has no founder decision: {d.get('research_brief')}",
            evidence=[{"id": f"stack:{d['key']}", "sha256": hashlib.sha256(
                str(d.get("research_brief") or "").encode()).hexdigest(), "excerpt": str(d.get("research_brief") or "")[:400]}]))
    ledger.retire_rule(DECISIONS_RULE, raised, "slot decided or removed by an architecture re-run")
    return raised
