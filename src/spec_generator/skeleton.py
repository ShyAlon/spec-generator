"""Pipeline-owned document skeletons (ADR-0020): code owns structure - ids,
kinds, ordering, trace links - and the LLM authors prose only.

SRS stage: one requirement slot per stakeholder need, R-n in need order, with
kind / need_ids / intent_fields carried 1:1 (the mapping is mechanical in
practice). Tests stage: one acceptance-level slot per SRS acceptance criterion
plus one unit slot per item spec, T-n continuous. Architecture: the model
proposes decisions in bulk, code assigns ADR-n in order.

merge_* assembles the contract-shaped body from skeleton + prose fills and
raises ModelStructureError on unusable structure (unknown slot, slot filled
twice, slot missing) - a retryable model error; the retry feedback names it.
"""
from __future__ import annotations

import re


class ModelStructureError(Exception):
    """The model's fills cannot be mapped onto the skeleton."""


def _fills_by_id(fills, key: str) -> dict:
    by_id = {}
    for f in fills:
        if not isinstance(f, dict):
            raise ModelStructureError(f"each fill must be an object with '{key}'")
        fid = f.get(key)
        if fid in by_id:
            raise ModelStructureError(f"{fid}: slot filled twice")
        by_id[fid] = f
    return by_id


# Future-phase commitments (founder decision 2026-09-13): a need the founder
# deferred IN the interview ('we WILL have a community feature - in the
# future') is recorded as deferred, never as a current requirement, and never
# left reading as a contradiction between the needs and the SRS scope.
_FUTURE_MARKERS = ("in the future", "future phase", "future release", "phase 2",
                   "phase two", "phase ii", "post-mvp", "post mvp", "down the road",
                   "later on", "later we", "in a later", "after launch", "eventually",
                   "someday", "not yet", "for now,")


def is_future_commitment(text: str) -> bool:
    """Deterministic deferred-detection from the need's own words."""
    t = str(text).lower()
    return any(m in t for m in _FUTURE_MARKERS)


def deferred_needs(needs: dict) -> list[dict]:
    return [n for n in (needs.get("needs") or []) if is_future_commitment(n.get("statement", ""))]

def _words(text: str) -> set[str]:
    stop={"the","a","an","and","or","for","to","of","in","on","with","should","shall","system","app","application"}
    return {w for w in re.findall(r"[a-z0-9]+",str(text).lower()) if len(w)>2 and w not in stop}


def merge_duplicate_needs(needs: dict, threshold: float = 0.72) -> dict:
    """Deterministically collapse near-identical generated needs.

    Needs are the structural source of SRS slots. If the needs model repeats
    one capability, letting both through guarantees duplicate requirements.
    Merge only high-confidence lexical duplicates of the same kind; preserve
    first wording/order and union all trace ids/fields. Borderline semantic
    overlap remains visible for human review rather than guessed away.
    """
    out=[]
    for raw in needs.get("needs") or []:
        item=dict(raw); words=_words(item.get("statement",""))
        match=None
        for prior in out:
            pwords=_words(prior.get("statement",""))
            score=(len(words&pwords)/len(words|pwords)) if words|pwords else 0
            same_title=bool(words and pwords and words==pwords)
            if prior.get("kind")==item.get("kind") and (score>=threshold or same_title):
                match=prior;break
        if match is None: out.append(item);continue
        match["intent_fields"]=list(dict.fromkeys((match.get("intent_fields") or [])+(item.get("intent_fields") or [])))
        match["intent_need_ids"]=list(dict.fromkeys((match.get("intent_need_ids") or [])+(item.get("intent_need_ids") or [])))
    for i,item in enumerate(out,1): item["need_id"]=f"N-{i}"
    return {**needs,"needs":out}


def build_srs_skeleton(needs: dict) -> list[dict]:
    """One requirement slot per CURRENT stakeholder need, R-n in need order.
    Deferred (future-phase) needs drive no slots: they are commitments about
    LATER, recorded in the SRS scope's deferred block (srs_document_data)."""
    slots = []
    for i, n in enumerate(needs.get("needs") or [], start=1):
        if is_future_commitment(n.get("statement", "")):
            continue
        slots.append({
            "requirement_id": f"R-{len(slots) + 1}",
            "kind": n.get("kind"),
            "need_ids": [n.get("need_id")],
            "intent_fields": list(n.get("intent_fields") or []),
        })
    return slots


def merge_srs_fills(slots: list[dict], body) -> dict:
    """Skeleton structure + model prose -> contract-shaped SRS body. AC ids
    are assigned here, sequentially, in slot order. Structure carried from the
    intent model (definitions, constraints, ...) is NOT part of this body -
    srs_document_data adds it at validation/assembly."""
    fills = body.get("fills") if isinstance(body, dict) else None
    if isinstance(fills, dict):
        by_id = dict(fills)  # per-skeleton grammar (schemas.srs_fill_schema): fills keyed by requirement_id
    elif isinstance(fills, list):
        by_id = _fills_by_id(fills, "requirement_id")  # free-mode fallback without the grammar
    else:
        raise ModelStructureError("expected 'fills' keyed by requirement_id, one entry per requirement slot")
    functional, non_functional = [], []
    ac = 0
    for slot in slots:
        rid = slot["requirement_id"]
        f = by_id.pop(rid, None)
        if f is None:
            raise ModelStructureError(f"{rid}: slot not filled - every requirement needs prose")
        criteria = []
        for stmt in f.get("acceptance") or []:
            ac += 1
            criteria.append({"criterion_id": f"AC-{ac}", "statement": stmt})
        req = {
            "requirement_id": rid,
            "title": f.get("title"),
            "statement": f.get("statement"),
            "priority": f.get("priority"),
            "acceptance": criteria,
            "kind": slot["kind"],
            "need_ids": slot["need_ids"],
            "intent_fields": slot["intent_fields"],
        }
        (functional if slot["kind"] == "functional" else non_functional).append(req)
    if by_id:
        raise ModelStructureError(f"fills reference unknown requirement ids: {sorted(by_id)}")
    return {"purpose": body.get("purpose"), "scope_summary": body.get("scope_summary"),
            "functional": functional, "non_functional": non_functional}


def build_test_skeleton(srs: dict, items: list) -> list[dict]:
    """One acceptance slot per SRS acceptance criterion, then one unit slot
    per item spec; T-n continuous, criterion/spec links owned by code."""
    slots = []
    n = 0
    for section in ("functional", "non_functional"):
        for req in srs.get(section) or []:
            for crit in req.get("acceptance") or []:
                n += 1
                slots.append({"test_id": f"T-{n}", "level": "acceptance",
                              "criterion_id": crit.get("criterion_id"), "spec_id": None})
    for it in items or []:
        n += 1
        slots.append({"test_id": f"T-{n}", "level": "unit",
                      "criterion_id": None, "spec_id": it.get("spec_id")})
    return slots


def merge_test_fills(slots: list[dict], body) -> list[dict]:
    fills = body.get("fills") if isinstance(body, dict) else None
    if isinstance(fills, dict):
        by_id = dict(fills)  # per-skeleton grammar (schemas.test_fill_schema): fills keyed by test_id
    elif isinstance(fills, list):
        by_id = _fills_by_id(fills, "test_id")  # free-mode fallback without the grammar
    else:
        raise ModelStructureError("expected 'fills' keyed by test_id, one entry per test slot")
    out = []
    for slot in slots:
        tid = slot["test_id"]
        f = by_id.pop(tid, None)
        if f is None:
            f = slot.get("seed")  # an add_test proposal is adopted as-is (ADR-0020)
        if f is None:
            raise ModelStructureError(f"{tid}: slot not filled - every test needs prose")
        out.append({"test_id": tid, "level": slot["level"],
                    "criterion_id": slot["criterion_id"], "spec_id": slot["spec_id"],
                    "description": f.get("description"), "runner": f.get("runner"),
                    "definition": f.get("definition")})
    if by_id:
        raise ModelStructureError(f"fills reference unknown test ids: {sorted(by_id)}")
    return out


def assign_architecture_ids(body: dict) -> dict:
    """Model proposes decisions in bulk without ids; code assigns ADR-n in
    order and rejects duplicate component names."""
    out = dict(body)
    names = [c.get("name") for c in out.get("components") or []]
    if len(names) != len(set(names)):
        raise ModelStructureError(f"duplicate component names: {sorted({n for n in names if names.count(n) > 1})}")
    decisions = []
    for i, d in enumerate(out.get("decisions") or [], start=1):
        dd = {k: v for k, v in d.items() if k != "decision_id"}
        dd["decision_id"] = f"ADR-{i}"
        decisions.append(dd)
    out["decisions"] = decisions
    return out
