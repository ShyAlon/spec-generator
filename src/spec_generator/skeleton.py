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


def build_srs_skeleton(needs: dict) -> list[dict]:
    """One requirement slot per stakeholder need, R-n in need order."""
    slots = []
    for i, n in enumerate(needs.get("needs") or [], start=1):
        slots.append({
            "requirement_id": f"R-{i}",
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
    if not isinstance(fills, list):
        raise ModelStructureError("expected a 'fills' array with one entry per requirement slot")
    by_id = _fills_by_id(fills, "requirement_id")
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
    if not isinstance(fills, list):
        raise ModelStructureError("expected a 'fills' array with one entry per test slot")
    by_id = _fills_by_id(fills, "test_id")
    out = []
    for slot in slots:
        tid = slot["test_id"]
        f = by_id.pop(tid, None)
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
