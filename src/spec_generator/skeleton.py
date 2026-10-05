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


def deferred_sn_ids(intent: dict | None) -> set[str]:
    """SN ids the interview flagged deferred (IntentNeed.deferred)."""
    return {n.get("need_id") for n in ((intent or {}).get("stakeholder_needs") or []) if n.get("deferred") is True}


def is_future_commitment(need, intent: dict | None = None) -> bool:
    """A need is a future commitment when the interview flagged it deferred
    (IntentNeed.deferred, directly or via intent_need_ids), OR - fallback for
    payloads from before the flag - its own words match a future marker.
    `need` may be a dict, a model with .statement, or bare text."""
    if isinstance(need, str):
        statement, flagged = need, False
    else:
        get = need.get if isinstance(need, dict) else lambda k, d=None: getattr(need, k, d)
        statement = get("statement", "") or ""
        flagged = get("deferred", False) is True or bool(
            set(get("intent_need_ids", None) or []) & deferred_sn_ids(intent))
    t = str(statement).lower()
    return flagged or any(m in t for m in _FUTURE_MARKERS)


def deferred_needs(needs: dict, intent: dict | None = None) -> list[dict]:
    return [n for n in (needs.get("needs") or []) if is_future_commitment(n, intent)]

def needs_from_intent(intent: dict) -> dict:
    """Carry validated interview needs forward without asking a model.

    The intent artifact already owns the substance, kind, identity and source
    provenance. A model rewrite added untraceable "implied" needs in the live
    supply-chain run, duplicating SN-1/SN-5 as N-6/N-9. One N slot per approved
    SN is the deterministic V-model relation; no new need may appear without
    founder evidence.
    """
    needs=[]
    for i,source in enumerate(intent.get("stakeholder_needs") or [],1):
        needs.append({
            "need_id":f"N-{i}",
            "stakeholder":"Founder",
            "statement":source.get("statement"),
            "kind":source.get("kind"),
            "intent_fields":["stakeholder_needs"],
            "intent_need_ids":[source.get("need_id")],
        })
    return {"session_id":intent.get("session_id"),"needs":needs}


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


def build_srs_skeleton(needs: dict, intent: dict | None = None) -> list[dict]:
    """One requirement slot per CURRENT stakeholder need, R-n in need order.
    Deferred (future-phase) needs drive no slots: they are commitments about
    LATER, recorded in the SRS scope's deferred block (srs_document_data)."""
    slots = []
    for i, n in enumerate(needs.get("needs") or [], start=1):
        if is_future_commitment(n, intent):
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


_RUNNER_REQUIRED = {"http": ("url", "method", "expected_status"), "cli": ("command",)}
# UI markup is the developer's: a spec-time selector is imagined implementation. The scan is narrow on
# purpose (test-hook attributes, attribute selectors, querySelector) so code identifiers in prose pass.
_MARKUP_RE = re.compile(r"data-test(?:id)?\b|\[[A-Za-z-]+[~|^$*]?=|queryselector", re.I)
# Outcomes that state nothing a check could fail on. A cheap tripwire, not a quality gate.
_VACUOUS_RE = re.compile(
    r"^(the |a |an )?(page|app|application|screen|view|element|component|ui|it)?\s*"
    r"(loads?|renders?|works?|exists?|runs?|is (visible|displayed|present|shown|rendered|available))"
    r"(\s+(successfully|correctly|properly|as expected))?$|^no errors?( occurs?| are shown)?$", re.I)


def _outcomes(d) -> list[str]:
    if isinstance(d, dict):
        out = list(d.get("expected_outcomes") or [])
        for c in d.get("cases") or []:
            out += list((c.get("expected_outcomes") if isinstance(c, dict) else getattr(c, "expected_outcomes", [])) or [])
        return [str(o) for o in out]
    out = list(getattr(d, "expected_outcomes", []) or [])
    for c in getattr(d, "cases", []) or []:
        out += list(getattr(c, "expected_outcomes", []) or [])
    return [str(o) for o in out]


def check_runner_execution(tests: list[dict]) -> None:
    """Deterministic gate on generated tests. The error names the exact test so the retry feedback
    is specific. (1) The old runner name 'dom' is now 'ui'. (2) A runner that
    executes something carries the machine-readable fields it needs. (3) UI markup is the
    developer's: no selector key in execution and no test-hook or attribute selector in the test
    text. (4) Loose tripwire: a test states at least one outcome, and not only a tautology
    ('page loads', 'no error', 'element is visible')."""
    problems = []
    for t in tests:
        tid = t.get("test_id")
        runner = str(t.get("runner") or "").strip().lower()
        if runner == "dom":
            problems.append(f"{tid}: runner 'dom' is now 'ui' (a ui test names no selectors)")
        d = t.get("definition")
        ex = (d.get("execution") if isinstance(d, dict) else getattr(d, "execution", None)) or {}
        get = ex.get if isinstance(ex, dict) else lambda k, default=None: getattr(ex, k, default)
        need = _RUNNER_REQUIRED.get(runner)
        if need:
            missing = [f for f in need if get(f) in (None, "", [])]
            if missing:
                problems.append(f"{tid}: runner '{t.get('runner')}' requires definition.execution."
                                + ", ".join(missing) + " (missing: " + ", ".join(missing) + ")")
        if isinstance(ex, dict) and ex.get("selector"):
            problems.append(f"{tid}: definition.execution.selector is not allowed; the developer defines the markup. "
                            "Describe the element in product terms in steps and expected_outcomes")
        if d is None:
            continue
        texts = [str(t.get("description") or "")]
        if isinstance(d, dict):
            texts += [str(x) for k in ("preconditions", "steps", "cleanup") for x in (d.get(k) or [])]
        else:
            texts += [str(x) for k in ("preconditions", "steps", "cleanup") for x in (getattr(d, k, []) or [])]
        outs = _outcomes(d)
        if any(_MARKUP_RE.search(x) for x in texts + outs):
            problems.append(f"{tid}: names UI markup (a test hook or attribute selector); the developer defines it. "
                            "Describe the element in product terms")
        real = [o for o in outs if o.strip() and not _VACUOUS_RE.match(o.strip().rstrip(".!").strip())]
        if outs is not None and not real:
            problems.append(f"{tid}: states no checkable outcome (empty or only 'loads', 'no error', 'is visible'). "
                            "State a value or a state that can be true or false")
    if problems:
        raise ModelStructureError("; ".join(problems))


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
