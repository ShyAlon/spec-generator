"""JEV-D10 findings ledger: one record of everything that can block assembly.

A finding is raised by a deterministic rule, by Jev, or by an LLM review. After
that everything is data and code:

  OPEN        raised, not closed
  ADDRESSED   a versioned deterministic predicate no longer fires AND the
              target actually changed (before/after hashes stored)
  REJECTED    the founder judged it wrong or accepted the risk (never "fixed")
  SUPERSEDED  a stale closure was invalidated; a new OPEN finding links back

Semantic findings close only by founder disposition (REJECTED with a reason
subtype). A changed artifact alone never closes one, and no model can. Structural
findings are never waivable. The gate is pure code over this ledger. Storage and
gating never use a display cap: dedup happens first, nothing is sliced.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from .content_acceptance import normalize

LEDGER_VERSION = 1
OPEN, ADDRESSED, REJECTED, SUPERSEDED = "OPEN", "ADDRESSED", "REJECTED", "SUPERSEDED"
SEMANTIC, STRUCTURAL = "semantic", "structural"
# Rules with a reproducible predicate. Only these can reach ADDRESSED.
DETERMINISTIC_RULES = frozenset({"echo.directive", "duplicate.body", "structural"})
DISPOSITION_SUBTYPES = frozenset({"founder_disposition", "false_positive", "accepted_risk",
                                  "resolved_by_reviewed_replacement"})
FOUNDER_SOURCES = frozenset({"decision-page", "founder-script"})


class LedgerError(ValueError):
    """A transition or script the ledger refuses. Loud, never a silent no-op."""


def revision_of(text: str) -> str:
    """Revision hash of cleaned-up text (formatting and extra whitespace removed)."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _is_founder(actor: str) -> bool:
    return isinstance(actor, str) and (actor == "founder" or actor.startswith("founder:")) and len(actor) < 240


class Ledger:
    def __init__(self, findings: dict | None = None, incomplete: list | None = None, history: list | None = None):
        self.findings: dict[str, dict] = findings or {}
        self.incomplete: list[dict] = incomplete or []
        self.history: list[dict] = history or []

    # ---- raising -------------------------------------------------------
    def add(self, *, rule_id: str, rule_version: str, source: str, kind: str, targets: list[str],
            artifact_revision: str, summary: str, evidence: list[dict] | None = None,
            dependency_revisions: dict | None = None, alias: str | None = None,
            scope: str = "spec", detail: dict | None = None, supersedes: str | None = None) -> str:
        """Add a finding, or merge into an existing one with the same stable key.
        Key = rule + target set + artifact revision + evidence identity. Different
        evidence is a different finding. Wording/source aliases are preserved."""
        if kind not in (SEMANTIC, STRUCTURAL):
            raise LedgerError(f"unknown finding kind {kind!r}")
        evidence = [dict(e) for e in (evidence or [])]
        evidence_identity = sorted((e.get("id", ""), e.get("sha256", "")) for e in evidence) or [("summary", _digest(normalize(summary)))]
        key = _digest([rule_id, sorted(targets), artifact_revision, evidence_identity])
        fid = "F-" + key[:16]
        existing = self.findings.get(fid)
        if existing is not None:
            for text in (alias or summary,):
                if text and text not in existing["aliases"]:
                    existing["aliases"].append(text)
            if source not in existing["sources"]:
                existing["sources"].append(source)
            return fid
        self.findings[fid] = {
            "finding_id": fid, "rule_id": rule_id, "rule_version": rule_version, "kind": kind,
            "sources": [source], "scope": scope, "targets": sorted(targets), "summary": summary,
            "aliases": [alias] if alias and alias != summary else [], "evidence": evidence,
            "artifact_revision": artifact_revision, "dependency_revisions": dict(dependency_revisions or {}),
            "state": OPEN, "closure": None, "supersedes": supersedes, "superseded_by": None, "detail": detail or {},
        }
        return fid

    def mark_incomplete(self, scope: str, reason: str, **detail) -> None:
        """A required check did not finish (chunk truncated, over limit, provider
        down). Blocks the gate; never counts as a clean pass."""
        marker = {"scope": scope, "reason": reason, **detail}
        if marker not in self.incomplete:
            self.incomplete.append(marker)

    def clear_incomplete(self, scope: str) -> None:
        self.incomplete = [m for m in self.incomplete if m["scope"] != scope]

    # ---- closing -------------------------------------------------------
    def _get(self, fid: str) -> dict:
        try:
            return self.findings[fid]
        except KeyError:
            raise LedgerError(f"unknown finding {fid}") from None

    def founder_dispose(self, fid: str, *, actor: str, source: str, rationale: str,
                        artifact_revision: str, subtype: str = "founder_disposition") -> None:
        """REJECTED: founder judgment, bound to the exact artifact revision."""
        f = self._get(fid)
        if f["state"] != OPEN:
            raise LedgerError(f"{fid} is {f['state']}, not OPEN")
        if f["kind"] == STRUCTURAL:
            raise LedgerError(f"{fid} is a structural failure and cannot be waived")
        if not _is_founder(actor) or source not in FOUNDER_SOURCES:
            raise LedgerError("only a founder disposition (decision page or founder-authored script) can close a semantic finding")
        if subtype not in DISPOSITION_SUBTYPES:
            raise LedgerError(f"unknown disposition subtype {subtype!r}")
        if not str(rationale or "").strip():
            raise LedgerError("a disposition needs a rationale")
        if artifact_revision != f["artifact_revision"]:
            raise LedgerError(f"{fid}: disposition is for revision {artifact_revision[:12]}, finding is bound to {f['artifact_revision'][:12]} (stale)")
        f["state"] = REJECTED
        f["closure"] = {"actor": actor, "source": source, "rationale": rationale, "subtype": subtype,
                        "artifact_revision": artifact_revision}
        self.history.append({"event": "disposed", "finding_id": fid, "subtype": subtype, "actor": actor})

    def address(self, fid: str, *, before_revision: str, after_revision: str,
                predicate_id: str, predicate_version: str, predicate_still_fires: bool) -> None:
        """ADDRESSED: only for a reproducible predicate, only if the target changed
        and the predicate no longer fires. Unrelated edits and no-ops stay OPEN."""
        f = self._get(fid)
        if f["state"] != OPEN:
            raise LedgerError(f"{fid} is {f['state']}, not OPEN")
        if f["rule_id"] not in DETERMINISTIC_RULES:
            raise LedgerError(f"{fid}: rule {f['rule_id']} has no reproducible predicate; it closes only by founder disposition")
        if before_revision != f["artifact_revision"]:
            raise LedgerError(f"{fid}: before-revision does not match the finding")
        if after_revision == before_revision:
            raise LedgerError(f"{fid}: the target is unchanged (no-op repair stays open)")
        if predicate_still_fires:
            raise LedgerError(f"{fid}: the predicate still fires")
        f["state"] = ADDRESSED
        f["closure"] = {"predicate_id": predicate_id, "predicate_version": predicate_version,
                        "before_revision": before_revision, "after_revision": after_revision, "result": "pass"}
        self.history.append({"event": "addressed", "finding_id": fid})

    def supersede_open(self, fid: str, new_id: str) -> None:
        """An OPEN finding whose target was re-authored and re-raised under a new
        revision: keep the history, point at the new record."""
        f = self._get(fid)
        if f["state"] == OPEN and fid != new_id:
            f["state"] = SUPERSEDED
            f["superseded_by"] = new_id
            self.findings[new_id]["supersedes"] = fid

    def refresh(self, current: dict[str, str]) -> list[str]:
        """current: target id -> current artifact revision. A closed finding whose
        target moved on is SUPERSEDED and re-raised OPEN (history is kept)."""
        reopened = []
        for fid, f in list(self.findings.items()):
            if f["state"] not in (ADDRESSED, REJECTED):
                continue
            moved = [t for t in f["targets"] if t in current and current[t] != f["artifact_revision"]
                     and not (f["state"] == ADDRESSED and current[t] == (f["closure"] or {}).get("after_revision"))]
            if not moved:
                continue
            f["state"] = SUPERSEDED
            new_rev = current[moved[0]]
            nid = self.add(rule_id=f["rule_id"], rule_version=f["rule_version"], source=f["sources"][0],
                           kind=f["kind"], targets=f["targets"], artifact_revision=new_rev,
                           summary=f["summary"], evidence=f["evidence"], scope=f["scope"],
                           dependency_revisions=f["dependency_revisions"], supersedes=fid,
                           detail={**f["detail"], "reopened_because": "target changed after closure"})
            f["superseded_by"] = nid
            reopened.append(nid)
        return reopened

    # ---- gate ----------------------------------------------------------
    def open_findings(self) -> list[dict]:
        return [f for f in self.findings.values() if f["state"] == OPEN]

    def gate(self, current: dict[str, str] | None = None) -> dict:
        """Assembly gate. Blocked = any OPEN finding, incomplete required check, or
        closure that is stale against the current artifact revisions."""
        stale = []
        for f in self.findings.values():
            if f["state"] == REJECTED and current:
                if any(t in current and current[t] != f["artifact_revision"] for t in f["targets"]):
                    stale.append(f["finding_id"])
            if f["state"] == ADDRESSED and current:
                after = (f["closure"] or {}).get("after_revision")
                if any(t in current and current[t] != after for t in f["targets"]):
                    stale.append(f["finding_id"])
        open_ = [f["finding_id"] for f in self.open_findings()]
        reasons = ([f"{len(open_)} open finding(s)"] if open_ else []) \
            + ([f"{len(self.incomplete)} incomplete required check(s)"] if self.incomplete else []) \
            + ([f"{len(stale)} stale closure(s)"] if stale else [])
        return {"blocked": bool(reasons), "reasons": reasons, "open": open_,
                "incomplete": list(self.incomplete), "stale": stale}

    def quarantined(self) -> set[str]:
        """Targets with an OPEN finding: nothing may derive from them."""
        return {t for f in self.open_findings() for t in f["targets"]}

    # ---- founder-authored script --------------------------------------
    def apply_script(self, script: dict, current: dict[str, str] | None = None) -> list[str]:
        """Replay founder-authored dispositions against exact findings. Anything
        unknown, stale, ambiguous or wildcard stops the run (LedgerError). A script
        not authored by the founder is refused. Generic feedback is not a script."""
        if not isinstance(script, dict) or script.get("version") != 1 or script.get("authored_by") != "founder":
            raise LedgerError("a scripted disposition must be version 1 and authored_by 'founder'")
        entries = script.get("dispositions")
        if not isinstance(entries, list) or not entries:
            raise LedgerError("script has no dispositions")
        staged = deepcopy(self)
        used: list[str] = []
        for i, e in enumerate(entries):
            sel = (e or {}).get("selector") or {}
            rule, targets = sel.get("rule_id"), sel.get("target_ids")
            if not rule or not targets or "*" in str(rule) or any("*" in str(t) for t in targets):
                raise LedgerError(f"disposition {i}: selector needs an exact rule_id and target_ids (no wildcards)")
            if e.get("action") != "reject" or not e.get("expected_artifact_sha256"):
                raise LedgerError(f"disposition {i}: action must be 'reject' with expected_artifact_sha256")
            matches = [f for f in staged.open_findings()
                       if f["rule_id"] == rule and f["targets"] == sorted(targets)]
            if not matches:
                raise LedgerError(f"disposition {i}: no open finding matches {rule} on {sorted(targets)}")
            if len(matches) > 1:
                raise LedgerError(f"disposition {i}: selector is ambiguous ({len(matches)} open findings)")
            f = matches[0]
            if e["expected_artifact_sha256"] != f["artifact_revision"]:
                raise LedgerError(f"disposition {i}: expected hash does not match the finding's artifact revision (stale)")
            staged.founder_dispose(f["finding_id"], actor="founder", source="founder-script",
                                   rationale=e.get("rationale", ""), artifact_revision=f["artifact_revision"],
                                   subtype=e.get("reason_subtype", "founder_disposition"))
            used.append(f["finding_id"])
        self.findings, self.history = staged.findings, staged.history
        return used

    # ---- persistence ---------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": LEDGER_VERSION, "findings": deepcopy(self.findings),
                "incomplete": deepcopy(self.incomplete), "history": deepcopy(self.history)}

    @classmethod
    def from_dict(cls, data: dict | None) -> "Ledger":
        if not data:
            return cls()
        if data.get("version") != LEDGER_VERSION:
            raise LedgerError("unsupported findings ledger version")
        return cls(deepcopy(data.get("findings") or {}), deepcopy(data.get("incomplete") or []),
                   deepcopy(data.get("history") or []))


def finding_from_check(ledger: Ledger, result: dict, spec_id: str, body: str, *, source: str = "rule") -> str:
    """Turn a content_acceptance CheckResult (dict) into a finding."""
    rule = "echo.directive" if result["rule"] == "echo" else "duplicate.body"
    evidence = [{"id": f"{spec_id}:body", "sha256": revision_of(body), "excerpt": (result.get("matched_span") or "")[:400]}]
    if result.get("sibling_id"):
        evidence.append({"id": f"{result['sibling_id']}:body", "sha256": result.get("other_sha256", ""), "excerpt": ""})
    if result.get("catalogue_id"):
        evidence.append({"id": f"catalogue:{result['catalogue_id']}", "sha256": result.get("other_sha256", ""), "excerpt": ""})
    targets = [spec_id]  # the candidate is bound and quarantined; the accepted sibling is evidence
    return ledger.add(rule_id=rule, rule_version=result.get("policy_version", ""), source=source, kind=SEMANTIC,
                      targets=targets, artifact_revision=revision_of(body), evidence=evidence,
                      summary=f"{result['rule']} check {result['status']}: {result['reason']}",
                      detail={"check": result})
