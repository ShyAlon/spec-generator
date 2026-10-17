"""Intent model in, repo-ready bundle out.

Usage: python -m spec_generator.cli out/<session>/intent_model.json

Bundle layout (written INCREMENTALLY as each stage completes, atomic
tmp+rename - a late failure leaves all completed work on disk):
  needs.json / needs.md            after stage 1/7
  srs.json / srs.md                after stage 2/7 (re-written if review fixes apply)
  architecture.json / architecture.md  after stage 3/7
  items.json + specs/<SPEC-id>.md  after stage 4/7
  tests.json                       after stage 5/7
  validation.json                  after stage 6/7 (what only a person can judge)
  AGENTS.md / traceability.json / manifest.json   at assembly (stage 7/7)
  gate_srs.* / gate_architecture.* (--review)     at the gates
  usage.jsonl / checkpoints.sqlite                run bookkeeping
"""
from __future__ import annotations
from .docs import refresh_docs, archive_review
from . import cli_io

import argparse
import hashlib
import json
import os
import shlex
import shutil
from datetime import datetime
from pathlib import Path

from vmodel_contracts import (
    run_with_system_awake, AcceptanceTest, ArchitectureOverview, ItemSpec, PunchListItem,
    RequirementSet, SpecManifest, SRSDocument, StakeholderNeedsSet, SystemBaseline,
)
from vmodel_contracts.runtime import artifacts_root

from .errors import UserFacingError, run_cli
from .graph import build_graph, checkpoint_serde, srs_document_data, validated_architecture
from .progress import ProgressBus, stderr_printer
from .repair import coerce_for_model, normalize_enums
from .render import (render_agents_md, render_architecture, render_needs,
                     render_srs, render_spec_file)
from .telemetry import UsageLog, jsonl_sink


def _append_audit(path,record):
    from .stop_fence import guarded_publish
    def append():
        with path.open("a") as fh:fh.write(json.dumps(record)+"\n")
    guarded_publish(append)

def _validate_gate_edit(gate: str, edited: dict, intent: dict) -> None:
    """A hand edit must satisfy the same contract the assembly enforces - a
    gate can never push contract-breaking state downstream."""
    from vmodel_contracts import ArchitectureOverview, SRSDocument

    from .graph import srs_document_data

    try:
        if gate == "srs":
            data = srs_document_data(intent, edited)
            data.setdefault("session_id", intent["session_id"])
            data.setdefault("intent_model_ref", intent["session_id"])
            SRSDocument.model_validate(data)
        else:
            edited.setdefault("session_id", intent["session_id"])
            ArchitectureOverview.model_validate(edited)
    except Exception as e:
        raise UserFacingError(
            f"the edited {gate} JSON does not pass the contract schema.",
            f"validation detail: {type(e).__name__}: {str(e)[:400]}",
            "fix the JSON in the editor - the original is unchanged until a valid edit lands.",
        ) from e


def _render_gate(gate: str, artifact: dict, intent: dict) -> str:
    from pydantic import ValidationError
    from vmodel_contracts import ArchitectureOverview, SRSDocument

    from .graph import _contract_errors, srs_document_data
    from .render import render_architecture, render_srs

    try:
        if gate == "srs":
            data = srs_document_data(intent, artifact)
            data.setdefault("session_id", intent["session_id"])
            data.setdefault("intent_model_ref", intent["session_id"])
            return render_srs(SRSDocument.model_validate(data))
        return render_architecture(ArchitectureOverview.model_validate({**artifact, "session_id": intent["session_id"]}))
    except ValidationError as e:
        # the stage's own contract validation (graph._call_json) should have
        # caught this before the gate - reaching here means a bug, but the
        # founder still gets the friendly treatment, never a raw traceback.
        raise UserFacingError(
            f"the {gate} draft does not pass the contract schema, so the review gate cannot render it.",
            f"validation detail: {_contract_errors(e)}",
            f"{_rerun_guidance_from_env()}; if it repeats, the stage "
            "produced a contract-breaking draft its own validation missed - please report it as a bug.",
        ) from e


def _rerun_guidance_from_env() -> str:
    from . import model_config

    return model_config._rerun_guidance("")


def _backfill_persist(persist, values: dict) -> list[str]:
    """Resume companion: persist artifacts for stages completed BEFORE
    incremental persistence shipped (resume skips completed stages, so
    persist() never fires for them and the bundle dir stays empty). State is
    authoritative - overwrite stale files. Returns the stages written."""
    written = []
    for stage in ("needs", "srs", "architecture", "items", "tests", "validation"):
        if values.get(stage):
            persist(stage, values[stage], values)
            written.append(stage)
    return written


def _atomic_write(path: Path, text: str) -> None:
    """tmp + rename: a crash mid-write can never leave a torn artifact."""
    from .stop_fence import guarded_publish
    def write():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
    guarded_publish(write)


def _decided_review_on_disk(path: Path, semantic: dict) -> bool:
    """True when the file already holds the founder's decisions for this same
    review and the checkpoint copy has none. A resume's backfill must never
    replace recorded decisions with the checkpoint's undecided copy."""
    try:
        disk = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (isinstance(disk, dict) and bool(disk.get("decisions")) and not semantic.get("decisions")
            and disk.get("revision") == semantic.get("revision")
            and disk.get("session_id") == semantic.get("session_id"))


def _make_partials(outdir: Path, architecture_lifecycle=None):
    """Per-item resume WITHIN an authoring stage (his ask 2026-09-10, after a
    Ctrl+C at T-34 of 36 reran everything): completed fills/bodies land in a
    partial file the moment they close; a rerun of the stage skips them, the
    same way completed stages are skipped. Cleared when the stage completes;
    --from <stage> deletes partials for that stage and everything downstream
    (explicit rewind forces real re-runs)."""
    FILES = {"srs_fills": "srs_fills.partial.json",
             "architecture": "architecture.partial.json",
             "items": "items.partial.json",
             "items_plan": "items_plan.partial.json",
             "tests_fills": "tests_fills.partial.json",
             "validation_fills": "validation_fills.partial.json",
             "review": "review.partial.json"}

    class _Partials:
        def load(self, key: str):
            p = outdir / FILES[key]
            if key == "architecture" and architecture_lifecycle and p.exists():
                architecture_lifecycle.guard_partial_read()
            return json.loads(p.read_text()) if p.exists() else None

        def save(self, key: str, data) -> None:
            if key == "architecture" and architecture_lifecycle:
                try:
                    architecture_lifecycle.checkpoint_reentry(data)
                except ValueError as e:
                    raise UserFacingError(
                        "stage 'architecture': lifecycle blocked a partial checkpoint write.",
                        f"artifact state transition was not opened before regeneration: {e}",
                        "update spec-generator and rerun the same command; the graph checkpoint preserves completed work.") from e
            _atomic_write(outdir / FILES[key], json.dumps(data, indent=2))

        def regenerate(self, key: str, reason: str) -> None:
            if key == "architecture" and architecture_lifecycle:
                try:
                    architecture_lifecycle.regenerate(reason)
                except ValueError as e:
                    raise UserFacingError(
                        "stage 'architecture': lifecycle could not reopen the artifact for regeneration.",
                        f"invalid artifact state transition: {e}",
                        "update spec-generator and rerun the same command; the sealed architecture and checkpoint are preserved.") from e

        def clear(self, key: str) -> None:
            p = outdir / FILES[key]
            if p.exists():
                p.unlink()

    from .stop_fence import guarded_publish
    from .stop_publication import FencedSink
    return FencedSink(_Partials(),guarded_publish)


# partial files invalidated by --from <stage> (that stage and everything after)
_PARTIAL_DOWNSTREAM = {"srs": ("srs_fills", "architecture", "items", "items_plan", "tests_fills", "validation_fills", "review"),
                      "architecture": ("architecture", "items", "items_plan", "tests_fills", "validation_fills", "review"),
                      "items": ("items", "items_plan", "tests_fills", "validation_fills", "review"),
                      "tests": ("tests_fills", "validation_fills", "review"),
                      "validation": ("validation_fills", "review"),
                      "review": ("review",)}


def _apply_ledger_actions(graph, cfg, prior, args):
    """JEV-D10: founder dispositions and explicit repair requests go through the checkpoint, never free text."""
    import json as _json
    from .findings_ledger import Ledger, LedgerError, revision_of
    if not prior or prior.get("findings_ledger") is None:
        raise UserFacingError("there is no findings ledger in this checkpoint.",
                              "--dispose and --repair act on findings recorded by a reviewed run.",
                              "run the review stage first, then retry.")
    ledger = Ledger.from_dict(prior["findings_ledger"])
    update: dict = {}
    try:
        if args.dispose:
            script = _json.loads(args.dispose.read_text())
            current = {i.get("spec_id"): revision_of(i.get("body_markdown") or "") for i in prior.get("items") or []}
            ledger.apply_script(script, current)
            update["findings_ledger"] = ledger.to_dict()
        if args.repair:
            ids = [x.strip() for x in args.repair.split(",") if x.strip()]
            unknown = [i for i in ids if i not in ledger.findings]
            if unknown:
                raise LedgerError(f"unknown finding id(s): {', '.join(unknown)}")
            update["repair_request"] = {"finding_ids": ids, "requested_by": "founder"}
            update["repair_quick"] = bool(getattr(args, "quick", False))
            update["quick_stop"] = False
    except (LedgerError, OSError, ValueError) as exc:
        raise UserFacingError(f"ledger action refused: {exc}", "nothing was changed.",
                              "fix the script or ids and rerun.") from exc
    graph.update_state(cfg, update, as_node="review")
    return graph.get_state(cfg).values


def _pre_run_actions(graph, cfg, prior, args):
    """Ledger actions and quick_stop handling before a run. Returns (prior, dispose_only).

    A bare --dispose (no --repair) only records the founder's dispositions: it must NOT
    clear quick_stop, because clearing it re-points the checkpoint at items and the next
    plain run then regenerates everything the quick run deliberately skipped."""
    dispose_only = bool(args.dispose and not args.repair)
    if dispose_only:
        return _apply_ledger_actions(graph, cfg, prior, args), True
    if prior and prior.get("quick_stop") and not args.quick:
        # a finished quick run ends the graph at its review; any later run starts clean
        graph.update_state(cfg, {"quick_stop": False, "repair_quick": False}, as_node="apply_fixes")  # resume at items: regenerate what the quick run skipped
        prior = graph.get_state(cfg).values
    if args.dispose or args.repair:
        prior = _apply_ledger_actions(graph, cfg, prior, args)
    return prior, False


def _quick_summary(before: dict | None, after: dict, outdir) -> str:
    """What a quick repair run leaves behind: what closed, what is still open. Printed and saved."""
    from collections import Counter
    from .findings_ledger import Ledger
    new = Ledger.from_dict(after)
    old_open = {f["finding_id"] for f in Ledger.from_dict(before).open_findings()} if before else set()
    closed = [f for f in new.findings.values()
              if f["state"] == "SUPERSEDED" and (f.get("closure") or {}).get("reason", "").startswith("repaired")]
    opened = [f for f in new.open_findings()]
    fresh = [f for f in opened if f["finding_id"] not in old_open]
    by = Counter((f["scope"]) for f in opened)
    lines = ["QUICK REPAIR RESULT (evidence run: items and tests were NOT regenerated; no bundle assembled)", "",
             f"Closed by repair (target changed, not raised again): {len(closed)}"]
    lines += [f"  - {f['finding_id']} {f['scope']} {','.join(f['targets'])}: {f['summary'][:110]}" for f in closed]
    lines += ["", f"Still OPEN: {len(opened)}  by scope: {dict(by)}  (new this run: {len(fresh)})"]
    lines += [f"  - {f['finding_id']} {f['scope']} {','.join(f['targets'])}: {f['summary'][:110]}"
              for f in sorted(opened, key=lambda x: (x['scope'], x['targets']))]
    last = [h for h in new.history if h.get("event") == "repair_auto_approved"]
    if last and last[-1].get("unchanged_targets"):
        lines += ["", "Repair targets that came back UNCHANGED (model changed nothing, or an accepted SRS decision restored "
                      f"the text): {', '.join(last[-1]['unchanged_targets'])}"]
    lines += ["", f"Before/after of the SRS fix: {outdir / 'repair-proposal.md'}"]
    return "\n".join(lines) + "\n"


def _resolve_repair_proposal(graph, cfg, prior, args, persist):
    """Founder decision on a pending repair proposal. Approve applies the proposed SRS and continues at items;
    reject drops it and ends at assembly (the findings stay open)."""
    from .findings_ledger import Ledger
    proposal = (prior or {}).get("repair_proposal")
    if not proposal:
        raise UserFacingError("there is no pending repair proposal.", "nothing was changed.",
                              "run --repair first; it stops with a proposal for you to read.")
    if args.approve_repair and args.reject_repair:
        raise UserFacingError("--approve-repair and --reject-repair are mutually exclusive.", "nothing was changed.", "pick one.")
    ledger = Ledger.from_dict(prior.get("findings_ledger"))
    verdict = "approved" if args.approve_repair else "rejected"
    ledger.history.append({"event": f"repair_{verdict}", "finding_ids": proposal.get("finding_ids") or [],
                           "changed": sorted(proposal.get("changed") or {}), "actor": "founder"})
    if args.approve_repair:
        state = {**prior, "srs": proposal["srs"], "repair_proposal": None, "findings_ledger": ledger.to_dict()}
        persist("srs", proposal["srs"], state)
        graph.update_state(cfg, {"srs": proposal["srs"], "repair_proposal": None,
                                 "findings_ledger": ledger.to_dict()}, as_node="apply_fixes")
    else:
        graph.update_state(cfg, {"repair_proposal": None, "repair_scope": None, "findings_ledger": ledger.to_dict()},
                           as_node="review")
    return graph.get_state(cfg).values


def _stamped(value):
    """Contract stamp for artifacts written from raw dicts. The incremental
    writer bypasses the contract models (which add contract_version by default),
    and the handoff gate rejects unstamped payloads as pre-versioning artifacts.
    A stamp already on the payload is kept."""
    from vmodel_contracts import CONTRACT_VERSION
    if isinstance(value, list):
        return [_stamped(x) for x in value]
    if isinstance(value, dict):
        return {"contract_version": CONTRACT_VERSION, **value}
    return value


def _write_decisions_companion(outdir: Path, architecture: dict) -> None:
    """architecture.decisions.json: what the gate page needs to collect founder decisions.
    No research_required slot means no file. An earlier answer for the same slot key and brief
    is carried as `prefill` for the founder to confirm; it is never applied silently."""
    from .architecture_decisions import COMPANION_NAME, companion_for
    path = outdir / COMPANION_NAME
    companion = companion_for(architecture)
    if companion is None:
        if path.is_file() and not path.is_symlink():
            path.unlink()
        return
    earlier: dict[tuple[str, str], dict] = {}
    if path.is_file() and not path.is_symlink():
        try:
            for s in json.loads(path.read_text()).get("slots") or []:
                answer = s.get("founder_decision") or s.get("prefill")
                if answer:
                    earlier[(s.get("key"), s.get("brief"))] = answer
        except (ValueError, OSError):
            earlier = {}
    for s in companion["slots"]:
        answer = earlier.get((s["key"], s["brief"]))
        s["prefill"] = answer if answer and not s.get("founder_decision") else None
    _atomic_write(path, json.dumps(companion, indent=2))


def _update_architecture_metadata(graph, cfg, update):
    """Preserve the checkpoint's producing node when changing metadata.

    LangGraph cannot infer it reliably from a multi-channel update. Omitting
    as_node can reschedule START; guessing architecture can rewind later work.
    """
    # Source node is observable in checkpoint versions even when update metadata
    # has no writes field (current LangGraph serialization).
    checkpoint = graph.checkpointer.get_tuple(cfg).checkpoint
    seen = checkpoint.get("versions_seen") or {}
    ranked = sorted((version, node) for node, channels in seen.items()
                    if node != "__start__" for version in channels.values())
    if ranked:
        latest = ranked[-1][0]
        nodes = {node for version, node in ranked if version == latest}
    else:
        nodes = {node for node in seen if node != "__start__"}
    if len(nodes) != 1:
        raise UserFacingError("Checkpoint metadata owner is ambiguous.",
                              "cannot safely preserve its scheduled continuation.",
                              "inspect the current checkpoint before updating technical choices.")
    graph.update_state(cfg, update, as_node=next(iter(nodes)))


def _apply_saved_architecture_decisions(graph, cfg, prior: dict, outdir: Path, persist):
    """Resume: fold the founder's saved decisions (written by the gate page) into the checkpoint
    so items and everything after read them. Does not re-run architecture."""
    from .architecture_decisions import COMPANION_NAME, merge_companion, raise_findings
    from .findings_ledger import Ledger
    path = outdir / COMPANION_NAME
    arch = prior.get("architecture")
    if not arch or not path.is_file() or path.is_symlink():
        return prior
    try:
        merged = merge_companion(arch, json.loads(path.read_text()))
    except ValueError as exc:
        raise UserFacingError("Saved architecture decisions are not usable.", str(exc),
                              "open the architecture gate, review the decisions and save them again.") from exc
    if merged == arch:
        return prior
    ledger = Ledger.from_dict(prior.get("findings_ledger"))
    raise_findings(ledger, merged)
    update = {"architecture": merged, "findings_ledger": ledger.to_dict()}
    _update_architecture_metadata(graph, cfg, update)
    state = {**prior, **update}
    persist("architecture", merged, state)
    print("Saved architecture choices applied; continuing at the next stage.", file=cli_io.stderr())
    return state


def _refresh_technical_choices(graph, cfg, prior, outdir, persist, log, progress=None):
    """Metadata only, resumable after each slot; never regenerate architecture."""
    import copy
    from .graph import _technical_choice_call
    from .architecture_decisions import raise_findings
    from .findings_ledger import Ledger
    if not prior or not prior.get("architecture"):
        raise UserFacingError("No saved architecture to update.", "existing checkpoint required.", "finish architecture before refreshing technical choices.")
    if any(prior.get(k) for k in ("items", "tests", "manifest")):
        raise UserFacingError("Technical choices already have downstream results.",
                              "metadata refresh must not silently change built item specifications.",
                              "review the downstream impact before changing this architecture.")
    # Fold existing companion answers first so a refresh never replaces them.
    prior = _apply_saved_architecture_decisions(graph, cfg, prior, outdir, persist)
    architecture = copy.deepcopy(prior["architecture"])
    for entry in architecture.get("stack_entry_dispositions") or []:
        if entry.get("status") != "research_required" or entry.get("technical_choice") or entry.get("founder_decision"):
            continue
        entry["technical_choice"] = _technical_choice_call(prior, entry["key"], entry["research_brief"], log, progress)
        entry["candidate_options"] = [o["value"] for o in entry["technical_choice"]["options"]]
        ArchitectureOverview.model_validate(architecture)
        ledger = Ledger.from_dict(prior.get("findings_ledger"))
        raise_findings(ledger, architecture)
        update = {"architecture": copy.deepcopy(architecture), "findings_ledger": ledger.to_dict()}
        _update_architecture_metadata(graph, cfg, update)
        prior = {**prior, **update}
        persist("architecture", architecture, prior)
    _write_decisions_companion(outdir, architecture)
    print("Technical-choice metadata updated. Architecture generation was not rerun; no downstream stage started.", file=cli_io.stderr())
    return prior


def _make_persist(intent: dict, outdir: Path, greenfield: bool = False,
                  conops_present: bool = True, architecture_lifecycle=None):
    """The incremental bundle writer (his ask 2026-09-09: 'persist them
    intermittently so we'll have some of the data'). Fires after each stage;
    the final assembly writes remain the canonical full bundle."""
    sid = intent["session_id"]

    def _persist(stage: str, data, state: dict) -> None:
        revision = (state.get("artifact_revisions") or {}).get(stage)
        if revision:
            _atomic_write(outdir / f"{stage}.revision.json", json.dumps(revision, indent=2))
        if stage == "needs":
            body = {**data, "session_id": sid}
            _atomic_write(outdir / "needs.json", json.dumps(_stamped(body), indent=2))
            needs = StakeholderNeedsSet.model_validate(normalize_enums(coerce_for_model(body, StakeholderNeedsSet), StakeholderNeedsSet))
            _atomic_write(outdir / "needs.md", render_needs(needs))
        elif stage == "srs":
            semantic=(state.get('semantic_reviews') or {}).get('srs')
            if semantic and not _decided_review_on_disk(outdir/'srs.semantic-review.json',semantic):
                _atomic_write(outdir/'srs.semantic-review.json',json.dumps(semantic,indent=2))
            doc = srs_document_data(intent, {**data, "session_id": sid, "intent_model_ref": sid})
            _atomic_write(outdir / "srs.json", json.dumps(_stamped(doc), indent=2))
            srs = SRSDocument.model_validate(coerce_for_model(doc, SRSDocument))
            _atomic_write(outdir / "srs.md", render_srs(srs))
        elif stage == "architecture":
            body = {**data, "session_id": sid}
            arch = validated_architecture(body)
            lifecycle_event = architecture_lifecycle.finalize(body) if architecture_lifecycle else None
            _atomic_write(outdir / "architecture.json", json.dumps(_stamped(body), indent=2))
            _atomic_write(outdir / "architecture.md", render_architecture(arch, greenfield=greenfield, conops_present=conops_present))
            _write_decisions_companion(outdir, body)
            if architecture_lifecycle:
                architecture_lifecycle.commit_finalize(lifecycle_event)
        elif stage == "items":
            _atomic_write(outdir / "items.json", json.dumps(_stamped(data), indent=2))
            for i in data:
                item = ItemSpec.model_validate(coerce_for_model(i, ItemSpec))
                _atomic_write(outdir / "specs" / f"{item.spec_id}.md", render_spec_file(item))
        elif stage == "tests":
            _atomic_write(outdir / "tests.json", json.dumps(_stamped(data), indent=2))
        elif stage == "validation":
            _atomic_write(outdir / "validation.json",
                          json.dumps(_stamped({"session_id": sid, **data}), indent=2))

        refresh_docs(outdir, intent=intent, event=stage, state={**state, stage: data})

    _persist.review = lambda record: archive_review(outdir, record, intent=intent)

    def _content_evidence(record: dict) -> None:
        """Rejected candidates are evidence: kept verbatim, never deleted by a rerun."""
        path = outdir / "content-rejections.json"
        prior = json.loads(path.read_text()) if path.exists() else []
        _atomic_write(path, json.dumps(prior + [record], indent=2))

    _persist.content_evidence = _content_evidence
    def _grounding(spec_id: str, envelope: dict) -> None:
        path = outdir / "grounding-evidence.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        data[spec_id] = envelope
        _atomic_write(path, json.dumps(data, indent=2))
    _persist.grounding = _grounding
    def _repair_proposal_files(proposal: dict, text: str) -> None:
        _atomic_write(outdir / "repair-proposal.json", json.dumps(proposal, indent=2))
        _atomic_write(outdir / "repair-proposal.md", text)
    _persist.repair_proposal = _repair_proposal_files
    _persist.ledger = lambda data: _atomic_write(outdir / "findings-ledger.json", json.dumps(data, indent=2))
    from .stop_fence import guarded_publish
    from .stop_publication import FencedSink
    return FencedSink(_persist,guarded_publish)


STAGE_ORDER = ("needs", "srs", "architecture", "items", "tests", "validation", "review", "assemble")

# What a --from rewind invalidates: the named stage's own output and every
# downstream product (LangGraph state has no key delete; overwrite with
# empties - review_round resets so the review loop runs fresh).
_CLEAR_DOWNSTREAM = {
    "srs": {"srs": None, "architecture": None, "items": None, "tests": None, "validation": None,
            "review_round": 0, "review_fixes": [], "bundle": None, "violations": [],
            "guard_flags": []},
    "architecture": {"architecture": None, "items": None, "tests": None, "validation": None,
                     "review_round": 0, "review_fixes": [], "bundle": None, "violations": []},
    "items": {"items": None, "tests": None, "validation": None, "review_round": 0, "review_fixes": [],
              "bundle": None, "violations": []},
    "tests": {"tests": None, "validation": None, "review_round": 0, "review_fixes": [], "bundle": None,
              "violations": []},
    "validation": {"validation": None, "review_round": 0, "review_fixes": [], "bundle": None,
                   "violations": []},
    "review": {"review_round": 0, "review_fixes": [], "bundle": None, "violations": []},
    "assemble": {"bundle": None, "violations": []},
}


SRS_APPLIED_KEY = "srs_applied_sha256"


def _srs_candidate_digest(candidate: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(candidate, sort_keys=True, default=str).encode()).hexdigest()


def _srs_review_already_applied(prior: dict, candidate: dict, reviewed=None) -> bool:
    """True when this reviewed SRS is already recorded in the checkpoint.
    Marker first (written when we apply). Otherwise the checkpoint's own review must
    already carry the same decisions as the reviewed file. Equal requirement sections
    prove nothing: an all-Accept review leaves them untouched, and the checkpoint
    review would stay undecided and fail the architecture stage."""
    held = (prior.get("semantic_reviews") or {})
    if held.get(SRS_APPLIED_KEY) == _srs_candidate_digest(candidate):
        return True
    if reviewed is None:
        return False
    mine = held.get("srs") or {}
    want = reviewed.model_dump(mode="json").get("decisions") or []
    return bool(want) and (mine.get("decisions") or []) == want


def _predecessor(stage: str, review_mode: bool) -> str:
    """The node the checkpoint is rewritten AS (update_state as_node=...), so
    resume continues at the stage the founder asked to re-run. Review-mode
    graphs route through the gate nodes."""
    return {"srs": "needs",
            "architecture": "gate_srs" if review_mode else "srs",
            "items": "gate_architecture" if review_mode else "architecture",
            "tests": "items",
            "validation": "tests",
            "review": "validation",
            "assemble": "review"}[stage]


def _rerun_missing_prerequisite(stage: str, current: dict) -> str | None:
    """First absent upstream stage for a sequential --from rewind."""
    prerequisites = {
        "needs": (), "srs": ("needs",), "architecture": ("needs", "srs"),
        "items": ("needs", "srs", "architecture"),
        "tests": ("needs", "srs", "architecture", "items"),
        "validation": ("needs", "srs", "architecture", "items", "tests"),
        "review": ("needs", "srs", "architecture", "items", "tests", "validation"),
        "assemble": ("needs", "srs", "architecture", "items", "tests", "validation", "review"),
    }
    for prerequisite in prerequisites[stage]:
        fulfilled = current.get(prerequisite) is not None
        if prerequisite == "review": fulfilled = bool(current.get("review_round"))
        if not fulfilled: return prerequisite
    return None


def _invalidate_from(graph, cfg: dict, stage: str, review_mode: bool) -> None:
    """--from <stage>: rewind a checkpoint so the named stage and everything
    after it re-runs under the CURRENT code, state before it preserved (his
    2026-09-09 need: the completed run held pre-fix items with invented
    requirement ids; rerunning only resumed INTO the old state)."""
    if stage not in STAGE_ORDER:
        raise UserFacingError(
            f"--from: unknown stage {stage!r}.",
            f"valid stages: {', '.join(STAGE_ORDER)}.",
            "use --from items to regenerate specs/tests/review/assembly from the checkpointed SRS.",
        )
    if not graph.get_state(cfg).values:
        raise UserFacingError(
            f"--from {stage}: no checkpoint found for this session in the output directory.",
            "--from rewinds an existing run; there is nothing to rewind.",
            "run without --from first.",
        )
    if stage == "needs":
        return  # a fresh run already starts at needs and overwrites everything
    current = graph.get_state(cfg).values
    missing = _rerun_missing_prerequisite(stage, current)
    if missing:
        raise UserFacingError(
            f"--from {stage}: prerequisite stage '{missing}' is not complete.",
            "stages are sequential; rewinding downstream without every upstream result would produce ungrounded output.",
            f"run or rerun from '{missing}' first, then retry --from {stage}.",
        )
    invalidated_gates = {
        "srs": {"srs", "architecture"},
        "architecture": {"architecture"},
    }.get(stage, set())
    cleared = dict(_CLEAR_DOWNSTREAM[stage])
    if invalidated_gates:
        # Gate decisions and their retry budget describe the artifact version
        # that is being discarded. Preserve decisions for earlier stages only.
        cleared["expert_opinions"] = [
            opinion for opinion in (current.get("expert_opinions") or [])
            if opinion.get("gate") not in invalidated_gates
        ]
        cleared["gate_feedback"] = {
            gate: feedback for gate, feedback in (current.get("gate_feedback") or {}).items()
            if gate not in invalidated_gates
        }
    graph.update_state(cfg, cleared, as_node=_predecessor(stage, review_mode))
    from . import model_config
    model_config._console_line(
        f"  boundary=checkpoint-invalidation from_stage={stage} predecessor={_predecessor(stage, review_mode)} "
        f"cleared_state={sorted(_CLEAR_DOWNSTREAM[stage])} invalidated_gates={sorted(invalidated_gates)} "
        f"preserved_expert_gates={sorted({o.get('gate') for o in cleared.get('expert_opinions', current.get('expert_opinions') or [])})} "
        "decision=rewind-current-and-downstream")


def _run_with_gates(graph, cfg: dict, initial_state: dict, outdir, resume: bool = False,
                    auto_approve: bool = False) -> dict:
    """ADR-0019 review gates (opt-in via --review): after the SRS and after the
    architecture, the founder accepts, edits (JSON, contract-validated), sends
    feedback (stage re-runs with it), or rejects. Crash mid-gate: rerun the
    same command - the checkpoint re-renders and re-asks."""
    import subprocess

    from langgraph.types import Command

    # Compatibility for internal callers resuming an already-seeded graph.
    if "intent_model" not in initial_state:
        initial_state = {"intent_model": initial_state}
    intent = initial_state["intent_model"]
    result = graph.invoke(None, cfg) if resume else graph.invoke(initial_state, cfg)
    restarted = False
    while True:
        st = graph.get_state(cfg)
        if not st.next:
            return result
        payload = next((t.interrupts[0].value for t in st.tasks if t.interrupts), None)
        if payload is None:
            result = graph.invoke(None, cfg)
            continue
        gate, artifact = payload["gate"], payload["artifact"]
        advisories = payload.get("advisories") or []
        title = "SRS (requirements + acceptance criteria)" if gate == "srs" else "architecture overview"
        try:
            md = _render_gate(gate, artifact, intent)
        except UserFacingError:
            # a checkpoint that predates stage-level contract validation can hold
            # a poisoned draft; restarting regenerates it under the fixed schema
            # (invoke-with-input on an interrupted thread restarts from entry).
            # The first failure is EXPECTED on a pre-fix checkpoint - say so
            # calmly. Do NOT print the Error block here: it reads as a fresh
            # failure and founders kill the run mid-heal (his 2:58 paste).
            if restarted:
                raise  # second failure on a REGENERATED draft: real bug, full detail
            restarted = True
            print(f"\nThe checkpointed {gate} draft predates today's validation fixes. "
                  "This is the expected one-time heal: discarding it and regenerating "
                  "the stage under the fixed pipeline - the run continues on its own, "
                  "do NOT rerun the command.\n", file=cli_io.stderr())
            result = graph.invoke(initial_state, cfg)
            continue
        if advisories:
            md += "\n\n## Review advisories\n" + "\n".join(
                f"- {a['left_id']} {a['left_slug']} and {a['right_id']} {a['right_slug']}: {a['reason']}. Nothing was merged; decide at this gate."
                for a in advisories)
        _atomic_write(outdir / f"gate_{gate}.md",md)
        _atomic_write(outdir / f"gate_{gate}.json",json.dumps(artifact, indent=2))
        _atomic_write(outdir / f"gate_{gate}.advisories.json",json.dumps({"gate":gate,"advisories":advisories},indent=2))
        print(f"\n===== REVIEW GATE: the {title} =====\n", file=cli_io.stderr())
        print(md, file=cli_io.stderr())
        if advisories:
            print(f"(review advisories: {outdir}/gate_{gate}.advisories.json)", file=cli_io.stderr())
        print(f"(full artifact: {outdir}/gate_{gate}.md | editable JSON: {outdir}/gate_{gate}.json)", file=cli_io.stderr())
        if auto_approve:
            action = "a"
            record={"event":"gate.auto_decision", "gate":gate, "action":"accept", "mode":"hands_free",
                    "artifact":f"gate_{gate}.json", "advisory_count":len(advisories)}
            _append_audit(outdir/"audit.jsonl",record)
            print(f"boundary=hands-free-gate gate={gate} action=accept advisories={len(advisories)} decision=continue", file=cli_io.stderr())
        else:
            from vmodel_contracts.decisions import founder_decision
            options=[
                {"value":"a","label":"Accept","summary":"Approve this artifact and continue."},
                {"value":"e","label":"Edit","summary":"Open the generated JSON for a validated edit."},
                {"value":"f","label":"Feedback","summary":"Regenerate this phase from founder feedback."},
                {"value":"r","label":"Reject","summary":"Stop at this checkpoint."}]
            # A hosted run never opens an editor on the server; the app edits through its own review screen.
            if cli_io.bound():options=[o for o in options if o["value"]!="e"]
            decision=founder_decision(f"{gate}-review",f"Review {title}",f"What should happen at the {gate} gate?",options)
            cli_io.emit(decision)
            valid=[o["value"] for o in options]
            while True:
                action = cli_io.read(decision,"[a]ccept / [e]dit / [f]eedback / [r]eject? ").strip().lower()
                if action in valid:
                    break
        if action == "r":
            from . import model_config as _mc
            print(f"Rejected at the {gate} gate. The run is checkpointed - {_mc._rerun_guidance(gate)}.", file=cli_io.stderr())
            raise SystemExit(1)
        if action == "f":
            decision=founder_decision(f"{gate}-feedback",f"Feedback for {title}","What should regeneration change?",[
                {"value":"feedback","label":"Send feedback","summary":"Regenerate the phase using your written steer."}],
                rationale_required=True,rationale_label="Founder feedback",answer_format="feedback - <rationale>")
            cli_io.emit(decision)
            raw_feedback = cli_io.read(decision,"Feedback (one line, founder language): ").strip()
            fb = raw_feedback.partition(" - ")[2].strip() if raw_feedback.startswith("feedback - ") else raw_feedback
            if fb:
                result = graph.invoke(Command(resume={"action": "regenerate", "feedback": fb}), cfg)
            continue
        if action == "e":
            editor = os.environ.get("EDITOR", "vi")
            while True:
                subprocess.run([editor, str(outdir / f"gate_{gate}.json")])
                try:
                    edited = json.loads((outdir / f"gate_{gate}.json").read_text())
                    _validate_gate_edit(gate, edited, intent)
                    break
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON ({e}) - reopening the editor.", file=cli_io.stderr())
                except UserFacingError as e:
                    print(e.render() + "\nReopening the editor.", file=cli_io.stderr())
            graph.update_state(cfg, {gate: edited})
            print("Edit accepted and validated.", file=cli_io.stderr())
        result = graph.invoke(Command(resume={"action": "accept"}), cfg)


def _resume_after_bounded_stop(graph, cfg: dict, prior: dict) -> str | None:
    """Re-open a checkpoint intentionally ended by --to STAGE.

    A stop-after graph records a complete state with no pending task. Rebuilding
    the graph without --to cannot manufacture the missing edge from SQLite, so
    explicitly replay the last completed node against the new full topology.
    """
    if prior.get("bundle") or graph.get_state(cfg).next:
        return None
    for stage, downstream in (("items", "tests"), ("architecture", "items"),
                              ("srs", "architecture"), ("needs", "srs")):
        if prior.get(stage) and not prior.get(downstream):
            graph.update_state(cfg, {}, as_node=stage)
            return stage
    # --to tests ends the graph at tests. Review has not run when no review round
    # is recorded; a finished review that stopped for a repair proposal or a quick
    # run has recorded one (or set its own stop flag) and must not be replayed.
    if (prior.get("validation") and not prior.get("review_round")
            and not prior.get("quick_stop") and not prior.get("repair_proposal")):
        graph.update_state(cfg, {}, as_node="validation")
        return "validation"
    if (prior.get("tests") and not prior.get("validation") and not prior.get("review_round")
            and not prior.get("quick_stop") and not prior.get("repair_proposal")):
        graph.update_state(cfg, {}, as_node="tests")
        return "tests"
    return None


def _default_outdir(intent_path: Path, session_id: str) -> Path:
    """Canonical stable output: <artifacts>/<session>/spec."""
    from . import model_config
    return artifacts_root(model_config._env("VMODEL_ARTIFACTS_DIR")) / session_id / "spec"

def _migration_command(path: Path) -> str:
    """Copy-pasteable repair for a version-stale contract artifact."""
    return f"python -m vmodel_contracts.migrate {shlex.quote(str(path))}"


def _load_system_baseline(path: Path) -> dict:
    if not path.is_file():
        raise UserFacingError(
            f"system baseline file not found: {path}.",
            "architecture generation requires an explicit existing or greenfield baseline.",
            "pass --system-baseline <file>; copy examples/system-baseline.greenfield.json for a new product.",
        )
    try:
        raw = json.loads(path.read_text())
        from vmodel_contracts import ContractVersionMismatch, check_compatible
        check_compatible(raw)
        return SystemBaseline.model_validate(raw).model_dump(mode="json")
    except ContractVersionMismatch as e:
        raise UserFacingError(
            f"system baseline was produced against vmodel-contracts {e.have}, but this spec-generator reads {e.want.split('.')[0]}.{e.want.split('.')[1]}.x.",
            "the baseline was created before a contract upgrade.",
            f"migrate this file, then run the spec-generator command again: {_migration_command(path)}",
        ) from e
    except Exception as e:
        raise UserFacingError(
            f"system baseline is invalid: {path}.",
            f"validation detail: {type(e).__name__}: {str(e)[:400]}",
            "fix the file against vmodel-contracts SystemBaseline; use mode=greenfield with empty repositories for a new product.",
        ) from e


_ONLY_STAGE_INPUTS = {
    "srs": ("needs",),
    "architecture": ("srs",),
    "items": ("srs", "architecture"),
    "tests": ("srs", "architecture", "items"),
    "validation": ("needs", "srs", "architecture", "items", "tests"),
    "review": ("srs", "architecture", "items", "tests"),
}
_ARTIFACT_FILES = {"needs": "needs.json", "srs": "srs.json", "architecture": "architecture.json",
                   "items": "items.json", "tests": "tests.json",
                   "validation": "validation.json"}


def _stage_outdir(source: Path, stage: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return source.with_name(f"{source.name}-{stage}-{stamp}")


def _load_stage_state(source: Path, intent: dict, baseline: dict, stage: str) -> dict:
    """Load the immutable upstream artifact set for one stage iteration."""
    if stage not in _ONLY_STAGE_INPUTS:
        raise UserFacingError(
            f"--only-stage: unknown stage {stage!r}.",
            "only authoring/review stages can run alone.",
            f"valid stages: {', '.join(_ONLY_STAGE_INPUTS)}.",
        )
    state = {"intent_model": intent, "system_baseline": baseline}
    missing = []
    for key in _ONLY_STAGE_INPUTS[stage]:
        path = source / _ARTIFACT_FILES[key]
        if not path.is_file():
            missing.append(str(path))
            continue
        state[key] = json.loads(path.read_text())
    if missing:
        raise UserFacingError(
            f"cannot run only stage '{stage}': required upstream artifacts are missing.",
            "the source run is incomplete or the wrong directory was selected.",
            "missing: " + ", ".join(missing),
        )
    return state


def _prepare_stage_iteration(source: Path, outdir: Path) -> None:
    """Clone human artifacts into a new directory, never runtime/checkpoint state."""
    if source.resolve() == outdir.resolve():
        raise UserFacingError(
            "--only-stage refuses to write into the source artifacts directory.",
            "stage iteration must be non-destructive.",
            "choose a new --out directory or omit --out for an automatic one.",
        )
    if outdir.exists():
        raise UserFacingError(
            f"stage iteration output already exists: {outdir}.",
            "reusing it could mix two experiments.",
            "choose a fresh --out directory.",
        )
    ignored = shutil.ignore_patterns("checkpoints.sqlite*", "*.partial.json", "usage.jsonl",
                                     "llm-calls.jsonl", "run.log")
    shutil.copytree(source, outdir, ignore=ignored)
    (outdir / "specs").mkdir(exist_ok=True)


def _run_only_stage(args, intent: dict, sid: str) -> None:
    source = args.source_artifacts.resolve()
    if not source.is_dir():
        raise UserFacingError(
            f"source artifacts directory not found: {source}.",
            "--only-stage reads upstream artifacts from a completed run.",
            "pass --source-artifacts <artifacts-...>.",
        )
    outdir = (args.out.resolve() if args.out is not None else _stage_outdir(source, args.only_stage))
    baseline = _load_system_baseline(args.system_baseline)
    state = _load_stage_state(source, intent, baseline, args.only_stage)
    _prepare_stage_iteration(source, outdir)
    _atomic_write(outdir / "system-baseline.json", json.dumps(baseline, indent=2))
    if conops:
        _atomic_write(outdir / "conops.json", json.dumps(conops, indent=2))
    loaded = [*(_ONLY_STAGE_INPUTS[args.only_stage])]
    print(f"Stage iteration: {args.only_stage}", file=cli_io.stderr())
    print(f"Loaded immutable source: {source}", file=cli_io.stderr())
    print(f"System baseline: {args.system_baseline.resolve()} ({baseline['mode']}, {len(baseline['repositories'])} repos)", file=cli_io.stderr())
    print(f"Loaded upstream: {', '.join(loaded) or 'intent only'}", file=cli_io.stderr())
    print(f"Recomputing ONLY: {args.only_stage}; downstream artifacts are copied for comparison, not regenerated.",
          file=cli_io.stderr())
    print(f"New output: {outdir} (source will not be modified)", file=cli_io.stderr())
    refresh_docs(outdir, intent=intent, conops=state.get("conops"), start=True, invalidate=args.only_stage)
    log = UsageLog(sid, sink=jsonl_sink(outdir / "usage.jsonl"))
    bus = ProgressBus(); bus.subscribe(stderr_printer())
    from .structured_progress import StructuredProgress
    emitter=StructuredProgress(os.environ.get("VMODEL_PROGRESS_TASK_ID") or sid,sid,args.only_stage)
    bus.subscribe(emitter.subscriber)
    from . import structured_progress
    structured_progress.active_sink=emitter
    persist = _make_persist(intent, outdir, greenfield=baseline["mode"] == "greenfield")
    graph = build_graph(log, progress=bus, persist=persist, partials=_make_partials(outdir),
                        only_stage=args.only_stage)
    result = graph.invoke(state, {"configurable": {"thread_id": sid + "-" + args.only_stage}})
    if args.only_stage == "review":
        _atomic_write(outdir / "review-fixes.json", json.dumps(result.get("review_fixes") or [], indent=2))
    refresh_docs(outdir, intent=intent, event=args.only_stage, state=result)
    print(f"Stage '{args.only_stage}' written to {outdir}. Source unchanged.", file=cli_io.stdout())



_SEMANTIC_EQUIVALENCE_VERSION = "1"


def _resume_constraint_content(value):
    """Ignore version stamps for resume equality, preserving every other field.

    Load-time contract compatibility checks remain authoritative. A re-stamp
    does not change the constraint that completed stages were grounded in.
    """
    if isinstance(value, dict):
        return {key: _resume_constraint_content(item)
                for key, item in value.items() if key != "contract_version"}
    if isinstance(value, list):
        return [_resume_constraint_content(item) for item in value]
    return value


def _semantic_hash(value) -> str:
    """Hash contract meaning, not rendering or retrieval annotations.

    Version 1 deliberately excludes only evidence metadata. Open questions and
    every requirement/architecture decision remain semantic, so resolving a
    question stales downstream rather than being waved through as presentation.
    """
    def canonical(v):
        if isinstance(v, dict):
            return {k: canonical(val) for k, val in sorted(v.items())
                    if k not in {"evidence", "_resolved_open_questions"}}
        if isinstance(v, list):
            return [canonical(x) for x in v]
        return v
    payload = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _reconcile_artifacts(intent: dict, baseline: dict, source: Path,
                         target: Path | None = None, *, conops_present: bool = True) -> dict:
    """Re-run deterministic cross-artifact rules and views with zero LLM calls."""
    from .graph import _resolve_answered_open_questions

    source = source.resolve()
    if not source.is_dir():
        raise UserFacingError(
            f"artifact directory not found: {source}.",
            "reconciliation needs an existing completed or partial bundle.",
            "pass -o <artifact-directory> for the run you want to reconcile.")
    required = [source / "srs.json", source / "architecture.json"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise UserFacingError(
            "cannot reconcile: source artifacts are missing.",
            "the run did not reach both SRS and architecture.",
            "missing: " + ", ".join(missing))
    destination = target.resolve() if target else source
    if target:
        if destination.exists():
            raise UserFacingError(
                f"reconciliation output already exists: {destination}.",
                "a comparison generation must never overwrite another directory.",
                "choose a fresh --reconcile-out path.")
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("checkpoints.sqlite*", "*.partial.json"))

    old_srs = json.loads((source / "srs.json").read_text())
    architecture_data = json.loads((source / "architecture.json").read_text())
    old_architecture = dict(architecture_data)
    state = {"intent_model": intent, "srs": dict(old_srs), "architecture": architecture_data}
    resolved = _resolve_answered_open_questions(state, persist=None, architecture=architecture_data)
    new_srs = srs_document_data(intent, state["srs"])
    new_srs.setdefault("session_id", intent["session_id"])
    new_srs.setdefault("intent_model_ref", intent["session_id"])
    srs = SRSDocument.model_validate(coerce_for_model(new_srs, SRSDocument))
    architecture = ArchitectureOverview.model_validate(coerce_for_model(architecture_data, ArchitectureOverview))

    # Compare canonical contract instances so defaulted fields and version
    # stamps cannot create false semantic changes.
    old_srs_contract = SRSDocument.model_validate(coerce_for_model(old_srs, SRSDocument)).model_dump(mode="json")
    old_arch_contract = ArchitectureOverview.model_validate(
        coerce_for_model(old_architecture, ArchitectureOverview)).model_dump(mode="json")
    before = {"srs": _semantic_hash(old_srs_contract), "architecture": _semantic_hash(old_arch_contract)}
    after = {"srs": _semantic_hash(srs.model_dump(mode="json")),
             "architecture": _semantic_hash(architecture.model_dump(mode="json"))}
    changed = [name for name in before if before[name] != after[name]]
    render_changed = []
    rendered_srs = render_srs(srs)
    rendered_arch = render_architecture(architecture, greenfield=baseline["mode"] == "greenfield",
                                       conops_present=conops_present)
    for name, text in (("srs.md", rendered_srs), ("architecture.md", rendered_arch)):
        old = (source / name).read_text() if (source / name).exists() else ""
        if old != text: render_changed.append(name)

    _atomic_write(destination / "srs.json", json.dumps(new_srs, indent=2))
    _atomic_write(destination / "srs.md", rendered_srs)
    _atomic_write(destination / "architecture.json", architecture.model_dump_json(indent=2))
    _atomic_write(destination / "architecture.md", rendered_arch)
    cause = ("semantic basis changed during deterministic reconciliation: " + ", ".join(changed)
             if changed else "deterministic views refreshed; semantic basis unchanged")
    status_path = destination / "artifact-status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"artifacts": {}}
    if changed:
        for name in ("items.json", "tests.json", "validation.json", "specs", "AGENTS.md", "traceability.json", "manifest.json"):
            status["artifacts"][name] = {"state": "stale", "cause": cause,
                                             "basis_hashes_before": before,
                                             "basis_hashes_after": after}
    _atomic_write(status_path, json.dumps(status, indent=2))
    audit = {"event": "artifacts.reconciled", "actor": "local user", "at": datetime.now().astimezone().isoformat(),
             "equivalence_version": _SEMANTIC_EQUIVALENCE_VERSION, "source": str(source),
             "destination": str(destination), "semantic_changed": changed,
             "render_changed": render_changed, "resolved_questions": resolved,
             "downstream": "stale" if changed else "valid"}
    with (destination / "audit.jsonl").open("a") as fh:
        fh.write(json.dumps(audit, sort_keys=True) + "\n")
    refresh_docs(destination, intent=intent, invalidate="items" if changed else None)
    return audit

def main(argv=None, io=None) -> None:
    """Command-line entry. A trusted caller running the generator in its own process passes argv and a cli_io.RunIO
    for one run; the command line passes neither and reads the process arguments, stdin, stdout and stderr."""
    if io is None:
        return _main(argv)
    with cli_io.bind(io):
        return _main(argv)


def _main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("intent_model", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="artifact directory (default: <workspace>/artifacts/artifacts-YYYY-MM-DD_HH-MM-SS, "
                         "from the session id; VMODEL_ARTIFACTS_DIR overrides the root)")
    ap.add_argument("--review", action="store_true",
                    help="opt-in human review gates after the SRS and architecture stages")
    ap.add_argument("--expert-opinion", action="store_true",
                    help="simulate gate decisions with an LLM for test runs; records pending human review")
    ap.add_argument("--auto-approve", action="store_true",
                    help="accept SRS/architecture gates unattended; every decision is written to audit.jsonl")
    ap.add_argument("--from", dest="from_stage", metavar="STAGE", default=None,
                    help="rewind the checkpoint: re-run STAGE and everything after it under the "
                         "current code, earlier stages preserved (stages: needs, srs, architecture, "
                         "items, tests, review, assemble)")
    ap.add_argument("--to", dest="to_stage", choices=("needs", "srs", "architecture", "items", "tests", "validation"),
                    help="stop after this stage; completed state remains resumable")
    ap.add_argument("--reconcile-artifacts", action="store_true",
                    help="zero-LLM deterministic reconciliation of existing SRS/architecture and views")
    ap.add_argument("--reconcile-out", type=Path, default=None,
                    help="write reconciliation to a fresh comparison generation instead of the live bundle")
    ap.add_argument("--only-stage", choices=tuple(_ONLY_STAGE_INPUTS), default=None,
                    help="run exactly one stage from an existing completed artifact set; never mutates source")
    ap.add_argument("--source-artifacts", type=Path, default=None,
                    help="completed artifacts directory supplying upstream inputs for --only-stage")
    ap.add_argument("--system-baseline", type=Path, default=None,
                    help="versioned SystemBaseline JSON; existing repo manifest or explicit greenfield declaration")
    ap.add_argument("--conops", type=Path, default=None,
                    help="founder-confirmed conops.json from interview-engine; omit only for legacy sessions")
    ap.add_argument("--grounding", choices=("off", "jev"), default="off",
                    help="JEV-D10 rule 3: Jev grounding check at review (needs TYPESAFE_API_KEY and typesafe-sdk; no fallback; off = NOT CHECKED)")
    ap.add_argument("--dispose", type=Path, default=None, metavar="FOUNDER_SCRIPT.json",
                    help="JEV-D10: apply a founder-authored disposition script (exact selectors, hash-bound) to the findings ledger")
    ap.add_argument("--repair", default=None, metavar="FINDING_ID[,FINDING_ID]",
                    help="JEV-D10: explicit scoped repair of the named open review findings (bounded attempts; never closes a finding)")
    ap.add_argument("--quick", action="store_true",
                    help="with --repair: no stops. Auto-apply the SRS fix (before/after saved as evidence), skip items and tests, "
                         "re-review only the repaired requirements, then stop and print what closed and what is still open. "
                         "Evidence run: the bundle is NOT assembled and items/tests are stale until a full run")
    ap.add_argument("--approve-repair", action="store_true",
                    help="apply the pending repair proposal (repair-proposal.md) and continue to items, tests, review")
    ap.add_argument("--reject-repair", action="store_true",
                    help="drop the pending repair proposal; the SRS stays as it was and findings stay open")
    ap.add_argument("--refresh-technical-choices", action="store_true",
                    help="targeted metadata-only framing for an existing architecture; may call the configured model per missing slot; never runs architecture/components/ADRs")
    ap.add_argument("--require-decisions", action="store_true",
                    help="attended run: require a usable technical design default or founder override before items. "
                         "Off by default: slots with neither raise an open ledger finding")
    ap.add_argument("--export-handoff",action="store_true",help="zero-LLM validated coding-agent handoff from completed outputs")
    ap.add_argument("--preflight", action="store_true",
                    help="validate inputs and checkpoint compatibility without building a model or spending a call")
    args = ap.parse_args(argv)
    from .cooperative_run import cli_stop_binding
    if cli_stop_binding() is not None and (args.only_stage or args.reconcile_artifacts or args.export_handoff):
        raise UserFacingError("this auxiliary CLI mode has no cooperative Stop binding",
                              "its separate publication paths are not reviewed",
                              "use the normal job-bound producer path; no work was started")
    if bool(args.only_stage) != bool(args.source_artifacts):
        ap.error("--only-stage and --source-artifacts must be used together")
    if args.only_stage and args.from_stage:
        ap.error("--only-stage cannot be combined with --from")
    if args.only_stage and args.review:
        ap.error("--only-stage cannot be combined with --review")
    if args.reconcile_out and not args.reconcile_artifacts:
        ap.error("--reconcile-out requires --reconcile-artifacts")
    if args.reconcile_artifacts and (args.from_stage or args.only_stage or args.to_stage or args.review):
        ap.error("--reconcile-artifacts cannot be combined with --from, --to, --only-stage, or --review")

    if not args.intent_model.exists():
        raise UserFacingError(
            f"intent model file not found: {args.intent_model}",
            "wrong path, or the interview has not been run yet.",
            "run the interview first (interview-engine CLI) or pass the correct path to out/<session>/intent_model.json.",
        )
    try:
        intent = json.loads(args.intent_model.read_text())
    except json.JSONDecodeError as e:
        raise UserFacingError(
            f"{args.intent_model} is not valid JSON ({e}).",
            "the file was truncated or hand-edited into an invalid state.",
            "re-run the interview-engine to regenerate the intent model.",
        ) from e
    if "session_id" not in intent:
        raise UserFacingError(
            f"{args.intent_model} parses as JSON but is not an intent model (no session_id).",
            "wrong file passed in.",
            "pass the intent_model.json written by the interview-engine CLI.",
        )
    from vmodel_contracts import ContractVersionMismatch, check_compatible

    try:
        check_compatible(intent)
    except ContractVersionMismatch as e:
        raise UserFacingError(
            f"this intent model was produced against vmodel-contracts {e.have}, but this spec-generator reads {e.want.split('.')[0]}.{e.want.split('.')[1]}.x.",
            "the interview ran before a contract upgrade.",
            f"if the schema is otherwise compatible, re-stamp it: {_migration_command(args.intent_model)} "
            "(it validates against the current schema first and refuses on real drift); "
            "otherwise re-run the interview (minutes on a local model) to regenerate the intent model.",
        ) from e
    conops = None
    if args.conops is not None:
        from vmodel_contracts import ConOpsDocument
        try:
            conops = ConOpsDocument.model_validate_json(args.conops.read_text()).model_dump(mode="json")
        except Exception as e:
            raise UserFacingError(
                f"Concept of Operations file is invalid: {args.conops}.",
                f"contract validation failed: {str(e)[:240]}",
                "pass the conops.json written by interview-engine; do not hand-edit a concept decision.") from e
        if conops["session_id"] != intent["session_id"]:
            raise UserFacingError("Concept of Operations belongs to a different session.",
                                  f"intent={intent['session_id']}, conops={conops['session_id']}",
                                  "pass intent_model.json and conops.json from the same interview output directory.")
    else:
        print("WARNING: no Concept of Operations supplied (legacy session) - any product-concept "
              "choice in architecture is unverified and must be challenged.", file=cli_io.stderr())

    if args.system_baseline is None:
        raise UserFacingError(
            "--system-baseline is required.",
            "architecture must distinguish an existing system from an explicit greenfield product.",
            "pass a checked SystemBaseline JSON; for a new product copy examples/system-baseline.greenfield.json.",
        )
    baseline = _load_system_baseline(args.system_baseline)
    print(f"System baseline: {args.system_baseline.resolve()} "
          f"(id={baseline['baseline_id']}, mode={baseline['mode']}, "
          f"{len(baseline['repositories'])} repos)", file=cli_io.stderr())
    sid = intent["session_id"]
    if args.only_stage:
        _run_only_stage(args, intent, sid)
        return
    outdir = args.out if args.out is not None else _default_outdir(args.intent_model, sid)
    if args.export_handoff:
        from .handoff import build_handoff
        handoff=build_handoff(args.intent_model.parent,outdir)
        print(json.dumps(handoff), file=cli_io.stdout());return
    if args.preflight:
        import sqlite3
        cp=outdir / "checkpoints.sqlite"
        if cp.exists():
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn=sqlite3.connect(str(cp),check_same_thread=False)
            probe=build_graph(UsageLog(sid),checkpointer=SqliteSaver(conn,serde=checkpoint_serde()),review=args.review)
            prior=probe.get_state({"configurable":{"thread_id":sid}}).values
            if prior:
                if _resume_constraint_content(prior.get("conops")) != _resume_constraint_content(conops):
                    raise UserFacingError("preflight: checkpoint Concept of Operations is incompatible.",
                        "the existing output was started without this CONOPS or with a different one.",
                        "choose a fresh -o directory, or pass the original CONOPS used by this checkpoint.")
                if prior.get("system_baseline") is not None and _resume_constraint_content(prior["system_baseline"]) != _resume_constraint_content(baseline):
                    raise UserFacingError("preflight: checkpoint system baseline is incompatible.",
                        "the existing output was generated under a different architecture anchor.",
                        "choose a fresh -o directory, or explicitly rerun from architecture under the new baseline.")
        print(f"Preflight OK: session={sid}; seed=intent_model,system_baseline,conops; "
              f"baseline={baseline['baseline_id']}; conops={conops is not None}; "
              f"checkpoint={'compatible' if cp.exists() else 'fresh'}", file=cli_io.stdout())
        return
    if args.reconcile_artifacts:
        audit = _reconcile_artifacts(intent, baseline, outdir, args.reconcile_out,
                                     conops_present=conops is not None)
        print("Reconciled artifacts with ZERO LLM calls.", file=cli_io.stderr())
        print(f"Semantic changes: {', '.join(audit['semantic_changed']) or '(none)'}", file=cli_io.stderr())
        print(f"Downstream: {audit['downstream']} ({audit['destination']})", file=cli_io.stderr())
        return
    outdir.mkdir(parents=True, exist_ok=True)
    refresh_docs(outdir, intent=intent, conops=conops, start=True)
    from vmodel_contracts.runtime import install_plain_log
    run_log = outdir / "run.log"
    cli_io.set_stderr(install_plain_log(run_log, cli_io.stderr()))
    # Per-call telemetry is part of every normal run, not an experiment-only
    # opt-in. Preserve an explicit external sink, otherwise keep it beside the bundle.
    os.environ.setdefault("VMODEL_LLM_LOG", str(outdir / "llm-calls.jsonl"))
    print(f"Run log: {run_log.resolve()} (plain text)", file=cli_io.stderr())
    (outdir / "specs").mkdir(exist_ok=True)
    _atomic_write(outdir / "system-baseline.json", json.dumps(baseline, indent=2))

    log = UsageLog(sid, sink=jsonl_sink(outdir / "usage.jsonl"))
    bus = ProgressBus()
    bus.subscribe(stderr_printer())
    from .structured_progress import StructuredProgress
    from .structured_progress import STAGE_PHASE
    emitter=StructuredProgress(os.environ.get("VMODEL_PROGRESS_TASK_ID") or sid,sid,STAGE_PHASE.get(args.to_stage or "needs","needs"))
    bus.subscribe(emitter.subscriber)
    from . import structured_progress
    structured_progress.active_sink=emitter

    # Disk checkpoints: a crash mid-generation resumes from the last completed
    # stage on rerun (same command) - no stage is paid for twice.
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(str(outdir / "checkpoints.sqlite"), check_same_thread=False)
    from .artifact_lifecycle import ArchitectureLifecycle, stable_hash
    architecture_lifecycle = ArchitectureLifecycle(outdir, sid, baseline)
    partials = _make_partials(outdir, architecture_lifecycle)
    grounding = None
    if args.grounding == "jev":
        from .grounding import ProviderUnavailable, TypeSafeGroundingProvider
        try:
            grounding = TypeSafeGroundingProvider(api_key=os.environ.get("TYPESAFE_API_KEY"))
        except ProviderUnavailable as exc:
            raise UserFacingError(f"Jev grounding is unavailable: {exc}", "no fallback provider is used.",
                                  "install typesafe-sdk and set TYPESAFE_API_KEY, or run with --grounding off (recorded as NOT CHECKED).") from exc
    persist_fn = _make_persist(intent, outdir, greenfield=baseline["mode"] == "greenfield",
                               conops_present=conops is not None,
                               architecture_lifecycle=architecture_lifecycle)
    from .cooperative_run import cli_stop_binding
    stop_binding=cli_stop_binding()
    graph = build_graph(log, stop_fence=stop_binding[0] if stop_binding else None,
                        stop_generation=stop_binding[1] if stop_binding else None,
                        checkpointer=SqliteSaver(conn, serde=checkpoint_serde()), progress=bus, review=args.review, expert_opinion=args.expert_opinion, grounding=grounding, require_decisions=args.require_decisions,
                        persist=persist_fn,
                        partials=partials, stop_after=args.to_stage)
    from . import model_config as _mc

    if args.from_stage:
        os.environ["SPEC_GENERATOR_FROM"] = args.from_stage  # guardrail errors must not echo --from back
    print(f"Generating bundle for session {sid} (crash? {_mc._rerun_guidance('')})", file=cli_io.stderr())
    print(f"Models: {_mc.describe_models()}", file=cli_io.stderr())
    print(f"Bundle dir: {outdir.resolve()}", file=cli_io.stderr())
    print("  progress persists there per item (*.partial.json as items close; stage artifacts land "
          "as each stage completes)\n", file=cli_io.stderr())
    cfg = {"configurable": {"thread_id": sid}}
    prior = graph.get_state(cfg).values
    if prior and prior.get("repair_proposal") and not (args.approve_repair or args.reject_repair):
        raise UserFacingError("a repair proposal is waiting for your decision.",
                              f"the repaired requirements are in {outdir / 'repair-proposal.md'} and are NOT applied.",
                              "read it, then rerun with --approve-repair (apply and continue) or --reject-repair (drop it).")
    if args.approve_repair or args.reject_repair:
        prior = _resolve_repair_proposal(graph, cfg, prior, args, persist_fn)
    if args.quick and not args.repair:
        raise UserFacingError("--quick only applies to --repair.", "nothing was changed.",
                              "use --repair <finding ids> --quick, or drop --quick.")
    prior, dispose_only = _pre_run_actions(graph, cfg, prior, args)
    if dispose_only:
        print("Disposition recorded in the findings ledger. Nothing was resumed or regenerated; "
              "rerun without --dispose to continue.", file=cli_io.stderr())
        return
    prior_ledger_before_run = (prior or {}).get("findings_ledger")
    resume = bool(prior)
    if resume:
        prior_conops = prior.get("conops")
        if _resume_constraint_content(prior_conops) != _resume_constraint_content(conops):
            if prior_conops and conops:
                reason = "the supplied Concept of Operations differs from the checkpoint"
            elif conops:
                reason = "this legacy checkpoint predates the supplied Concept of Operations"
            else:
                reason = "the checkpoint has a Concept of Operations but this command omitted --conops"
            raise UserFacingError(
                "cannot resume with a different Concept of Operations constraint.",
                reason + "; completed Stakeholder Needs and downstream artifacts are grounded in the prior concept state.",
                "start a new output directory for this concept, or rerun the original command with the same --conops file."
            )
    if (resume and prior.get("system_baseline") is not None
            and _resume_constraint_content(prior["system_baseline"]) == _resume_constraint_content(baseline)):
        # Keep the original sealed anchor, including its stamp, when only
        # versions changed. The adapter is shared with persistence callbacks;
        # do not replace it or rewrite historical lifecycle hashes.
        architecture_lifecycle.anchor_hash = stable_hash(prior["system_baseline"])
    if resume and prior.get("architecture"):
        validated_architecture({**prior["architecture"], "session_id": sid})
        architecture_lifecycle.bootstrap_complete(
            prior["architecture"], prior.get("system_baseline") or baseline)
    baseline_changed = (resume and prior.get("system_baseline") is not None
                        and _resume_constraint_content(prior["system_baseline"]) != _resume_constraint_content(baseline))
    if resume and not baseline_changed:
        next_artifact = "gate_architecture" if args.review else "items"
        # Check a previously recorded continuation even if SQLite was already
        # reconnected before the crash. Never reseal an arbitrary partial.
        if (prior.get("architecture") and architecture_lifecycle.state.value == "partial"):
            body = {**prior["architecture"], "session_id": sid}
            validated_architecture(body)
            architecture_lifecycle.resume(next_artifact, content=body,
                compatible_anchor_hashes=(stable_hash(baseline),))
        bounded_stage = _resume_after_bounded_stop(graph, cfg, prior)
        if bounded_stage:
            if bounded_stage == "architecture":
                body = {**prior["architecture"], "session_id": sid}
                validated_architecture(body)
                architecture_lifecycle.resume(next_artifact, content=body,
                    compatible_anchor_hashes=(stable_hash(baseline),))
            destination = f"the requested '{args.to_stage}' boundary" if args.to_stage else "the full pipeline"
            print(f"Checkpoint intentionally stopped after '{bounded_stage}' - continuing through {destination}.", file=cli_io.stderr())
    # --from needs/srs regenerates the SRS and its review, so an unresolved old review must not stop the rewind.
    if resume and args.from_stage not in ('needs','srs') and (prior.get('semantic_reviews') or {}).get('srs'):
        from .semantic_review import load_reviewed_srs
        try:resolved=load_reviewed_srs(intent,outdir,prior)
        except ValueError as exc:
            raise UserFacingError('SRS candidate review is not ready.',str(exc),'Review every flagged SRS item with Remove, Replace, or Accept before continuing.') from exc
        if resolved:
            candidate,reviewed=resolved
            if _srs_review_already_applied(prior,candidate,reviewed):
                # update_state(as_node=<predecessor of architecture>) re-queues architecture, so
                # re-applying an unchanged review on every resume re-ran a finished stage (39 min).
                print('Reviewed SRS items already applied to this checkpoint; continuing at the next stage.',file=cli_io.stderr())
            else:
                reviews={**prior['semantic_reviews'],'srs':reviewed.model_dump(mode='json'),
                         SRS_APPLIED_KEY:_srs_candidate_digest(candidate)}
                graph.update_state(cfg,{'srs':candidate,'semantic_reviews':reviews},as_node=_predecessor('architecture',args.review))
                prior={**prior,'srs':candidate,'semantic_reviews':reviews}
                print('Reviewed SRS items applied and structurally validated; continuing without rewriting accepted items.',file=cli_io.stderr())
    if args.refresh_technical_choices:
        _refresh_technical_choices(graph, cfg, prior, outdir, persist_fn, log, bus)
        return
    if resume:
        prior = _apply_saved_architecture_decisions(graph, cfg, prior, outdir, persist_fn)
    if resume:
        print("Checkpoint found - resuming; completed stages are NOT re-run.", file=cli_io.stderr())
        # backfill-persist artifacts for stages completed BEFORE incremental
        # persistence shipped: resume skips completed stages (persist never
        # fires for them), so without this the bundle dir stays empty. State
        # is authoritative - overwrite any stale files.
        # Backfill under the checkpoint's anchor. A changed current anchor is
        # recorded as stale/invalidate/restart immediately below; it must not
        # relabel or rewrite the old complete architecture first.
        checkpoint_lifecycle = ArchitectureLifecycle(
            outdir, sid, prior.get("system_baseline") or baseline)
        written = _backfill_persist(_make_persist(intent, outdir,
                                                        conops_present=conops is not None,
                                                        architecture_lifecycle=checkpoint_lifecycle), prior)
        if written:
            print(f"Artifacts for already-completed stages written to the bundle dir ({', '.join(written)}).",
                  file=cli_io.stderr())
    if resume and prior.get("system_baseline") is not None and _resume_constraint_content(prior["system_baseline"]) != _resume_constraint_content(baseline):
        # A completed stage is only "completed" under the baseline it was
        # built with. Resuming under a DIFFERENT anchor must never silently
        # re-label old output (2026-09-13: the poisoned overnight six-repo
        # bundle re-bundled as "greenfield" and nearly shipped). needs/srs
        # never read the baseline; architecture and everything downstream do
        # - invalidate from architecture, loudly, and re-run.
        old_b = prior["system_baseline"]
        print("WARNING: the --system-baseline for this run differs from the checkpoint's:", file=cli_io.stderr())
        print(f"  checkpoint: id={old_b.get('baseline_id')}, mode={old_b.get('mode')}, "
              f"{len(old_b.get('repositories') or [])} repos", file=cli_io.stderr())
        print(f"  this run:   id={baseline['baseline_id']}, mode={baseline['mode']}, "
              f"{len(baseline['repositories'])} repos", file=cli_io.stderr())
        print("  architecture and everything downstream were built under the old anchor - "
              "invalidating them (needs/srs do not depend on the baseline and are preserved).",
              file=cli_io.stderr())
        architecture_lifecycle.reanchor(["architecture", "items", "tests", "assembly", "manifest", "traceability"])
        graph.update_state(cfg, {"system_baseline": baseline, **_CLEAR_DOWNSTREAM["architecture"]},
                           as_node=_predecessor("architecture", args.review))
        for key in _PARTIAL_DOWNSTREAM.get("architecture", ()):
            saved = partials.load(key) or {}
            if len(saved):
                print(f"  discarding {len(saved)} preserved item(s) ({key}) - they will be regenerated "
                      "under the new baseline.", file=cli_io.stderr())
            partials.clear(key)
        for stale in ("architecture.json", "architecture.md", "items.json", "tests.json", "validation.json",
                      "AGENTS.md", "manifest.json", "traceability.json", "review-fixes.json"):
            (outdir / stale).unlink(missing_ok=True)
        for stale in (outdir / "specs").glob("SPEC-*.md"):
            stale.unlink()
        print("  stale downstream artifacts removed from the bundle dir; re-running from 'architecture'.",
              file=cli_io.stderr())
    elif resume and prior.get("system_baseline") is None:
        # --system-baseline is authoritative for THIS invocation. Old
        # checkpoints predate the baseline feature entirely (the 2026-09-12
        # --from architecture KeyError 'system_baseline'): merge the flag
        # value into the resumed state so baseline-consuming stages see it.
        # Write AS the last baseline-independent node. A bare update on a
        # failed/interrupted review graph can fork from the failed architecture
        # task and leave that task's state unchanged (the live 2026-09-15
        # accepted-SRS resume did exactly that). Re-anchoring at gate_srs
        # preserves needs + accepted SRS and schedules architecture next.
        graph.update_state(cfg, {"system_baseline": baseline},
                           as_node=_predecessor("architecture", args.review))
        print("System baseline merged into the resumed run (its checkpoint predates the baseline feature).",
              file=cli_io.stderr())
    if args.from_stage:
        _invalidate_from(graph, cfg, args.from_stage, args.review)
        for key in _PARTIAL_DOWNSTREAM.get(args.from_stage, ()):
            saved = partials.load(key) or {}
            if len(saved):
                print(f"warning: --from {args.from_stage} discards {len(saved)} preserved item(s) "
                      f"({key}) - they will be regenerated.", file=cli_io.stderr())
            partials.clear(key)
        refresh_docs(outdir, intent=intent, conops=conops, invalidate=args.from_stage)
        resume = args.from_stage != "needs"
        if resume:
            kept = [s for s in STAGE_ORDER[:STAGE_ORDER.index(args.from_stage)] if prior.get(s)]
            print(f"Re-running from stage '{args.from_stage}' under the current code - "
                  f"preserved: {', '.join(kept) or '(nothing)'}; downstream state invalidated.",
                  file=cli_io.stderr())
    if args.review:
        result = _run_with_gates(graph, cfg,
                                 {"intent_model": intent, "system_baseline": baseline, "conops": conops},
                                 outdir, resume=resume, auto_approve=args.auto_approve)
        if args.expert_opinion:
            opinions=(graph.get_state(cfg).values.get("expert_opinions") or [])
            _atomic_write(outdir / "expert-opinions.json",json.dumps({"mode":"simulated_expert_opinion","founder_approved":False,"human_review_required":True,"decisions":opinions},indent=2))
            for opinion in opinions:_append_audit(outdir/"audit.jsonl",{"event":"gate.simulated_expert_opinion",**opinion,"founder_approved":False})
    else:
        result = graph.invoke(None, cfg) if resume else graph.invoke(
            {"intent_model": intent, "system_baseline": baseline, "conops": conops}, cfg)

    refresh_docs(outdir, intent=intent, conops=conops, state=result)
    if result.get("quick_stop"):
        text = _quick_summary(prior_ledger_before_run, result.get("findings_ledger"), outdir)
        _atomic_write(outdir / "quick-repair-summary.md", text)
        print(text, file=cli_io.stdout())
        return
    if args.to_stage:
        print(f"Stopped after stage '{args.to_stage}' as requested. Checkpoint and stage artifacts are resumable in {outdir}.", file=cli_io.stdout())
        return

    b = result.get("bundle")
    if b is None:
        state_keys = ", ".join(k for k, v in result.items() if v is not None)
        raise UserFacingError(
            "assembly produced no bundle after the pipeline reported completion.",
            f"code-level lifecycle error: terminal checkpoint contains [{state_keys}] but no bundle; "
            "a bounded --to checkpoint may not have been reconnected to the full graph.",
            "rerun the same command after updating spec-generator; do not change models or discard completed stages.",
        )

    def validate_artifact(label, fn):
        try:
            return fn()
        except Exception as e:
            from pydantic import ValidationError
            category = "contract validation" if isinstance(e, ValidationError) else "code exception"
            guidance = ("regenerate only that producing stage if the contract error persists; model escalation is optional"
                        if isinstance(e, ValidationError) else
                        "update/report spec-generator; changing the model cannot fix a code exception")
            raise UserFacingError(
                f"bundle render failed while validating artifact '{label}'.",
                f"{category} in assemble/render: {type(e).__name__}: {str(e)[:300]}",
                guidance + "; completed checkpoint stages remain preserved.",
            ) from e

    needs = validate_artifact("stakeholder needs", lambda: StakeholderNeedsSet.model_validate(
        normalize_enums(coerce_for_model(b.get("_needs"), StakeholderNeedsSet), StakeholderNeedsSet)))
    reqs = validate_artifact("requirements", lambda: RequirementSet.model_validate(
        normalize_enums(coerce_for_model(b.get("_requirements"), RequirementSet), RequirementSet)))
    arch = validate_artifact("architecture", lambda: ArchitectureOverview.model_validate(
        coerce_for_model(b.get("_architecture"), ArchitectureOverview)))
    items = validate_artifact("item specs", lambda: [ItemSpec.model_validate(coerce_for_model(i, ItemSpec))
                                                       for i in b.get("items") or []])
    manifest = validate_artifact("manifest", lambda: SpecManifest.model_validate(
        coerce_for_model(b.get("_manifest"), SpecManifest)))
    punch = validate_artifact("punch list", lambda: [PunchListItem.model_validate(coerce_for_model(x, PunchListItem))
                                                       for x in b.get("punch_list") or []])


    _atomic_write((outdir / "needs.md"),render_needs(needs))
    srs = SRSDocument.model_validate(coerce_for_model(b["_srs"], SRSDocument))
    _atomic_write((outdir / "srs.md"),render_srs(srs))
    _atomic_write(outdir / "architecture.md",
        render_architecture(arch, greenfield=baseline["mode"] == "greenfield",
                            conops_present=conops is not None))
    _atomic_write((outdir / "architecture.json"),arch.model_dump_json(indent=2))
    for i in items:
        _atomic_write((outdir / "specs" / f"{i.spec_id}.md"),render_spec_file(i))
    _atomic_write((outdir / "AGENTS.md"),render_agents_md(arch, items, manifest, punch))
    _atomic_write((outdir / "traceability.json"),json.dumps(b["traceability"], indent=2))
    _atomic_write((outdir / "manifest.json"),manifest.model_dump_json(indent=2))
    _atomic_write((outdir / "tests.json"),json.dumps(b["_tests"], indent=2))
    if b.get("_validation"):
        _atomic_write((outdir / "validation.json"),json.dumps(b["_validation"], indent=2))

    refresh_docs(outdir, intent=intent, conops=conops, event="assemble", state=result)
    if result.get("violations"):
        print("WARNING - traceability violations (in punch list):", file=cli_io.stdout())
        for v in result["violations"]:
            print(f"  - {v}", file=cli_io.stdout())
    print(f"Bundle written to {outdir}/  |  generation cost: ${log.total_usd():.4f}", file=cli_io.stdout())


if __name__ == "__main__":
    run_cli(lambda: run_with_system_awake(main))



