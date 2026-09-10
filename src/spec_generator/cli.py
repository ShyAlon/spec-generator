"""Intent model in, repo-ready bundle out.

Usage: python -m spec_generator.cli out/<session>/intent_model.json -o bundle/

Bundle layout (written INCREMENTALLY as each stage completes, atomic
tmp+rename - a late failure leaves all completed work on disk):
  needs.json / needs.md            after stage 1/6
  srs.json / srs.md                after stage 2/6 (re-written if review fixes apply)
  architecture.json / architecture.md  after stage 3/6
  items.json + specs/<SPEC-id>.md  after stage 4/6
  tests.json                       after stage 5/6
  AGENTS.md / traceability.json / manifest.json   at assembly (stage 6/6)
  gate_srs.* / gate_architecture.* (--review)     at the gates
  usage.jsonl / checkpoints.sqlite                run bookkeeping
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from vmodel_contracts import (
    AcceptanceTest, ArchitectureOverview, ItemSpec, PunchListItem,
    RequirementSet, SpecManifest, SRSDocument, StakeholderNeedsSet,
)

from .errors import UserFacingError, run_cli
from .graph import build_graph, checkpoint_serde, srs_document_data, _legacy_arch_fill
from .progress import ProgressBus, stderr_printer
from .repair import coerce_for_model, normalize_enums
from .render import (render_agents_md, render_architecture, render_needs,
                     render_srs, render_spec_file)
from .telemetry import UsageLog, jsonl_sink


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
            "rerun the same command - the checkpoint resumes at the gate; if it repeats, the stage "
            "produced a contract-breaking draft its own validation missed - please report it as a bug.",
        ) from e


def _backfill_persist(persist, values: dict) -> list[str]:
    """Resume companion: persist artifacts for stages completed BEFORE
    incremental persistence shipped (resume skips completed stages, so
    persist() never fires for them and the bundle dir stays empty). State is
    authoritative - overwrite stale files. Returns the stages written."""
    written = []
    for stage in ("needs", "srs", "architecture", "items", "tests"):
        if values.get(stage):
            persist(stage, values[stage], values)
            written.append(stage)
    return written


def _atomic_write(path: Path, text: str) -> None:
    """tmp + rename: a crash mid-write can never leave a torn artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _make_partials(outdir: Path):
    """Per-item resume WITHIN an authoring stage (his ask 2026-09-10, after a
    Ctrl+C at T-34 of 36 reran everything): completed fills/bodies land in a
    partial file the moment they close; a rerun of the stage skips them, the
    same way completed stages are skipped. Cleared when the stage completes;
    --from <stage> deletes partials for that stage and everything downstream
    (explicit rewind forces real re-runs)."""
    FILES = {"srs_fills": "srs_fills.partial.json",
             "items": "items.partial.json",
             "tests_fills": "tests_fills.partial.json"}

    class _Partials:
        def load(self, key: str):
            p = outdir / FILES[key]
            return json.loads(p.read_text()) if p.exists() else None

        def save(self, key: str, data) -> None:
            _atomic_write(outdir / FILES[key], json.dumps(data, indent=2))

        def clear(self, key: str) -> None:
            p = outdir / FILES[key]
            if p.exists():
                p.unlink()

    return _Partials()


# partial files invalidated by --from <stage> (that stage and everything after)
_PARTIAL_DOWNSTREAM = {"srs": ("srs_fills", "items", "tests_fills"),
                      "architecture": ("items", "tests_fills"),
                      "items": ("items", "tests_fills"),
                      "tests": ("tests_fills",)}


def _make_persist(intent: dict, outdir: Path):
    """The incremental bundle writer (his ask 2026-09-09: 'persist them
    intermittently so we'll have some of the data'). Fires after each stage;
    the final assembly writes remain the canonical full bundle."""
    sid = intent["session_id"]

    def _persist(stage: str, data, state: dict) -> None:
        if stage == "needs":
            body = {**data, "session_id": sid}
            _atomic_write(outdir / "needs.json", json.dumps(body, indent=2))
            needs = StakeholderNeedsSet.model_validate(normalize_enums(coerce_for_model(body, StakeholderNeedsSet), StakeholderNeedsSet))
            _atomic_write(outdir / "needs.md", render_needs(needs))
        elif stage == "srs":
            doc = srs_document_data(intent, {**data, "session_id": sid, "intent_model_ref": sid})
            _atomic_write(outdir / "srs.json", json.dumps(doc, indent=2))
            srs = SRSDocument.model_validate(coerce_for_model(doc, SRSDocument))
            _atomic_write(outdir / "srs.md", render_srs(srs))
        elif stage == "architecture":
            body = _legacy_arch_fill({**data, "session_id": sid})  # pre-0.8.0 checkpoints lack the ADR-0021 fields
            _atomic_write(outdir / "architecture.json", json.dumps(body, indent=2))
            arch = ArchitectureOverview.model_validate(coerce_for_model(body, ArchitectureOverview))
            _atomic_write(outdir / "architecture.md", render_architecture(arch))
        elif stage == "items":
            _atomic_write(outdir / "items.json", json.dumps(data, indent=2))
            for i in data:
                item = ItemSpec.model_validate(coerce_for_model(i, ItemSpec))
                _atomic_write(outdir / "specs" / f"{item.spec_id}.md", render_spec_file(item))
        elif stage == "tests":
            _atomic_write(outdir / "tests.json", json.dumps(data, indent=2))

    return _persist


STAGE_ORDER = ("needs", "srs", "architecture", "items", "tests", "review", "assemble")

# What a --from rewind invalidates: the named stage's own output and every
# downstream product (LangGraph state has no key delete; overwrite with
# empties - review_round resets so the review loop runs fresh).
_CLEAR_DOWNSTREAM = {
    "srs": {"srs": None, "architecture": None, "items": None, "tests": None,
            "review_round": 0, "review_fixes": [], "bundle": None, "violations": [],
            "guard_flags": []},
    "architecture": {"architecture": None, "items": None, "tests": None,
                     "review_round": 0, "review_fixes": [], "bundle": None, "violations": []},
    "items": {"items": None, "tests": None, "review_round": 0, "review_fixes": [],
              "bundle": None, "violations": []},
    "tests": {"tests": None, "review_round": 0, "review_fixes": [], "bundle": None,
              "violations": []},
    "review": {"review_round": 0, "review_fixes": [], "bundle": None, "violations": []},
    "assemble": {"bundle": None, "violations": []},
}


def _predecessor(stage: str, review_mode: bool) -> str:
    """The node the checkpoint is rewritten AS (update_state as_node=...), so
    resume continues at the stage the founder asked to re-run. Review-mode
    graphs route through the gate nodes."""
    return {"srs": "needs",
            "architecture": "gate_srs" if review_mode else "srs",
            "items": "gate_architecture" if review_mode else "architecture",
            "tests": "items",
            "review": "tests",
            "assemble": "review"}[stage]


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
    graph.update_state(cfg, dict(_CLEAR_DOWNSTREAM[stage]),
                       as_node=_predecessor(stage, review_mode))


def _run_with_gates(graph, cfg: dict, intent: dict, outdir, resume: bool = False) -> dict:
    """ADR-0019 review gates (opt-in via --review): after the SRS and after the
    architecture, the founder accepts, edits (JSON, contract-validated), sends
    feedback (stage re-runs with it), or rejects. Crash mid-gate: rerun the
    same command - the checkpoint re-renders and re-asks."""
    import subprocess

    from langgraph.types import Command

    result = graph.invoke(None, cfg) if resume else graph.invoke({"intent_model": intent}, cfg)
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
                  "do NOT rerun the command.\n", file=sys.stderr)
            result = graph.invoke({"intent_model": intent}, cfg)
            continue
        (outdir / f"gate_{gate}.md").write_text(md)
        (outdir / f"gate_{gate}.json").write_text(json.dumps(artifact, indent=2))
        print(f"\n===== REVIEW GATE: the {title} =====\n", file=sys.stderr)
        print(md, file=sys.stderr)
        print(f"(full artifact: {outdir}/gate_{gate}.md | editable JSON: {outdir}/gate_{gate}.json)", file=sys.stderr)
        while True:
            action = input("[a]ccept / [e]dit / [f]eedback / [r]eject? ").strip().lower()
            if action in ("a", "e", "f", "r"):
                break
        if action == "r":
            print(f"Rejected at the {gate} gate. The run is checkpointed - rerun the same command to resume.", file=sys.stderr)
            raise SystemExit(1)
        if action == "f":
            fb = input("Feedback (one line, founder language): ").strip()
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
                    print(f"Invalid JSON ({e}) - reopening the editor.", file=sys.stderr)
                except UserFacingError as e:
                    print(e.render() + "\nReopening the editor.", file=sys.stderr)
            graph.update_state(cfg, {gate: edited})
            print("Edit accepted and validated.", file=sys.stderr)
        result = graph.invoke(Command(resume={"action": "accept"}), cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("intent_model", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("bundle"))
    ap.add_argument("--review", action="store_true",
                    help="opt-in founder review gates after the SRS and architecture stages (ADR-0019)")
    ap.add_argument("--from", dest="from_stage", metavar="STAGE", default=None,
                    help="rewind the checkpoint: re-run STAGE and everything after it under the "
                         "current code, earlier stages preserved (stages: needs, srs, architecture, "
                         "items, tests, review, assemble)")
    args = ap.parse_args()

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
            "if the schema is otherwise compatible, re-stamp it: python -m vmodel_contracts.migrate "
            f"{args.intent_model} (it validates against the current schema first and refuses on real drift); "
            "otherwise re-run the interview (minutes on a local model) to regenerate the intent model.",
        ) from e
    sid = intent["session_id"]
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "specs").mkdir(exist_ok=True)

    log = UsageLog(sid, sink=jsonl_sink(outdir / "usage.jsonl"))
    bus = ProgressBus()
    bus.subscribe(stderr_printer())

    # Disk checkpoints: a crash mid-generation resumes from the last completed
    # stage on rerun (same command) - no stage is paid for twice.
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(str(outdir / "checkpoints.sqlite"), check_same_thread=False)
    partials = _make_partials(outdir)
    graph = build_graph(log, checkpointer=SqliteSaver(conn, serde=checkpoint_serde()), progress=bus, review=args.review,
                        persist=_make_persist(intent, outdir), partials=partials)
    print(f"Generating bundle for session {sid} (crash? rerun the same command - it resumes)\n", file=sys.stderr)
    cfg = {"configurable": {"thread_id": sid}}
    prior = graph.get_state(cfg).values
    resume = bool(prior)
    if resume:
        print("Checkpoint found - resuming; completed stages are NOT re-run.", file=sys.stderr)
        # backfill-persist artifacts for stages completed BEFORE incremental
        # persistence shipped: resume skips completed stages (persist never
        # fires for them), so without this the bundle dir stays empty. State
        # is authoritative - overwrite any stale files.
        written = _backfill_persist(_make_persist(intent, outdir), prior)
        if written:
            print(f"Artifacts for already-completed stages written to the bundle dir ({', '.join(written)}).",
                  file=sys.stderr)
    if args.from_stage:
        _invalidate_from(graph, cfg, args.from_stage, args.review)
        for key in _PARTIAL_DOWNSTREAM.get(args.from_stage, ()):
            partials.clear(key)
        resume = args.from_stage != "needs"
        if resume:
            kept = [s for s in STAGE_ORDER[:STAGE_ORDER.index(args.from_stage)] if prior.get(s)]
            print(f"Re-running from stage '{args.from_stage}' under the current code - "
                  f"preserved: {', '.join(kept) or '(nothing)'}; downstream state invalidated.",
                  file=sys.stderr)
    if args.review:
        result = _run_with_gates(graph, cfg, intent, outdir, resume=resume)
    else:
        result = graph.invoke(None, cfg) if resume else graph.invoke({"intent_model": intent}, cfg)

    b = result["bundle"]
    try:
        needs = StakeholderNeedsSet.model_validate(normalize_enums(coerce_for_model(b["_needs"], StakeholderNeedsSet), StakeholderNeedsSet))
        reqs = RequirementSet.model_validate(normalize_enums(coerce_for_model(b["_requirements"], RequirementSet), RequirementSet))
        arch = ArchitectureOverview.model_validate(coerce_for_model(b["_architecture"], ArchitectureOverview))
        items = [ItemSpec.model_validate(coerce_for_model(i, ItemSpec)) for i in b["items"]]
        manifest = SpecManifest.model_validate(coerce_for_model(b["_manifest"], SpecManifest))
        punch = [PunchListItem.model_validate(coerce_for_model(p, PunchListItem)) for p in b["punch_list"]]
    except Exception as e:
        raise UserFacingError(
            "a generated artifact fails the vmodel-contracts schema at render time.",
            f"validation detail: {type(e).__name__}: {str(e)[:300]}",
            "retry the run; if it repeats, route the failing stage to a stronger model "
            "(SPEC_GENERATOR_MODEL_* env overrides) - authoring stages earn the strong model.",
        ) from e

    (outdir / "needs.md").write_text(render_needs(needs))
    srs = SRSDocument.model_validate(coerce_for_model(b["_srs"], SRSDocument))
    (outdir / "srs.md").write_text(render_srs(srs))
    (outdir / "architecture.md").write_text(render_architecture(arch))
    (outdir / "architecture.json").write_text(arch.model_dump_json(indent=2))
    for i in items:
        (outdir / "specs" / f"{i.spec_id}.md").write_text(render_spec_file(i))
    (outdir / "AGENTS.md").write_text(render_agents_md(arch, items, manifest, punch))
    (outdir / "traceability.json").write_text(json.dumps(b["traceability"], indent=2))
    (outdir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (outdir / "tests.json").write_text(json.dumps(b["_tests"], indent=2))

    if result.get("violations"):
        print("WARNING - traceability violations (in punch list):")
        for v in result["violations"]:
            print(f"  - {v}")
    print(f"Bundle written to {outdir}/  |  generation cost: ${log.total_usd():.4f}")


if __name__ == "__main__":
    run_cli(main)

