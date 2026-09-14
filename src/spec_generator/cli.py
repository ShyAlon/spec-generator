"""Intent model in, repo-ready bundle out.

Usage: python -m spec_generator.cli out/<session>/intent_model.json

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
import shutil
import sys
from datetime import datetime
from pathlib import Path

from vmodel_contracts import (
    run_with_system_awake, AcceptanceTest, ArchitectureOverview, ItemSpec, PunchListItem,
    RequirementSet, SpecManifest, SRSDocument, StakeholderNeedsSet, SystemBaseline,
)
from vmodel_contracts.runtime import artifacts_root

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
             "architecture": "architecture.partial.json",
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
_PARTIAL_DOWNSTREAM = {"srs": ("srs_fills", "architecture", "items", "tests_fills"),
                      "architecture": ("architecture", "items", "tests_fills"),
                      "items": ("items", "tests_fills"),
                      "tests": ("tests_fills",)}


def _make_persist(intent: dict, outdir: Path, greenfield: bool = False,
                  conops_present: bool = True):
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
            _atomic_write(outdir / "architecture.md", render_architecture(arch, greenfield=greenfield, conops_present=conops_present))
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
            print(f"Rejected at the {gate} gate. The run is checkpointed - {_mc._rerun_guidance(gate)}.", file=sys.stderr)
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


def _default_outdir(intent_path: Path, session_id: str) -> Path:
    """Stable default output dir, derived from the run identity.

    Interview session ids begin YYYYMMDD-HHMMSS, so a resumed command selects
    the same directory instead of inventing a fresh timestamp. Hand-authored
    ids fall back to the intent file's mtime, which is likewise stable for the
    file being run.

    The directory lands in the artifacts SIBLING folder (<workspace>/artifacts/
    when run from a repo checkout inside the workspace - shared convention in
    vmodel_contracts.runtime.artifacts_root), keeping generated output out of
    the git trees. VMODEL_ARTIFACTS_DIR (env or the shared .env) overrides the
    root; -o overrides the whole path.
    """
    import re

    from . import model_config

    match = re.match(r"^(\d{8})-(\d{6})(?:-|$)", session_id)
    if match:
        stamp = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    else:
        stamp = datetime.fromtimestamp(intent_path.stat().st_mtime)
    return artifacts_root(model_config._env("VMODEL_ARTIFACTS_DIR")) / f"artifacts-{stamp:%Y-%m-%d_%H-%M-%S}"


def _load_system_baseline(path: Path) -> dict:
    if not path.is_file():
        raise UserFacingError(
            f"system baseline file not found: {path}.",
            "architecture generation requires an explicit existing or greenfield baseline.",
            "pass --system-baseline <file>; copy examples/system-baseline.greenfield.json for a new product.",
        )
    try:
        raw = json.loads(path.read_text())
        from vmodel_contracts import check_compatible
        check_compatible(raw)
        return SystemBaseline.model_validate(raw).model_dump(mode="json")
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
    "review": ("srs", "architecture", "items", "tests"),
}
_ARTIFACT_FILES = {"needs": "needs.json", "srs": "srs.json", "architecture": "architecture.json",
                   "items": "items.json", "tests": "tests.json"}


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
    print(f"Stage iteration: {args.only_stage}", file=sys.stderr)
    print(f"Loaded immutable source: {source}", file=sys.stderr)
    print(f"System baseline: {args.system_baseline.resolve()} ({baseline['mode']}, {len(baseline['repositories'])} repos)", file=sys.stderr)
    print(f"Loaded upstream: {', '.join(loaded) or 'intent only'}", file=sys.stderr)
    print(f"Recomputing ONLY: {args.only_stage}; downstream artifacts are copied for comparison, not regenerated.",
          file=sys.stderr)
    print(f"New output: {outdir} (source will not be modified)", file=sys.stderr)
    log = UsageLog(sid, sink=jsonl_sink(outdir / "usage.jsonl"))
    bus = ProgressBus(); bus.subscribe(stderr_printer())
    persist = _make_persist(intent, outdir, greenfield=baseline["mode"] == "greenfield")
    graph = build_graph(log, progress=bus, persist=persist, partials=_make_partials(outdir),
                        only_stage=args.only_stage)
    result = graph.invoke(state, {"configurable": {"thread_id": sid + "-" + args.only_stage}})
    if args.only_stage == "review":
        _atomic_write(outdir / "review-fixes.json", json.dumps(result.get("review_fixes") or [], indent=2))
    print(f"Stage '{args.only_stage}' written to {outdir}. Source unchanged.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("intent_model", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="artifact directory (default: <workspace>/artifacts/artifacts-YYYY-MM-DD_HH-MM-SS, "
                         "from the session id; VMODEL_ARTIFACTS_DIR overrides the root)")
    ap.add_argument("--review", action="store_true",
                    help="opt-in founder review gates after the SRS and architecture stages (ADR-0019)")
    ap.add_argument("--from", dest="from_stage", metavar="STAGE", default=None,
                    help="rewind the checkpoint: re-run STAGE and everything after it under the "
                         "current code, earlier stages preserved (stages: needs, srs, architecture, "
                         "items, tests, review, assemble)")
    ap.add_argument("--only-stage", choices=tuple(_ONLY_STAGE_INPUTS), default=None,
                    help="run exactly one stage from an existing completed artifact set; never mutates source")
    ap.add_argument("--source-artifacts", type=Path, default=None,
                    help="completed artifacts directory supplying upstream inputs for --only-stage")
    ap.add_argument("--system-baseline", type=Path, default=None,
                    help="versioned SystemBaseline JSON; existing repo manifest or explicit greenfield declaration")
    ap.add_argument("--conops", type=Path, default=None,
                    help="founder-confirmed conops.json from interview-engine; omit only for legacy sessions")
    args = ap.parse_args()
    if bool(args.only_stage) != bool(args.source_artifacts):
        ap.error("--only-stage and --source-artifacts must be used together")
    if args.only_stage and args.from_stage:
        ap.error("--only-stage cannot be combined with --from")
    if args.only_stage and args.review:
        ap.error("--only-stage cannot be combined with --review")

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
              "choice in architecture is unverified and must be challenged.", file=sys.stderr)

    if args.system_baseline is None:
        raise UserFacingError(
            "--system-baseline is required.",
            "architecture must distinguish an existing system from an explicit greenfield product.",
            "pass a checked SystemBaseline JSON; for a new product copy examples/system-baseline.greenfield.json.",
        )
    baseline = _load_system_baseline(args.system_baseline)
    print(f"System baseline: {args.system_baseline.resolve()} "
          f"(id={baseline['baseline_id']}, mode={baseline['mode']}, "
          f"{len(baseline['repositories'])} repos)", file=sys.stderr)
    sid = intent["session_id"]
    if args.only_stage:
        _run_only_stage(args, intent, sid)
        return
    outdir = args.out if args.out is not None else _default_outdir(args.intent_model, sid)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "specs").mkdir(exist_ok=True)
    _atomic_write(outdir / "system-baseline.json", json.dumps(baseline, indent=2))

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
                        persist=_make_persist(intent, outdir, greenfield=baseline["mode"] == "greenfield",
                                              conops_present=conops is not None),
                        partials=partials)
    from . import model_config as _mc

    if args.from_stage:
        os.environ["SPEC_GENERATOR_FROM"] = args.from_stage  # guardrail errors must not echo --from back
    print(f"Generating bundle for session {sid} (crash? {_mc._rerun_guidance('')})", file=sys.stderr)
    print(f"Models: {_mc.describe_models()}", file=sys.stderr)
    print(f"Bundle dir: {outdir.resolve()}", file=sys.stderr)
    print("  progress persists there per item (*.partial.json as items close; stage artifacts land "
          "as each stage completes)\n", file=sys.stderr)
    cfg = {"configurable": {"thread_id": sid}}
    prior = graph.get_state(cfg).values
    resume = bool(prior)
    if resume:
        prior_conops = prior.get("conops")
        if prior_conops != conops:
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
        print("Checkpoint found - resuming; completed stages are NOT re-run.", file=sys.stderr)
        # backfill-persist artifacts for stages completed BEFORE incremental
        # persistence shipped: resume skips completed stages (persist never
        # fires for them), so without this the bundle dir stays empty. State
        # is authoritative - overwrite any stale files.
        written = _backfill_persist(_make_persist(intent, outdir,
                                                        conops_present=conops is not None), prior)
        if written:
            print(f"Artifacts for already-completed stages written to the bundle dir ({', '.join(written)}).",
                  file=sys.stderr)
    if resume and prior.get("system_baseline") is not None and prior["system_baseline"] != baseline:
        # A completed stage is only "completed" under the baseline it was
        # built with. Resuming under a DIFFERENT anchor must never silently
        # re-label old output (2026-09-13: the poisoned overnight six-repo
        # bundle re-bundled as "greenfield" and nearly shipped). needs/srs
        # never read the baseline; architecture and everything downstream do
        # - invalidate from architecture, loudly, and re-run.
        old_b = prior["system_baseline"]
        print("WARNING: the --system-baseline for this run differs from the checkpoint's:", file=sys.stderr)
        print(f"  checkpoint: id={old_b.get('baseline_id')}, mode={old_b.get('mode')}, "
              f"{len(old_b.get('repositories') or [])} repos", file=sys.stderr)
        print(f"  this run:   id={baseline['baseline_id']}, mode={baseline['mode']}, "
              f"{len(baseline['repositories'])} repos", file=sys.stderr)
        print("  architecture and everything downstream were built under the old anchor - "
              "invalidating them (needs/srs do not depend on the baseline and are preserved).",
              file=sys.stderr)
        graph.update_state(cfg, {"system_baseline": baseline, **_CLEAR_DOWNSTREAM["architecture"]},
                           as_node=_predecessor("architecture", args.review))
        for key in _PARTIAL_DOWNSTREAM.get("architecture", ()):
            saved = partials.load(key) or {}
            if len(saved):
                print(f"  discarding {len(saved)} preserved item(s) ({key}) - they will be regenerated "
                      "under the new baseline.", file=sys.stderr)
            partials.clear(key)
        for stale in ("architecture.json", "architecture.md", "items.json", "tests.json",
                      "AGENTS.md", "manifest.json", "traceability.json", "review-fixes.json"):
            (outdir / stale).unlink(missing_ok=True)
        for stale in (outdir / "specs").glob("SPEC-*.md"):
            stale.unlink()
        print("  stale downstream artifacts removed from the bundle dir; re-running from 'architecture'.",
              file=sys.stderr)
    elif resume and prior.get("system_baseline") is None:
        # --system-baseline is authoritative for THIS invocation. Old
        # checkpoints predate the baseline feature entirely (the 2026-09-12
        # --from architecture KeyError 'system_baseline'): merge the flag
        # value into the resumed state so baseline-consuming stages see it.
        graph.update_state(cfg, {"system_baseline": baseline})
        print("System baseline merged into the resumed run (its checkpoint predates the baseline feature).",
              file=sys.stderr)
    if args.from_stage:
        _invalidate_from(graph, cfg, args.from_stage, args.review)
        for key in _PARTIAL_DOWNSTREAM.get(args.from_stage, ()):
            saved = partials.load(key) or {}
            if len(saved):
                print(f"warning: --from {args.from_stage} discards {len(saved)} preserved item(s) "
                      f"({key}) - they will be regenerated.", file=sys.stderr)
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
        result = graph.invoke(None, cfg) if resume else graph.invoke(
            {"intent_model": intent, "system_baseline": baseline, "conops": conops}, cfg)

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
    (outdir / "architecture.md").write_text(
        render_architecture(arch, greenfield=baseline["mode"] == "greenfield",
                            conops_present=conops is not None))
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
    run_cli(lambda: run_with_system_awake(main))



