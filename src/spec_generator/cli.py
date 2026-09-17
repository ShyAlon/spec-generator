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
import hashlib
import json
import os
import shlex
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
             "tests_fills": "tests_fills.partial.json",
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

    return _Partials()


# partial files invalidated by --from <stage> (that stage and everything after)
_PARTIAL_DOWNSTREAM = {"srs": ("srs_fills", "architecture", "items", "tests_fills", "review"),
                      "architecture": ("architecture", "items", "tests_fills", "review"),
                      "items": ("items", "tests_fills", "review"),
                      "tests": ("tests_fills", "review"),
                      "review": ("review",)}


def _make_persist(intent: dict, outdir: Path, greenfield: bool = False,
                  conops_present: bool = True, architecture_lifecycle=None):
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
            arch = ArchitectureOverview.model_validate(coerce_for_model(body, ArchitectureOverview))
            lifecycle_event = architecture_lifecycle.finalize(body) if architecture_lifecycle else None
            _atomic_write(outdir / "architecture.json", json.dumps(body, indent=2))
            _atomic_write(outdir / "architecture.md", render_architecture(arch, greenfield=greenfield, conops_present=conops_present))
            if architecture_lifecycle:
                architecture_lifecycle.commit_finalize(lifecycle_event)
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
    current = graph.get_state(cfg).values
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


def _run_with_gates(graph, cfg: dict, initial_state: dict, outdir, resume: bool = False) -> dict:
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
                  "do NOT rerun the command.\n", file=sys.stderr)
            result = graph.invoke(initial_state, cfg)
            continue
        if advisories:
            md += "\n\n## Review advisories\n" + "\n".join(
                f"- {a['left_id']} {a['left_slug']} and {a['right_id']} {a['right_slug']}: {a['reason']}. Nothing was merged; decide at this gate."
                for a in advisories)
        (outdir / f"gate_{gate}.md").write_text(md)
        (outdir / f"gate_{gate}.json").write_text(json.dumps(artifact, indent=2))
        (outdir / f"gate_{gate}.advisories.json").write_text(json.dumps({"gate":gate,"advisories":advisories},indent=2))
        print(f"\n===== REVIEW GATE: the {title} =====\n", file=sys.stderr)
        print(md, file=sys.stderr)
        if advisories:
            print(f"(review advisories: {outdir}/gate_{gate}.advisories.json)", file=sys.stderr)
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


def _resume_after_bounded_stop(graph, cfg: dict, prior: dict) -> str | None:
    """Re-open a checkpoint intentionally ended by --to STAGE.

    A stop-after graph records a complete state with no pending task. Rebuilding
    the graph without --to cannot manufacture the missing edge from SQLite, so
    explicitly replay the last completed node against the new full topology.
    """
    if prior.get("bundle") or graph.get_state(cfg).next:
        return None
    for stage, downstream in (("architecture", "items"), ("srs", "architecture"), ("needs", "srs")):
        if prior.get(stage) and not prior.get(downstream):
            graph.update_state(cfg, {}, as_node=stage)
            return stage
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



_SEMANTIC_EQUIVALENCE_VERSION = "1"


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
        for name in ("items.json", "tests.json", "specs", "AGENTS.md", "traceability.json", "manifest.json"):
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
    return audit

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("intent_model", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="artifact directory (default: <workspace>/artifacts/artifacts-YYYY-MM-DD_HH-MM-SS, "
                         "from the session id; VMODEL_ARTIFACTS_DIR overrides the root)")
    ap.add_argument("--review", action="store_true",
                    help="opt-in human review gates after the SRS and architecture stages")
    ap.add_argument("--expert-opinion", action="store_true",
                    help="simulate gate decisions with an LLM for test runs; records pending human review")
    ap.add_argument("--from", dest="from_stage", metavar="STAGE", default=None,
                    help="rewind the checkpoint: re-run STAGE and everything after it under the "
                         "current code, earlier stages preserved (stages: needs, srs, architecture, "
                         "items, tests, review, assemble)")
    ap.add_argument("--to", dest="to_stage", choices=("needs", "srs", "architecture", "items", "tests"),
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
    ap.add_argument("--preflight", action="store_true",
                    help="validate inputs and checkpoint compatibility without building a model or spending a call")
    args = ap.parse_args()
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
    if args.preflight:
        import sqlite3
        cp=outdir / "checkpoints.sqlite"
        if cp.exists():
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn=sqlite3.connect(str(cp),check_same_thread=False)
            probe=build_graph(UsageLog(sid),checkpointer=SqliteSaver(conn,serde=checkpoint_serde()),review=args.review)
            prior=probe.get_state({"configurable":{"thread_id":sid}}).values
            if prior:
                if prior.get("conops") != conops:
                    raise UserFacingError("preflight: checkpoint Concept of Operations is incompatible.",
                        "the existing output was started without this CONOPS or with a different one.",
                        "choose a fresh -o directory, or pass the original CONOPS used by this checkpoint.")
                if prior.get("system_baseline") not in (None,baseline):
                    raise UserFacingError("preflight: checkpoint system baseline is incompatible.",
                        "the existing output was generated under a different architecture anchor.",
                        "choose a fresh -o directory, or explicitly rerun from architecture under the new baseline.")
        print(f"Preflight OK: session={sid}; seed=intent_model,system_baseline,conops; "
              f"baseline={baseline['baseline_id']}; conops={conops is not None}; "
              f"checkpoint={'compatible' if cp.exists() else 'fresh'}")
        return
    if args.reconcile_artifacts:
        audit = _reconcile_artifacts(intent, baseline, outdir, args.reconcile_out,
                                     conops_present=conops is not None)
        print("Reconciled artifacts with ZERO LLM calls.", file=sys.stderr)
        print(f"Semantic changes: {', '.join(audit['semantic_changed']) or '(none)'}", file=sys.stderr)
        print(f"Downstream: {audit['downstream']} ({audit['destination']})", file=sys.stderr)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    from vmodel_contracts.runtime import install_plain_log
    run_log = outdir / "run.log"
    sys.stderr = install_plain_log(run_log, sys.stderr)
    # Per-call telemetry is part of every normal run, not an experiment-only
    # opt-in. Preserve an explicit external sink, otherwise keep it beside the bundle.
    os.environ.setdefault("VMODEL_LLM_LOG", str(outdir / "llm-calls.jsonl"))
    print(f"Run log: {run_log.resolve()} (plain text)", file=sys.stderr)
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
    from .artifact_lifecycle import ArchitectureLifecycle
    architecture_lifecycle = ArchitectureLifecycle(outdir, sid, baseline)
    partials = _make_partials(outdir, architecture_lifecycle)
    graph = build_graph(log, checkpointer=SqliteSaver(conn, serde=checkpoint_serde()), progress=bus, review=args.review, expert_opinion=args.expert_opinion,
                        persist=_make_persist(intent, outdir, greenfield=baseline["mode"] == "greenfield",
                                              conops_present=conops is not None,
                                              architecture_lifecycle=architecture_lifecycle),
                        partials=partials, stop_after=args.to_stage)
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
    if resume and prior.get("architecture"):
        architecture_lifecycle.bootstrap_complete(
            prior["architecture"], prior.get("system_baseline") or baseline)
    if resume and not args.to_stage:
        bounded_stage = _resume_after_bounded_stop(graph, cfg, prior)
        if bounded_stage:
            if bounded_stage == "architecture":
                architecture_lifecycle.resume("gate_architecture" if args.review else "items")
            print(f"Checkpoint intentionally stopped after '{bounded_stage}' - continuing through the full pipeline.", file=sys.stderr)
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
        architecture_lifecycle.reanchor(["architecture", "items", "tests", "assembly", "manifest", "traceability"])
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
        # Write AS the last baseline-independent node. A bare update on a
        # failed/interrupted review graph can fork from the failed architecture
        # task and leave that task's state unchanged (the live 2026-09-15
        # accepted-SRS resume did exactly that). Re-anchoring at gate_srs
        # preserves needs + accepted SRS and schedules architecture next.
        graph.update_state(cfg, {"system_baseline": baseline},
                           as_node=_predecessor("architecture", args.review))
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
        result = _run_with_gates(graph, cfg,
                                 {"intent_model": intent, "system_baseline": baseline, "conops": conops},
                                 outdir, resume=resume)
        if args.expert_opinion:
            opinions=(graph.get_state(cfg).values.get("expert_opinions") or [])
            _atomic_write(outdir / "expert-opinions.json",json.dumps({"mode":"simulated_expert_opinion","founder_approved":False,"human_review_required":True,"decisions":opinions},indent=2))
            with (outdir / "audit.jsonl").open("a") as fh:
                for opinion in opinions: fh.write(json.dumps({"event":"gate.simulated_expert_opinion",**opinion,"founder_approved":False})+"\n")
    else:
        result = graph.invoke(None, cfg) if resume else graph.invoke(
            {"intent_model": intent, "system_baseline": baseline, "conops": conops}, cfg)

    if args.to_stage:
        print(f"Stopped after stage '{args.to_stage}' as requested. Checkpoint and stage artifacts are resumable in {outdir}.")
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



