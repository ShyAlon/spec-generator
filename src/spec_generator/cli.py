"""Intent model in, repo-ready bundle out.

Usage: python -m spec_generator.cli out/<session>/intent_model.json -o bundle/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vmodel_contracts import (
    AcceptanceTest, ArchitectureOverview, ItemSpec, PunchListItem,
    RequirementSet, SpecManifest, StakeholderNeedsSet,
)

from .errors import UserFacingError, run_cli
from .graph import build_graph
from .progress import ProgressBus, stderr_printer
from .repair import coerce_for_model, normalize_enums
from .render import render_agents_md, render_needs, render_requirements, render_spec_file
from .telemetry import UsageLog, jsonl_sink


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("intent_model", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("bundle"))
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
            "re-run the interview (minutes on a local model) to regenerate the intent model - "
            "or install the matching older vmodel-contracts to consume this file.",
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
    graph = build_graph(log, checkpointer=SqliteSaver(conn), progress=bus)
    print(f"Generating bundle for session {sid} (crash? rerun the same command - it resumes)\n", file=sys.stderr)
    result = graph.invoke({"intent_model": intent}, {"configurable": {"thread_id": sid}})

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
    (outdir / "requirements.md").write_text(render_requirements(reqs))
    (outdir / "architecture.md").write_text(arch.model_dump_json(indent=2))
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
