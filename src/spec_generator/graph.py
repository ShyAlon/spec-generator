"""Generation pipeline as a LangGraph state machine.

LangGraph earns its place here (ADR-0003): staged drafts with independent
review passes (3 drafts + 2 review rounds is table stakes), resumable
background execution (Vercel's 300s cap means this runs as a job, not a
request), and a deterministic assembly tail that no LLM touches.
"""
from __future__ import annotations

import json
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver

from .progress import ProgressBus, track
from langgraph.graph import END, StateGraph

from vmodel_contracts import (
    AcceptanceTest,
    ArchitectureOverview,
    IntentModel,
    ItemSpec,
    PunchListItem,
    RequirementSet,
    SpecBundle,
    SpecManifest,
    SRSDocument,
    StakeholderNeedsSet,
    text_qa,
)

from . import model_config
from .assembly import build_traceability, order_milestones, validate_traceability
from .prompts import METHODOLOGY, STAGE_PROMPTS
from .telemetry import UsageLog

REVIEW_PASSES = 2  # independent review rounds; fixes applied by re-running the flagged stage's prompt with the fix list


class GenState(TypedDict, total=False):
    intent_model: dict
    needs: dict
    srs: dict
    requirements: dict
    guard_flags: list[str]
    architecture: dict
    items: list[dict]
    tests: list[dict]
    review_round: int
    review_fixes: list[str]
    bundle: dict
    violations: list[str]


def _invoke(stage: str, payload: str, log: UsageLog, progress: ProgressBus | None = None) -> str:
    """One model call per stage, STREAMED so the console can show each item as
    its JSON closes (same prompt, same parameters, same completion - progress
    is observability only, never a change to what is generated)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from .stream_items import ItemReporter

    model = model_config.build_chat_model(stage)
    reporter = ItemReporter(progress, stage)
    chunks: list[str] = []
    usage: dict = {}
    for chunk in model_config.stream_checked(
            model, [SystemMessage(content=METHODOLOGY), HumanMessage(content=payload)], stage):
        text = chunk.content if isinstance(chunk.content, str) else "".join(
            b.get("text", "") for b in chunk.content if isinstance(b, dict))
        chunks.append(text)
        reporter.feed("".join(chunks))
        usage = getattr(chunk, "usage_metadata", None) or usage
    cfg = model_config.for_stage(stage)
    log.record(stage, cfg.provider, cfg.model, usage)
    return "".join(chunks)


def _json(text: str, stage: str = "unknown"):
    try:
        return json.loads(text.strip().strip("`").removeprefix("json").strip())
    except json.JSONDecodeError as e:
        from .errors import translate_json_error

        raise translate_json_error(e, stage) from e


def build_graph(log: UsageLog, checkpointer=None, progress: ProgressBus | None = None):

    def _wrap(stage: str, label: str, detail: str | None = None):
        def deco(fn):
            def inner(state: GenState) -> dict:
                with track(progress, stage, label, detail):
                    return fn(state)
            return inner
        return deco

    def node_needs(state: GenState) -> dict:
        intent = IntentModel.model_validate(state["intent_model"])
        raw = _invoke("needs", STAGE_PROMPTS["needs"] + "\n\nINTENT:\n" + intent.model_dump_json(), log, progress)
        data = _json(raw, "needs")
        data.setdefault("session_id", intent.session_id)
        return {"needs": data}

    def node_srs(state: GenState) -> dict:
        """The SRS stage: authors the standalone requirements document body.
        Structure (definitions, constraints, assumptions, open questions) is
        carried DETERMINISTICALLY from the intent model in assembly - the LLM
        authors purpose, scope_summary, and the sectioned requirements only.
        Mechanical guards (text_qa) flag solution language and invented detail."""
        raw = _invoke("srs",
                      STAGE_PROMPTS["srs"] + "\n\nNEEDS:\n" + json.dumps(state["needs"])
                      + "\n\nINTENT:\n" + json.dumps(state["intent_model"]), log, progress)
        data = _json(raw, "srs")
        data.setdefault("session_id", state["intent_model"]["session_id"])
        data.setdefault("intent_model_ref", state["intent_model"]["session_id"])
        # mechanical guards: solution-agnostic + grounded (ADR-0015 doctrine)
        founder_text = json.dumps({k: state["intent_model"].get(k) for k in
                                   ("problem_statement", "core_value_proposition", "in_scope")})
        needs_text = json.dumps(state["needs"])
        flags = []
        for section in ("functional", "non_functional"):
            for req in data.get(section) or []:
                for field in ("title", "statement"):
                    if text_qa.is_solution_language(str(req.get(field, ""))):
                        flags.append(f"{req.get('requirement_id', '?')}.{field}: solution language")
                    if not text_qa.grounded_in(str(req.get(field, "")), [founder_text, needs_text]):
                        flags.append(f"{req.get('requirement_id', '?')}.{field}: ungrounded (no founder vocabulary)")
                kept = []
                for ac in req.get("acceptance") or []:
                    if text_qa.is_solution_language(str(ac.get("statement", ""))):
                        flags.append(f"{ac.get('criterion_id', '?')}: solution-language criterion dropped")
                        continue
                    kept.append(ac)
                if kept or not req.get("acceptance"):
                    req["acceptance"] = kept
        return {"srs": data, "guard_flags": state.get("guard_flags", []) + flags}

    def node_architecture(state: GenState) -> dict:
        raw = _invoke("architecture",
                      STAGE_PROMPTS["architecture"] + "\n\nSRS:\n" + json.dumps(state["srs"]), log, progress)
        data = _json(raw, "architecture")
        data.setdefault("session_id", state["intent_model"]["session_id"])
        return {"architecture": data}

    def node_items(state: GenState) -> dict:
        # Item-level, parallel-friendly: short passes per slice beat one mega-pass (cost + regen granularity).
        raw = _invoke("items",
                      STAGE_PROMPTS["items"] + "\n\nSRS:\n" + json.dumps(state["srs"])
                      + "\n\nARCHITECTURE:\n" + json.dumps(state["architecture"]), log, progress)
        return {"items": _json(raw, "items")}

    def node_tests(state: GenState) -> dict:
        """Two V pairings in one pass: acceptance tests pair with SRS criteria,
        unit/module tests pair with item specs."""
        raw = _invoke("tests",
                      STAGE_PROMPTS["tests"] + "\n\nSRS:\n" + json.dumps(state["srs"])
                      + "\n\nITEMS:\n" + json.dumps(state["items"]), log, progress)
        data = _json(raw, "tests")
        if isinstance(data, dict):
            return {"tests": (data.get("acceptance_tests") or []) + (data.get("unit_module_tests") or [])}
        return {"tests": data}  # tolerate a flat array from weaker models

    def node_review(state: GenState) -> dict:
        n = state.get("review_round", 0) + 1
        raw = _invoke("review", STAGE_PROMPTS["review"].format(n=n)
                      + "\n\nSRS:\n" + json.dumps(state["srs"])
                      + "\n\nITEMS:\n" + json.dumps(state["items"])
                      + "\n\nTESTS:\n" + json.dumps(state["tests"]), log, progress)
        return {"review_round": n, "review_fixes": _json(raw, "review")}

    def route_review(state: GenState) -> str:
        if state.get("review_round", 0) >= REVIEW_PASSES:
            return "assemble"
        if not state.get("review_fixes"):
            return "assemble"  # clean review: stop early, save the pass
        return "apply_fixes"

    def node_apply_fixes(state: GenState) -> dict:
        raw = _invoke("srs",
                      "Apply these review fixes to the SRS and return the corrected SRS body as STRICT JSON:\n"
                      + json.dumps(state["review_fixes"]) + "\n\nSRS:\n" + json.dumps(state["srs"]), log, progress)
        data = _json(raw, "srs")
        data.setdefault("session_id", state["intent_model"]["session_id"])
        data.setdefault("intent_model_ref", state["intent_model"]["session_id"])
        return {"srs": data}

    def node_assemble(state: GenState) -> dict:
        """Deterministic tail. Validate hard; ship nothing with orphan traces."""
        sid = state["intent_model"]["session_id"]
        intent = IntentModel.model_validate(state["intent_model"])
        needs = StakeholderNeedsSet.model_validate(state["needs"])
        srs_data = dict(state["srs"])
        # structure is carried from the intent model deterministically - never
        # trust the LLM to re-copy glossary/assumptions/open questions.
        srs_data["definitions"] = intent.glossary
        srs_data["constraints"] = intent.constraints
        srs_data["assumptions"] = [a.model_dump(mode="json") for a in intent.assumption_items]
        srs_data["open_questions"] = intent.open_questions
        srs = SRSDocument.model_validate(srs_data)
        # the flat view feeds the existing traceability math unchanged
        reqs = RequirementSet(session_id=sid, requirements=srs.requirements())
        arch = ArchitectureOverview.model_validate(state["architecture"])
        items = [ItemSpec.model_validate(i) for i in state["items"]]
        tests = [AcceptanceTest.model_validate(t) for t in state["tests"]]
        violations = validate_traceability(needs, reqs, items, tests) + state.get("guard_flags", [])
        trace = build_traceability(reqs, items, tests)
        milestones = order_milestones(items, tests)
        manifest = SpecManifest(
            bundle_version="0.2.0",
            files={
                "needs": "needs.md", "srs": "srs.md",
                "architecture": "architecture.md", "agents": "AGENTS.md",
                "traceability": "traceability.json", "manifest": "manifest.json",
                **{i.spec_id: f"specs/{i.spec_id}.md" for i in items},
            },
            milestones=milestones,
        )
        punch = [PunchListItem(item_id=f"PL-{n+1}", description=v, blocks=[], severity="blocking")
                 for n, v in enumerate(violations)]
        bundle = SpecBundle(
            session_id=sid, intent_model_ref=sid, items=items,
            agents_md="",  # rendered at delivery from manifest+arch (render.py)
            punch_list=punch, traceability=trace,
        )
        out = bundle.model_dump()
        out["_manifest"] = manifest.model_dump()
        out["_needs"] = needs.model_dump()
        out["_srs"] = srs.model_dump()
        out["_requirements"] = reqs.model_dump()
        out["_architecture"] = arch.model_dump()
        out["_tests"] = [t.model_dump() for t in tests]
        return {"bundle": out, "violations": violations}

    g = StateGraph(GenState)
    g.add_node("needs", _wrap("needs", "Formalizing stakeholder needs", "stage 1/6")(node_needs))
    g.add_node("srs", _wrap("srs", "Authoring the SRS", "stage 2/6")(node_srs))
    g.add_node("architecture", _wrap("architecture", "Drafting architecture overview", "stage 3/6")(node_architecture))
    g.add_node("items", _wrap("items", "Writing module specs", "stage 4/6")(node_items))
    g.add_node("tests", _wrap("tests", "Deriving test specifications", "stage 5/6")(node_tests))
    g.add_node("review", _wrap("review", "Cross-checking consistency", "stage 6/6")(node_review))
    g.add_node("apply_fixes", _wrap("apply_fixes", "Applying review fixes")(node_apply_fixes))
    g.add_node("assemble", _wrap("assemble", "Assembling + validating bundle", "deterministic")(node_assemble))
    g.set_entry_point("needs")
    g.add_edge("needs", "srs")
    g.add_edge("srs", "architecture")
    g.add_edge("architecture", "items")
    g.add_edge("items", "tests")
    g.add_edge("tests", "review")
    g.add_conditional_edges("review", route_review, {"apply_fixes": "apply_fixes", "assemble": "assemble"})
    g.add_edge("apply_fixes", "items")
    g.add_edge("assemble", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())
