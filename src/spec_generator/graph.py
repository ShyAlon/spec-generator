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
    StakeholderNeedsSet,
)

from . import model_config
from .assembly import build_traceability, order_milestones, validate_traceability
from .prompts import METHODOLOGY, STAGE_PROMPTS
from .telemetry import UsageLog

REVIEW_PASSES = 2  # independent review rounds; fixes applied by re-running the flagged stage's prompt with the fix list


class GenState(TypedDict, total=False):
    intent_model: dict
    needs: dict
    requirements: dict
    architecture: dict
    items: list[dict]
    tests: list[dict]
    review_round: int
    review_fixes: list[str]
    bundle: dict
    violations: list[str]


def _invoke(stage: str, payload: str, log: UsageLog) -> str:
    model = model_config.build_chat_model(stage)
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = model_config.invoke_checked(model, [SystemMessage(content=METHODOLOGY), HumanMessage(content=payload)], stage)
    log.record(stage, model_config.for_stage(stage).provider, model_config.for_stage(stage).model,
               (getattr(resp, "usage_metadata", None) or {}))
    return resp.content


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
        raw = _invoke("needs", STAGE_PROMPTS["needs"] + "\n\nINTENT:\n" + intent.model_dump_json(), log)
        data = _json(raw, "needs")
        data.setdefault("session_id", intent.session_id)
        return {"needs": data}

    def node_requirements(state: GenState) -> dict:
        raw = _invoke("requirements",
                      STAGE_PROMPTS["requirements"] + "\n\nNEEDS:\n" + json.dumps(state["needs"])
                      + "\n\nINTENT:\n" + json.dumps(state["intent_model"]), log)
        data = _json(raw, "requirements")
        data.setdefault("session_id", state["intent_model"]["session_id"])
        return {"requirements": data}

    def node_architecture(state: GenState) -> dict:
        raw = _invoke("architecture",
                      STAGE_PROMPTS["architecture"] + "\n\nREQUIREMENTS:\n" + json.dumps(state["requirements"]), log)
        data = _json(raw, "architecture")
        data.setdefault("session_id", state["intent_model"]["session_id"])
        return {"architecture": data}

    def node_items(state: GenState) -> dict:
        # Item-level, parallel-friendly: short passes per slice beat one mega-pass (cost + regen granularity).
        raw = _invoke("items",
                      STAGE_PROMPTS["items"] + "\n\nREQUIREMENTS:\n" + json.dumps(state["requirements"])
                      + "\n\nARCHITECTURE:\n" + json.dumps(state["architecture"]), log)
        return {"items": _json(raw, "items")}

    def node_tests(state: GenState) -> dict:
        raw = _invoke("tests",
                      STAGE_PROMPTS["tests"] + "\n\nREQUIREMENTS:\n" + json.dumps(state["requirements"])
                      + "\n\nITEMS:\n" + json.dumps(state["items"]), log)
        return {"tests": _json(raw, "tests")}

    def node_review(state: GenState) -> dict:
        n = state.get("review_round", 0) + 1
        raw = _invoke("review", STAGE_PROMPTS["review"].format(n=n)
                      + "\n\nREQUIREMENTS:\n" + json.dumps(state["requirements"])
                      + "\n\nITEMS:\n" + json.dumps(state["items"])
                      + "\n\nTESTS:\n" + json.dumps(state["tests"]), log)
        return {"review_round": n, "review_fixes": _json(raw, "review")}

    def route_review(state: GenState) -> str:
        if state.get("review_round", 0) >= REVIEW_PASSES:
            return "assemble"
        if not state.get("review_fixes"):
            return "assemble"  # clean review: stop early, save the pass
        return "apply_fixes"

    def node_apply_fixes(state: GenState) -> dict:
        raw = _invoke("requirements",
                      "Apply these review fixes to the requirements set and return the corrected set as STRICT JSON:\n"
                      + json.dumps(state["review_fixes"]) + "\n\nREQUIREMENTS:\n" + json.dumps(state["requirements"]), log)
        data = _json(raw, "requirements")
        data.setdefault("session_id", state["intent_model"]["session_id"])
        return {"requirements": data}

    def node_assemble(state: GenState) -> dict:
        """Deterministic tail. Validate hard; ship nothing with orphan traces."""
        sid = state["intent_model"]["session_id"]
        needs = StakeholderNeedsSet.model_validate(state["needs"])
        reqs = RequirementSet.model_validate(state["requirements"])
        arch = ArchitectureOverview.model_validate(state["architecture"])
        items = [ItemSpec.model_validate(i) for i in state["items"]]
        tests = [AcceptanceTest.model_validate(t) for t in state["tests"]]
        violations = validate_traceability(needs, reqs, items, tests)
        trace = build_traceability(reqs, items, tests)
        milestones = order_milestones(items, tests)
        manifest = SpecManifest(
            bundle_version="0.1.0",
            files={
                "needs": "needs.md", "requirements": "requirements.md",
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
        out["_requirements"] = reqs.model_dump()
        out["_architecture"] = arch.model_dump()
        out["_tests"] = [t.model_dump() for t in tests]
        return {"bundle": out, "violations": violations}

    g = StateGraph(GenState)
    g.add_node("needs", _wrap("needs", "Extracting stakeholder needs", "stage 1/6")(node_needs))
    g.add_node("requirements", _wrap("requirements", "Deriving requirements", "stage 2/6")(node_requirements))
    g.add_node("architecture", _wrap("architecture", "Drafting architecture overview", "stage 3/6")(node_architecture))
    g.add_node("items", _wrap("items", "Writing module specs", "stage 4/6")(node_items))
    g.add_node("tests", _wrap("tests", "Deriving test specifications", "stage 5/6")(node_tests))
    g.add_node("review", _wrap("review", "Cross-checking consistency", "stage 6/6")(node_review))
    g.add_node("apply_fixes", _wrap("apply_fixes", "Applying review fixes")(node_apply_fixes))
    g.add_node("assemble", _wrap("assemble", "Assembling + validating bundle", "deterministic")(node_assemble))
    g.set_entry_point("needs")
    g.add_edge("needs", "requirements")
    g.add_edge("requirements", "architecture")
    g.add_edge("architecture", "items")
    g.add_edge("items", "tests")
    g.add_edge("tests", "review")
    g.add_conditional_edges("review", route_review, {"apply_fixes": "apply_fixes", "assemble": "assemble"})
    g.add_edge("apply_fixes", "items")
    g.add_edge("assemble", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())
