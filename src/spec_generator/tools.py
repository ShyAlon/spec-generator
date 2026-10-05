"""Typed structural tools (ADR-0020): the model PROPOSES structural changes,
code DISPOSES. Each handler is pipeline code - validate, normalize, assign
ids - so malformed structure is impossible by construction: there is no path
from model output to an id, an ordering, or a link except through a handler.

Rollout: OPT-IN (SPEC_GENERATOR_TOOLS=1) until scripts/probe_tools.py shows
qwen2.5-coder:14b emits well-formed, schema-conformant tool calls reliably.
Fallback is the plain 1:1 skeleton - no splits, no additions, always valid.

SRS stage wires split_requirement + link_requirement_to_need; tests stage
wires add_test. Architecture stays bulk propose/dispose (its schema call IS
the proposal; code assigns ADR-ids) - add_component is defined here for the
probe and wired later if the probe reads well.
"""
from __future__ import annotations

import json
import os

from .skeleton import ModelStructureError


def tools_enabled() -> bool:
    """Opt-in until the probe reads well (ADR-0020 rollout rule)."""
    return os.environ.get("SPEC_GENERATOR_TOOLS", "0") == "1"


def _fn(name, description, properties, required):
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object", "properties": properties,
                                        "required": required}}}


_PROSE = {"type": "object",
          "properties": {"title": {"type": "string"},
                         "statement": {"type": "string"},
                         "priority": {"type": "string", "enum": ["must", "should", "could"]}},
          "required": ["title", "statement", "priority"]}

SPLIT_REQUIREMENT = _fn(
    "split_requirement",
    "Split ONE requirement slot into 2+ finer requirements when a need genuinely "
    "decomposes (structure is analysis). Code retires the slot and assigns new "
    "R-ids; every part inherits the slot's kind and trace links. You will write "
    "the full prose (including acceptance criteria) for the new ids afterwards - "
    "the parts here are your own outline.",
    {"requirement_id": {"type": "string", "description": "an existing slot id, e.g. R-3"},
     "parts": {"type": "array", "items": _PROSE,
               "description": "2 or more parts, each an outline: title, statement, priority"}},
    ["requirement_id", "parts"])

LINK_REQUIREMENT_TO_NEED = _fn(
    "link_requirement_to_need",
    "Trace one requirement slot to an ADDITIONAL stakeholder need (a requirement "
    "that realizes several needs). Code validates both ids and records the link.",
    {"requirement_id": {"type": "string"}, "need_id": {"type": "string"}},
    ["requirement_id", "need_id"])

ADD_TEST = _fn(
    "add_test",
    "Propose ONE additional test the skeleton does not cover (e.g. a module or "
    "system test spanning several specs). Code validates the target, assigns the "
    "T-id, and adopts your description/runner/definition as-is.",
    {"level": {"type": "string", "enum": ["unit", "module", "system", "acceptance"]},
     "criterion_id": {"type": "string", "description": "required for level=acceptance: an existing AC-id"},
     "spec_id": {"type": "string", "description": "required for unit/module/system: an existing SPEC-id"},
     "description": {"type": "string"},
     "runner": {"type": "string", "description": "'http' | 'ui' | 'cli' | 'static' | 'semantic'"},
     "components": {"type": "array", "items": {"type": "string"},
                    "description": "architecture component names this test exercises, exactly as listed"},
     "definition": {"type": "object", "description": "runner payload: URL+expected status, command, ... (ui tests carry no selectors)"}},
    ["level", "description", "runner", "components", "definition"])

ADD_COMPONENT = _fn(
    "add_component",
    "Propose ONE additional architecture component (name, responsibility, "
    "requirement ids it serves). Code validates uniqueness and references. "
    "Defined for the probe; wired into the architecture stage after the probe.",
    {"name": {"type": "string"}, "responsibility": {"type": "string"},
     "requirement_ids": {"type": "array", "items": {"type": "string"}}},
    ["name", "responsibility", "requirement_ids"])


class SrsToolbox:
    """split_requirement + link_requirement_to_need over a mutable SRS skeleton."""

    def __init__(self, slots: list[dict], needs_ids):
        self.slots = slots
        self.needs_ids = set(needs_ids)
        self.applied: list[str] = []

    tools = [SPLIT_REQUIREMENT, LINK_REQUIREMENT_TO_NEED]

    def call(self, name: str, args: dict):
        if name == "split_requirement":
            return self.split_requirement(args.get("requirement_id"), args.get("parts"))
        if name == "link_requirement_to_need":
            return self.link_requirement_to_need(args.get("requirement_id"), args.get("need_id"))
        raise ModelStructureError(f"unknown tool {name!r}")

    def _slot(self, requirement_id):
        slot = next((s for s in self.slots if s["requirement_id"] == requirement_id), None)
        if slot is None:
            raise ModelStructureError(f"{requirement_id}: no such requirement slot")
        return slot

    def split_requirement(self, requirement_id, parts):
        slot = self._slot(requirement_id)
        if not isinstance(parts, list) or len(parts) < 2:
            raise ModelStructureError("split_requirement needs at least 2 parts")
        for i, p in enumerate(parts, 1):
            if not isinstance(p, dict) or not (p.get("title") and p.get("statement")):
                raise ModelStructureError(f"part {i}: needs a non-empty title and statement")
            if p.get("priority", "must") not in ("must", "should", "could"):
                raise ModelStructureError(f"part {i}: priority must be must | should | could")
        top = max(int(s["requirement_id"].split("-", 1)[1]) for s in self.slots)
        new_ids = [f"R-{top + i}" for i in range(1, len(parts) + 1)]
        self.slots.remove(slot)
        for nid, p in zip(new_ids, parts):
            self.slots.append({"requirement_id": nid, "kind": slot["kind"],
                               "need_ids": list(slot["need_ids"]),
                               "intent_fields": list(slot["intent_fields"]),
                               "seed": {"title": p["title"], "statement": p["statement"],
                                        "priority": p.get("priority", "must")}})
        self.applied.append(f"split {requirement_id} -> {', '.join(new_ids)}")
        return {"retired": requirement_id, "new_ids": new_ids,
                "note": "write the full prose (with acceptance criteria) for the new ids in the fill step; the retired id is gone"}

    def link_requirement_to_need(self, requirement_id, need_id):
        slot = self._slot(requirement_id)
        if need_id not in self.needs_ids:
            raise ModelStructureError(f"{need_id}: no such stakeholder need")
        if need_id not in slot["need_ids"]:
            slot["need_ids"].append(need_id)
            self.applied.append(f"linked {requirement_id} -> {need_id}")
        return {"requirement_id": requirement_id, "need_ids": slot["need_ids"]}


class VerificationToolbox:
    """add_test over a mutable test skeleton."""

    def __init__(self, slots: list[dict], criterion_ids, spec_ids):
        self.slots = slots
        self.criterion_ids = set(criterion_ids)
        self.spec_ids = set(spec_ids)
        self.applied: list[str] = []

    tools = [ADD_TEST]

    def call(self, name: str, args: dict):
        if name == "add_test":
            return self.add_test(args.get("level"), args.get("description"), args.get("runner"),
                                 args.get("definition"), args.get("criterion_id"), args.get("spec_id"),
                                 args.get("components"))
        raise ModelStructureError(f"unknown tool {name!r}")

    def add_test(self, level, description, runner, definition, criterion_id=None, spec_id=None, components=None):
        if level not in ("unit", "module", "system", "acceptance"):
            raise ModelStructureError("level must be unit | module | system | acceptance")
        if level == "acceptance":
            if criterion_id not in self.criterion_ids:
                raise ModelStructureError(f"acceptance tests need an existing criterion_id (got {criterion_id!r})")
            spec_id = None
        else:
            if spec_id not in self.spec_ids:
                raise ModelStructureError(f"{level} tests need an existing spec_id (got {spec_id!r})")
            criterion_id = None
        if not description or not runner or not isinstance(definition, dict):
            raise ModelStructureError("add_test needs description, runner, and a definition object")
        top = max(int(s["test_id"].split("-", 1)[1]) for s in self.slots)
        tid = f"T-{top + 1}"
        self.slots.append({"test_id": tid, "level": level, "criterion_id": criterion_id,
                           "spec_id": spec_id,
                           "seed": {"description": description, "runner": runner,
                                    "components": components, "definition": definition}})
        self.applied.append(f"added {tid} ({level})")
        return {"test_id": tid, "note": "adopted as-is; no fill needed for this slot"}


def run_tool_loop(stage: str, toolbox, human_payload: str, log, progress=None,
                  max_rounds: int = 4):
    """The propose/dispose loop: the model calls tools; handlers validate and
    mutate the skeleton. Returns (applied_lines, degraded). degraded=True
    means the model could not hold up its side - the caller proceeds with the
    skeleton as it stands (every applied proposal is already valid), and the
    plain prose-fill call still runs. Never raises for model-side failure."""
    import time as _t

    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    from . import model_config
    from .progress import ProgressEvent
    from .prompts import METHODOLOGY

    def emit(line):
        if progress is not None:
            progress.emit(ProgressEvent(stage, "", "item", _t.time(), None, line))

    model = model_config.build_chat_model(stage, structured=False).bind_tools(toolbox.tools)
    messages = [SystemMessage(content=METHODOLOGY), HumanMessage(content=human_payload)]
    consecutive_errors = 0
    emitted = 0
    for _ in range(max_rounds):
        try:
            ai = model_config.invoke_checked(model, messages, stage, item="structure-proposal")
        except Exception as e:
            name = type(e).__name__
            if name in ("APIConnectionError", "ConnectError", "ConnectTimeout") or "connection" in str(e).lower():
                raise  # infra, not tool-calling: the stage's normal error path owns this
            emit(f"tool calling failed ({name}) - continuing with the plain skeleton")
            return toolbox.applied, True
        cfg = model_config.for_stage(stage)
        log.record(stage, cfg.provider, cfg.model, getattr(ai, "usage_metadata", None) or {})
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            return toolbox.applied, False  # clean finish: nothing more to propose
        messages.append(ai)
        for c in calls:
            try:
                content = json.dumps(toolbox.call(c["name"], c.get("args") or {}))
                consecutive_errors = 0
            except ModelStructureError as e:
                consecutive_errors += 1
                content = f"ERROR: {e} - correct the arguments or stop proposing."
            messages.append(ToolMessage(content=content, tool_call_id=c.get("id") or c["name"]))
        for line in toolbox.applied[emitted:]:
            emit("structure: " + line)
        emitted = len(toolbox.applied)
        if consecutive_errors >= 3:
            emit("model kept proposing invalid structure - continuing with what applied cleanly")
            return toolbox.applied, True
    return toolbox.applied, True
