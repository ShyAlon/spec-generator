"""Per-stage OUTPUT schemas for constrained decoding (his ask: enforce
structure instead of retrying). Derived from vmodel-contracts models - the
contracts stay the single source of truth; these are the same models minus
the fields the PIPELINE fills, never the LLM (session_id is setdefault-ed
after parse; SRS structure is injected deterministically at assembly).

Ollama enforces the schema grammar-side (native /api/chat `format` param,
structured outputs since Ollama 0.5). Grammar guarantees SHAPE, not content
and not length: a token-budget truncation still breaks, so tolerant
extraction + one retry stays as the net underneath.
"""
from __future__ import annotations

from vmodel_contracts import (
    AcceptanceTest,
    ArchitectureOverview,
    ItemSpec,
    SRSDocument,
    StakeholderNeedsSet,
)


def _body_schema(model_cls, drop: tuple[str, ...]) -> dict:
    """Object schema for an LLM-authored body: the contract model minus the
    fields the pipeline owns."""
    sch = model_cls.model_json_schema()
    out = {
        "type": "object",
        "properties": {k: v for k, v in sch.get("properties", {}).items() if k not in drop},
        "required": [r for r in sch.get("required", []) if r not in drop],
        "additionalProperties": False,
    }
    if "$defs" in sch:
        out["$defs"] = sch["$defs"]
    return out


def _array_schema(model_cls) -> dict:
    sch = model_cls.model_json_schema()
    item = {
        "type": "object",
        "properties": sch.get("properties", {}),
        "required": sch.get("required", []),
        "additionalProperties": False,
    }
    out = {"type": "array", "items": item}
    if "$defs" in sch:
        out["$defs"] = sch["$defs"]
    return out


def for_stage(stage: str) -> dict:
    if stage == "needs":
        return _body_schema(StakeholderNeedsSet, drop=("session_id",))
    if stage == "srs":
        # the LLM authors purpose/scope/requirements; structure is carried
        # deterministically from the intent model at assembly (ADR-0015).
        return _body_schema(SRSDocument, drop=("session_id", "intent_model_ref", "definitions",
                                               "constraints", "assumptions", "open_questions"))
    if stage == "architecture":
        return _body_schema(ArchitectureOverview, drop=("session_id",))
    if stage == "items":
        return _array_schema(ItemSpec)
    if stage == "tests":
        sch = AcceptanceTest.model_json_schema()
        item = {"type": "object", "properties": sch.get("properties", {}),
                "required": sch.get("required", []), "additionalProperties": False}
        out = {"type": "object",
               "properties": {"acceptance_tests": {"type": "array", "items": item},
                              "unit_module_tests": {"type": "array", "items": item}},
               "required": ["acceptance_tests", "unit_module_tests"],
               "additionalProperties": False}
        if "$defs" in sch:
            out["$defs"] = sch["$defs"]
        return out
    if stage == "review":
        return {"type": "array", "items": {"type": "string"}}
    raise KeyError(f"no output schema for stage {stage!r}")
