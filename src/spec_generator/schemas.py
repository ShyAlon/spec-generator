"""Per-stage OUTPUT schemas for constrained decoding (his ask: enforce
structure instead of retrying). needs/architecture derive from vmodel-contracts
models - the contracts stay the single source of truth; those are the same
models minus the fields the PIPELINE fills, never the LLM (session_id +
contract_version are set post-parse; ADR ids are pipeline-assigned, ADR-0020).
srs/tests are prose-fill shapes by design (ADR-0020: code owns the skeleton -
ids, kinds, ordering, trace links - the LLM authors prose only).

Ollama enforces the schema grammar-side (native /api/chat `format` param,
structured outputs since Ollama 0.5). Grammar guarantees SHAPE, not content
and not length: a token-budget truncation still breaks, so tolerant
extraction + one retry stays as the net underneath. The strip removes
keyword-position metadata ONLY: a contract field whose NAME collides with
a schema keyword (Requirement.title, AcceptanceTest.description) is kept
- dropping it made the grammar forbid a field the contract requires, and
every structured run failed validation downstream.

Every schema passes _grammar_safe() before it leaves this module: Ollama's
schema->grammar compiler only accepts a boring subset of JSON Schema and
400s the whole call otherwise (ollama#8325/#8185/#12726: `pattern` ->
"invalid JSON schema in format"; #8444: $ref ordering). So we send type /
properties / required / items / enum / additionalProperties only, with refs
inlined and T-or-null anyOfs collapsed. Nothing is lost: pattern, minItems,
defaults and friends are constraints the CONTRACTS validate after parse -
the grammar only needs to guarantee shape.
"""
from __future__ import annotations


# Keywords Ollama's grammar compiler rejects or ignores - stripped before send.
_STRIP = {"title", "description", "default", "pattern", "minItems", "maxItems",
          "format", "examples", "minLength", "maxLength", "multipleOf",
          "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
          "uniqueItems", "contains", "patternProperties", "propertyNames",
          "const", "$schema"}

_PIPELINE_OWNED = ("session_id", "contract_version")


def _grammar_safe(schema: dict) -> dict:
    """Reduce a pydantic JSON schema to the subset Ollama accepts: refs
    inlined (no $defs/$ref), T-or-null anyOfs collapsed to T, pipeline-owned
    fields pruned at any depth, metadata and post-parse-enforced constraints
    stripped."""
    import copy

    defs = schema.get("$defs", {})

    def resolve(node, *, prop_map: bool = False):
        if isinstance(node, list):
            return [resolve(v) for v in node]
        if not isinstance(node, dict):
            return node
        if prop_map:
            # Keys here are CONTRACT FIELD NAMES, not schema keywords - a field
            # literally named 'title' or 'description' (Requirement.title,
            # AcceptanceTest.description, ...) must survive the strip. Only the
            # pipeline-owned fields are pruned.
            return {k: resolve(v) for k, v in node.items() if k not in _PIPELINE_OWNED}
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            target = copy.deepcopy(defs[name])
            target.update({k: v for k, v in node.items() if k != "$ref"})
            node = target
        if "anyOf" in node:
            options = [o for o in node["anyOf"] if o != {"type": "null"}]
            if len(options) == 1:
                merged = {k: v for k, v in node.items() if k != "anyOf"}
                merged.update(options[0])
                node = merged
        out = {}
        for k, v in node.items():
            if k in _STRIP or k in ("$defs", "$ref"):
                continue  # keyword-position metadata only - never field names
            if k == "properties":
                out[k] = resolve(v, prop_map=True)
            elif k == "required":
                out[k] = [r for r in v if r not in _PIPELINE_OWNED]
            else:
                out[k] = resolve(v)
        return out

    return resolve(schema)


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
        "properties": {k: v for k, v in sch.get("properties", {}).items()
                    if k not in drop and k not in _PIPELINE_OWNED},
        "required": [r for r in sch.get("required", []) if r not in drop and r not in _PIPELINE_OWNED],
        "additionalProperties": False,
    }
    if "$defs" in sch:
        out["$defs"] = sch["$defs"]
    return out


def _array_schema(model_cls) -> dict:
    sch = model_cls.model_json_schema()
    item = {
        "type": "object",
        "properties": {k: v for k, v in sch.get("properties", {}).items() if k not in _PIPELINE_OWNED},
        "required": [r for r in sch.get("required", []) if r not in _PIPELINE_OWNED],
        "additionalProperties": False,
    }
    out = {"type": "array", "items": item}
    if "$defs" in sch:
        out["$defs"] = sch["$defs"]
    return out


def _raw_stage_schema(stage: str) -> dict:
    if stage == "needs":
        return _body_schema(StakeholderNeedsSet, drop=("session_id",))
    if stage == "srs":
        # ADR-0020: code owns the requirement skeleton (ids, kinds, links) -
        # the LLM fills PROSE per slot. Not contract-derived on purpose.
        fill = {"type": "object",
                "properties": {"requirement_id": {"type": "string"},
                               "title": {"type": "string"},
                               "statement": {"type": "string"},
                               "priority": {"enum": ["must", "should", "could"]},
                               "acceptance": {"type": "array", "items": {"type": "string"}}},
                "required": ["requirement_id", "title", "statement", "priority", "acceptance"],
                "additionalProperties": False}
        return {"type": "object",
                "properties": {"purpose": {"type": "string"},
                               "scope_summary": {"type": "string"},
                               "fills": {"type": "array", "items": fill}},
                "required": ["purpose", "scope_summary", "fills"],
                "additionalProperties": False}
    if stage == "architecture":
        # ADR-0020: bulk proposal, code disposes - ADR ids are assigned by the
        # pipeline, so decision_id leaves the schema.
        sch = _body_schema(ArchitectureOverview, drop=("session_id",))
        dec = sch["$defs"]["ArchitectureDecision"]
        dec["properties"].pop("decision_id", None)
        dec["required"] = [r for r in dec.get("required", []) if r != "decision_id"]
        # ADR-0021: the contract keeps driver nullable ONLY for pre-0.8.0
        # migrated artifacts; the model must always emit a real driver, so the
        # grammar narrows it to the closed enum.
        dec["properties"]["driver"] = {"enum": ["founder_constraint", "ecosystem_maturity",
                                                "community_adoption", "team_fit", "simplicity"]}
        return sch
    if stage == "items":
        return _array_schema(ItemSpec)
    if stage == "tests":
        # ADR-0020: code owns the test skeleton (T-ids, levels, criterion/spec
        # links); the LLM fills prose per slot.
        fill = {"type": "object",
                "properties": {"test_id": {"type": "string"},
                               "description": {"type": "string"},
                               "runner": {"type": "string"},
                               "definition": {"type": "object"}},
                "required": ["test_id", "description", "runner", "definition"],
                "additionalProperties": False}
        return {"type": "object",
                "properties": {"fills": {"type": "array", "items": fill}},
                "required": ["fills"],
                "additionalProperties": False}
    if stage == "review":
        return {"type": "array", "items": {"type": "string"}}
    raise KeyError(f"no output schema for stage {stage!r}")


def for_stage(stage: str) -> dict:
    """The schema handed to Ollama for a stage: contracts-derived, then reduced
    to the grammar-safe subset (see module docstring)."""
    if stage == "review":
        return {"type": "array", "items": {"type": "string"}}
    return _grammar_safe(_raw_stage_schema(stage))

