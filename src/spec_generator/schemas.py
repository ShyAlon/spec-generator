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
    SpecItemKind,
    SRSDocument,
    StakeholderNeedsSet,
    TestDefinition,
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


def srs_fill_schema(slots: list[dict]) -> dict:
    """The SRS fill grammar, generated PER SKELETON (2026-09-09 live failure
    'R-5: slot not filled'): every slot id is an explicitly required property
    of the fills object, so a missing slot is impossible by construction -
    the post-hoc merge check becomes a safety net, not the only guard."""
    fill = {"type": "object",
            "properties": {"title": {"type": "string"},
                           "statement": {"type": "string"},
                           "priority": {"enum": ["must", "should", "could"]},
                           "acceptance": {"type": "array", "items": {"type": "string"}}},
            "required": ["title", "statement", "priority", "acceptance"],
            "additionalProperties": False}
    ids = [sl["requirement_id"] for sl in slots]
    return {"type": "object",
            "properties": {"purpose": {"type": "string"},
                           "scope_summary": {"type": "string"},
                           "fills": {"type": "object",
                                     "properties": {i: fill for i in ids},
                                     "required": ids,
                                     "additionalProperties": False}},
            "required": ["purpose", "scope_summary", "fills"],
            "additionalProperties": False}


def _test_definition_schema() -> dict:
    """Grammar-safe shape derived from the shared TestDefinition contract."""
    return _grammar_safe(TestDefinition.model_json_schema())


def test_fill_schema(slots: list[dict]) -> dict:
    """Same per-skeleton tightening for the tests stage: every test slot id is
    a required property of the fills object."""
    fill = {"type": "object",
            "properties": {"description": {"type": "string"},
                           "runner": {"type": "string"},
                           "definition": _test_definition_schema()},
            "required": ["description", "runner", "definition"],
            "additionalProperties": False}
    ids = [sl["test_id"] for sl in slots]
    return {"type": "object",
            "properties": {"fills": {"type": "object",
                                     "properties": {i: fill for i in ids},
                                     "required": ids,
                                     "additionalProperties": False}},
            "required": ["fills"],
            "additionalProperties": False}


def items_schema(requirement_ids: list[str]) -> dict:
    """The items grammar, generated PER RUN (2026-09-09 live failure: the
    model invented R-001..R-005 and left all ten real requirements
    uncovered): requirement_ids references are an ENUM of the real ids, so
    citing a nonexistent requirement is impossible by construction. The spec
    SET itself stays model-proposed (module grouping is design analysis, not
    skeleton structure); coverage is validated at the stage, not at assembly.
    """
    item = {"type": "object",
            "properties": {"slug": {"type": "string"},  # the pipeline numbers SPEC-1..N; the model names slugs
                           "kind": {"enum": [k.value for k in SpecItemKind]},
                           "title": {"type": "string"},
                           "body_markdown": {"type": "string"},
                           "requirement_ids": {"type": "array",
                                               "items": {"enum": list(requirement_ids)}},
                           "depends_on": {"type": "array", "items": {"type": "string"}}},
            "required": ["slug", "kind", "title", "body_markdown", "requirement_ids"],
            "additionalProperties": False}
    return {"type": "array", "items": item}


def srs_overview_schema() -> dict:
    """ADR-0022: the purpose/scope micro-call."""
    return {"type": "object",
            "properties": {"purpose": {"type": "string"},
                           "scope_summary": {"type": "string"}},
            "required": ["purpose", "scope_summary"],
            "additionalProperties": False}


def srs_slot_fill_schema() -> dict:
    """ADR-0022: one requirement fill per call - a bad fill fails one cheap
    call, never a long composite payload."""
    return {"type": "object",
            "properties": {"title": {"type": "string"},
                           "statement": {"type": "string"},
                           "priority": {"enum": ["must", "should", "could"]},
                           "acceptance": {"type": "array", "items": {"type": "string"}}},
            "required": ["title", "statement", "priority", "acceptance"],
            "additionalProperties": False}


def items_plan_schema(requirement_ids: list[str]) -> dict:
    """ADR-0022: the items stage splits into PLAN (module grouping - the
    analysis - with references enum-constrained) then one body call per spec."""
    plan = {"type": "object",
            "properties": {"slug": {"type": "string"},  # the pipeline numbers SPEC-1..N; the model names slugs
                           "kind": {"enum": [k.value for k in SpecItemKind]},
                           "title": {"type": "string"},
                           "requirement_ids": {"type": "array",
                                               "items": {"enum": list(requirement_ids)}},
                           "depends_on": {"type": "array", "items": {"type": "string"}}},
            "required": ["slug", "kind", "title", "requirement_ids"],
            "additionalProperties": False}
    return {"type": "array", "items": plan}


def items_body_schema() -> dict:
    return {"type": "object",
            "properties": {"body_markdown": {"type": "string"}},
            "required": ["body_markdown"],
            "additionalProperties": False}


def _enum_or_str(values: list[str]) -> dict:
    """An empty enum is a broken grammar (nothing can satisfy it); fall back
    to a plain string and let the stage check judge (greenfield: any ref is
    rejected with retry feedback)."""
    return {"enum": list(values)} if values else {"type": "string"}


# --- 0.13.0 architecture micro-pipeline grammars (see prompts.ARCH_*) --------
# Every schema is shallow and per-slot: one small object (or a slot list) per
# call, with references enum-constrained to values code supplied - a small
# local model cannot drift across fields it never sees together.

def arch_stack_plan_schema() -> dict:
    """Plan call A: which technology slots to decide. Keys only."""
    return {"type": "object",
            "properties": {"stack_keys": {"type": "array", "items": {"type": "string"}}},
            "required": ["stack_keys"],
            "additionalProperties": False}


def arch_stack_entry_schema(valid_refs: list[str], technology_ids: list[str]) -> dict:
    """One stack slot: a concrete technology, grounded research delegation, or genuine exclusion."""
    concrete = {"type": "object", "properties": {
        "technology": {"type": "string", "minLength": 1},
        "off_list_justification": {"type": ["string", "null"]},
        "baseline_refs": {"type": "array", "items": _enum_or_str(valid_refs)}},
        "required": ["technology", "off_list_justification", "baseline_refs"], "additionalProperties": False}
    research = {"type": "object", "properties": {
        "status": {"const": "research_required"}, "research_brief": {"type": "string", "minLength": 1}},
        "required": ["status", "research_brief"], "additionalProperties": False}
    excluded = {"type": "object", "properties": {
        "status": {"const": "not_applicable"}, "reason": {"type": "string", "minLength": 1}},
        "required": ["status", "reason"], "additionalProperties": False}
    # Keep concrete properties at the root for grammar adapters that inspect
    # property names before honoring oneOf; branch required-sets still enforce
    # the three legal shapes.
    return {"properties": concrete["properties"],
            "oneOf": [concrete, research, excluded]}


def arch_component_plan_schema(requirement_ids: list[str], valid_refs: list[str]) -> dict:
    """Plan call B: the component slots. Requirement and ref enums are the
    real ids/refs, so coverage and grounding are checked against truth."""
    slot = {"type": "object",
            "properties": {"name": {"type": "string"},
                           "requirement_ids": {"type": "array",
                                               "items": _enum_or_str(requirement_ids)},
                           "baseline_refs": {"type": "array",
                                             "items": _enum_or_str(valid_refs)},
                           "migration": {"type": "boolean"}},
            "required": ["name", "requirement_ids", "baseline_refs", "migration"],
            "additionalProperties": False}
    return {"type": "array", "items": slot}


def arch_component_body_schema() -> dict:
    return {"type": "object",
            "properties": {"responsibility": {"type": "string"}},
            "required": ["responsibility"],
            "additionalProperties": False}


def arch_adr_plan_schema() -> dict:
    """Plan call C: OPTIONAL design-decision slots only. Migration ADRs are
    pipeline-generated from the declared departures (the 2026-09-12 live phi4
    failure: the model planned a ci_tool-absence slot with migration=false and
    authorizes=[] and left stack:ci_tool unauthorized) - the model cannot miss
    or mis-set a mechanical obligation it is never asked for."""
    slot = {"type": "object",
            "properties": {"slug": {"type": "string"}},
            "required": ["slug"],
            "additionalProperties": False}
    return {"type": "array", "items": slot}


def arch_adr_body_schema(valid_refs: list[str], repo_names: list[str], concept_ids: list[str]) -> dict:
    """One ADR against its fixed slot. Ids, migration flag, and authorizations
    stay pipeline-owned; the model authors the decision substance."""
    return {"type": "object",
            "properties": {"title": {"type": "string"},
                           "decision": {"type": "string"},
                           "rationale": {"type": "string"},
                           "driver": {"enum": ["founder_constraint", "ecosystem_maturity",
                                               "community_adoption", "team_fit", "simplicity"]},
                           "options_considered": {"type": "array", "items": {
                               "type": "object",
                               "properties": {"option": {"type": "string"},
                                              "reason": {"type": "string"}},
                               "required": ["option", "reason"],
                               "additionalProperties": False}},
                           "baseline_refs": {"type": "array",
                                             "items": _enum_or_str(valid_refs)},
                           "evidence": {"type": "array", "items": {"type": "string"}},
                           "affected_repositories": {"type": "array",
                                                     "items": _enum_or_str(repo_names)},
                           "executes_concept_ids": {"type": "array", "items": _enum_or_str(concept_ids)}},
            "required": ["title", "decision", "rationale", "driver", "options_considered",
                         "baseline_refs", "evidence", "affected_repositories", "executes_concept_ids"],
            "additionalProperties": False}


def arch_nfr_schema() -> dict:
    """One non-functional position against a fixed requirement slot."""
    return {"type": "object",
            "properties": {"position": {"type": "string"}},
            "required": ["position"],
            "additionalProperties": False}



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
        # The assembled-shape projection (review gates, test projection). Live
        # generation uses the micro-pipeline schemas below - the model never
        # emits this whole shape in one call (2026-09-11 phi4 failure).
        # ADR-0020: ADR ids are pipeline-assigned, so decision_id leaves the
        # grammar; migration_adr_id on stack entries is pipeline-assigned at
        # assembly too (0.13.0: from the authorizing ADR slot, not the model).
        sch = _body_schema(ArchitectureOverview, drop=("session_id", "baseline_id"))
        comp = sch["$defs"]["ComponentSpec"]
        comp["required"] = sorted(set(comp.get("required", [])) | {"baseline_refs", "migration", "migration_adr_title"})
        dec = sch["$defs"]["ArchitectureDecision"]
        dec["properties"].pop("decision_id", None)
        dec["required"] = [r for r in dec.get("required", []) if r != "decision_id"]
        dec["required"] = sorted(set(dec["required"]) | {"baseline_refs", "migration", "evidence", "affected_repositories"})
        # ADR-0021: the contract keeps driver nullable ONLY for pre-0.8.0
        # migrated artifacts; the model must always emit a real driver, so the
        # grammar narrows it to the closed enum.
        dec["properties"]["driver"] = {"enum": ["founder_constraint", "ecosystem_maturity",
                                                "community_adoption", "team_fit", "simplicity"]}
        entry = sch["$defs"]["StackEntry"]
        entry["properties"].pop("migration_adr_id", None)
        entry["required"] = [r for r in entry.get("required", []) if r != "migration_adr_id"]
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
                               "definition": _test_definition_schema()},
                "required": ["test_id", "description", "runner", "definition"],
                "additionalProperties": False}
        return {"type": "object",
                "properties": {"fills": {"type": "array", "items": fill}},
                "required": ["fills"],
                "additionalProperties": False}
    if stage == "review":
        return {"type": "array", "items": {"type": "object", "properties": {
            "scope": {"enum": ["requirement", "spec", "test", "systemic"]},
            "target_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "fix": {"type": "string", "maxLength": 320}},
            "required": ["scope", "target_ids", "fix"], "additionalProperties": False}}
    raise KeyError(f"no output schema for stage {stage!r}")


def for_stage(stage: str) -> dict:
    """The schema handed to Ollama for a stage: contracts-derived, then reduced
    to the grammar-safe subset (see module docstring)."""
    if stage == "review":
        return {"type": "array", "items": {"type": "object", "properties": {
            "scope": {"enum": ["requirement", "spec", "test", "systemic"]},
            "target_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "fix": {"type": "string", "maxLength": 320}},
            "required": ["scope", "target_ids", "fix"], "additionalProperties": False}}
    return _grammar_safe(_raw_stage_schema(stage))


