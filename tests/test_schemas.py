"""Structured-output wiring: schemas derive from contracts (single source of
truth), the ollama provider routes to the native ChatOllama with format=schema,
and SPEC_GENERATOR_STRUCTURED=0 opts out."""
import os

from spec_generator import model_config
from spec_generator.schemas import for_stage


def test_needs_schema_drops_pipeline_owned_fields():
    sch = for_stage("needs")
    assert "needs" in sch["properties"] and "session_id" not in sch["properties"]
    assert sch["required"] == ["needs"]  # session_id is setdefault-ed post-parse
    assert sch["additionalProperties"] is False


def test_srs_schema_is_prose_fills_only():
    """ADR-0020: the SRS schema asks for prose per pipeline-assigned slot -
    no requirement ids to invent (the slot id is echoed), no kinds, no links,
    no AC ids, no intent-carried structure."""
    sch = for_stage("srs")
    assert set(sch["properties"]) == {"purpose", "scope_summary", "fills"}
    fill = sch["properties"]["fills"]["items"]
    assert set(fill["properties"]) == {"requirement_id", "title", "statement", "priority", "acceptance"}
    assert fill["properties"]["acceptance"]["items"] == {"type": "string"}  # AC ids are code-assigned
    for owned in ("functional", "non_functional", "definitions", "constraints",
                  "assumptions", "open_questions", "session_id", "kind", "need_ids"):
        assert owned not in str(sch)


def test_items_and_review_shapes():
    assert for_stage("items")["type"] == "array"
    assert for_stage("items")["items"]["type"] == "object"
    assert for_stage("review") == {"type": "array", "items": {"type": "string"}}


def test_tests_schema_is_prose_fills_only():
    sch = for_stage("tests")
    assert set(sch["properties"]) == {"fills"}
    fill = sch["properties"]["fills"]["items"]
    assert set(fill["properties"]) == {"test_id", "description", "runner", "definition"}
    # ids, levels, and criterion/spec links are code-owned (ADR-0020)
    for owned in ("level", "criterion_id", "spec_id"):
        assert owned not in str(sch)


def test_ollama_routes_to_native_chatollama_with_schema(monkeypatch):
    monkeypatch.delenv("SPEC_GENERATOR_STRUCTURED", raising=False)
    m = model_config.build_chat_model("needs")
    from langchain_ollama import ChatOllama
    assert isinstance(m, ChatOllama)
    assert m.format == for_stage("needs")
    assert m.num_predict == 8192  # tests stage carries 12288
    assert model_config.build_chat_model("tests").num_predict == 12288


def test_structured_opt_out_falls_back_to_openai_compat(monkeypatch):
    monkeypatch.setenv("SPEC_GENERATOR_STRUCTURED", "0")
    from langchain_openai import ChatOpenAI
    assert isinstance(model_config.build_chat_model("needs"), ChatOpenAI)


def test_remote_providers_unaffected(monkeypatch):
    monkeypatch.setenv("SPEC_GENERATOR_MODEL_NEEDS", "deepseek:deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    from langchain_openai import ChatOpenAI
    m = model_config.build_chat_model("needs")
    assert isinstance(m, ChatOpenAI)


def test_schemas_are_reduced_to_ollamas_grammar_subset():
    """Ollama 400s the call on constructs its grammar compiler rejects
    (#8325/#8185/#12726: pattern; #8444: $ref ordering). Every stage schema
    must be the boring subset: type/properties/required/items/enum/
    additionalProperties, refs inlined, no null-unions, no pipeline fields.
    The dropped constraints (pattern, minItems, ...) stay enforced by the
    contracts after parse - the grammar only guarantees shape."""
    allowed = {"type", "properties", "required", "items", "enum", "additionalProperties"}

    def walk(node, seen):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "properties":
                    seen.add(k)
                    for subschema in v.values():  # keys here are field names, not keywords
                        walk(subschema, seen)
                else:
                    seen.add(k)
                    walk(v, seen)
        elif isinstance(node, list):
            for v in node:
                walk(v, seen)
        return seen

    for stage in ("needs", "srs", "architecture", "items", "tests", "review"):
        sch = for_stage(stage)
        assert walk(sch, set()) <= allowed, f"{stage}: {walk(sch, set()) - allowed}"
        text = str(sch)
        assert '"null"' not in text  # anyOf T-or-null collapsed
        assert "contract_version" not in text and "session_id" not in text


def test_schema_field_names_match_contract_field_names():
    """The derived schema must emit exactly the contract's field names (minus
    pipeline-owned) - a rename or drop here makes contract validation reject
    every real run. Locked for needs; the same derivation builds all stages."""
    from vmodel_contracts import StakeholderNeedsSet

    need = StakeholderNeedsSet.model_json_schema()["$defs"]["StakeholderNeed"]
    contract_fields = set(need["properties"]) - {"session_id", "contract_version"}
    contract_required = set(need["required"]) - {"session_id", "contract_version"}
    item = for_stage("needs")["properties"]["needs"]["items"]
    assert set(item["properties"]) == contract_fields
    assert set(item["required"]) == contract_required



def _resolve_refs(node, defs):
    """Inline $refs the way _grammar_safe does, so raw and grammar-safe
    schemas can be compared node-for-node."""
    if isinstance(node, list):
        return [_resolve_refs(v, defs) for v in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        import copy
        name = node["$ref"].rsplit("/", 1)[-1]
        merged = copy.deepcopy(defs[name])
        merged.update({k: v for k, v in node.items() if k != "$ref"})
        node = merged
    return {k: _resolve_refs(v, defs) for k, v in node.items() if k != "$defs"}


def _object_shapes(node, out):
    """Multiset of (properties, required) at every object node, recursively."""
    owned = {"session_id", "contract_version"}  # pipeline prunes these at ANY depth
    if isinstance(node, dict):
        if "properties" in node or "required" in node:
            out.append((tuple(sorted(set(node.get("properties", {})) - owned)),
                        tuple(sorted(set(node.get("required", [])) - owned))))
        for v in node.get("properties", {}).values():
            _object_shapes(v, out)
        if "items" in node:
            _object_shapes(node["items"], out)
    return out


def test_grammar_safe_preserves_every_property_and_required_entry():
    """His live crash: _grammar_safe stripped the PROPERTY 'title' (a contract
    field whose name collides with a schema keyword) while 'title' stayed in
    required - the grammar then forbade a field the contract requires and the
    SRS gate crashed on 'title: Field required' x10. The strip must remove
    keyword-position metadata ONLY: property sets and required arrays are
    byte-identical before and after, at EVERY nesting level, for every stage."""
    from spec_generator.schemas import _grammar_safe, _raw_stage_schema

    for stage in ("needs", "srs", "architecture", "items", "tests"):
        raw = _raw_stage_schema(stage)
        resolved = _resolve_refs(raw, raw.get("$defs", {}))
        before = sorted(_object_shapes(resolved, []))
        after = sorted(_object_shapes(for_stage(stage), []))
        assert before == after, f"{stage}: grammar-safe changed the structure: {set(before) ^ set(after)}"


def test_keyword_named_fields_survive_the_strip():
    """Explicit lock on the collision class: fields literally named like
    schema keywords must stay declared AND required where the contract
    requires them."""
    dec = for_stage("architecture")["properties"]["decisions"]["items"]
    assert "title" in dec["properties"] and "title" in dec["required"]
    item = for_stage("items")["items"]
    assert "title" in item["properties"] and "title" in item["required"]
    from spec_generator.schemas import test_fill_schema
    tst = test_fill_schema([{"test_id": "T-1"}])["properties"]["fills"]["properties"]["T-1"]
    assert "description" in tst["properties"] and "description" in tst["required"]


def test_architecture_schema_drops_pipeline_assigned_adr_ids():
    dec = for_stage("architecture")["properties"]["decisions"]["items"]
    assert "decision_id" not in dec["properties"]
    assert "decision_id" not in dec["required"]
    assert {"title", "decision", "rationale"} <= set(dec["properties"])


def _project(instance, schema):
    """What the grammar lets the model emit: every schema-required field the
    instance carries, nothing the schema does not declare."""
    if isinstance(instance, list):
        return [_project(v, schema.get("items", {})) for v in instance]
    if not isinstance(instance, dict):
        return instance
    props = schema.get("properties", {})
    return {k: _project(v, props[k]) for k, v in instance.items() if k in props}


def test_schema_conformant_payloads_pass_full_contract_validation():
    """End-to-end alignment, per stage: a payload reduced to exactly what the
    grammar schema declares must, after the pipeline merges its owned
    structure (ADR-0020) and fills its own fields, pass the stage's FULL
    contract validation - and the SRS gate render path must not crash. If a
    schema ever drops or un-requires a field the contract needs, this fails
    before a user's run does."""
    from test_progress import CONTRACT_SRS, INTENT, PAYLOADS
    from vmodel_contracts import (AcceptanceTest, ArchitectureOverview, ItemSpec,
                                  SRSDocument, StakeholderNeedsSet)
    from spec_generator.graph import _assign_spec_ids, srs_document_data
    from spec_generator.render import render_srs
    from spec_generator.schemas import srs_fill_schema, test_fill_schema
    from spec_generator.skeleton import (assign_architecture_ids, build_srs_skeleton,
                                         build_test_skeleton, merge_srs_fills,
                                         merge_test_fills)

    needs = _project(PAYLOADS["needs"], for_stage("needs"))
    StakeholderNeedsSet.model_validate({**needs, "session_id": INTENT["session_id"]})

    srs_slots = build_srs_skeleton(PAYLOADS["needs"])
    fills = _project(PAYLOADS["srs"], srs_fill_schema(srs_slots))
    merged = merge_srs_fills(srs_slots, fills)
    doc = SRSDocument.model_validate(srs_document_data(
        INTENT, {**merged, "session_id": INTENT["session_id"], "intent_model_ref": INTENT["session_id"]}))
    assert render_srs(doc)  # the gate render path (cli._render_gate) must not crash

    arch = _project(PAYLOADS["architecture"], for_stage("architecture"))
    ArchitectureOverview.model_validate(
        {**assign_architecture_ids(arch), "session_id": INTENT["session_id"]})

    items = _assign_spec_ids(_project(PAYLOADS["items"], for_stage("items")))
    for i in items:
        ItemSpec.model_validate(i)

    test_slots = build_test_skeleton(CONTRACT_SRS, items)
    tfills = _project(PAYLOADS["tests"], test_fill_schema(test_slots))
    tests = merge_test_fills(test_slots, tfills)
    assert tests and all(AcceptanceTest.model_validate(t) for t in tests) is not None


def test_test_definition_grammar_is_canonical_but_content_stays_free():
    from spec_generator.schemas import test_fill_schema

    definition = test_fill_schema([{"test_id": "T-1"}])["properties"]["fills"]["properties"]["T-1"]["properties"]["definition"]
    assert set(definition["properties"]) == {
        "preconditions", "steps", "expected_outcomes", "cleanup", "cases", "execution"
    }
    assert set(definition["required"]) == set(definition["properties"])
    assert definition["additionalProperties"] is False
    case = definition["properties"]["cases"]["items"]
    assert set(case["properties"]) == {"name", "inputs", "expected_outcomes"}
    assert set(case["required"]) == set(case["properties"])
    assert case["additionalProperties"] is False
    # Free strings retain runner/domain detail; only the container is fixed.
    assert definition["properties"]["steps"]["items"] == {"type": "string"}
