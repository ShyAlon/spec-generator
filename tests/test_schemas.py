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


def test_srs_schema_keeps_only_what_the_llm_authors():
    sch = for_stage("srs")
    props = sch["properties"]
    assert {"purpose", "scope_summary", "functional", "non_functional"} <= set(props)
    # structure is injected deterministically at assembly - the LLM must not emit it
    for owned in ("definitions", "constraints", "assumptions", "open_questions", "session_id"):
        assert owned not in props


def test_items_and_review_shapes():
    assert for_stage("items")["type"] == "array"
    assert for_stage("items")["items"]["type"] == "object"
    assert for_stage("review") == {"type": "array", "items": {"type": "string"}}


def test_tests_schema_two_arrays_of_acceptance_tests():
    sch = for_stage("tests")
    assert set(sch["required"]) == {"acceptance_tests", "unit_module_tests"}
    assert sch["properties"]["acceptance_tests"]["items"]["type"] == "object"


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
