"""QA-engineer test-generation prompt, runner/execution validator, review prompt scope."""
import pytest

from spec_generator.prompts import STAGE_PROMPTS
from spec_generator.skeleton import ModelStructureError, check_runner_execution


def t(runner, **ex):
    return {"test_id": "T-7", "runner": runner,
            "definition": {"preconditions": [], "steps": [], "expected_outcomes": [], "cleanup": [], "cases": [], "execution": ex}}


def test_dom_requires_selector_named_in_error():
    with pytest.raises(ModelStructureError, match=r"T-7.*dom.*selector"):
        check_runner_execution([t("dom")])
    check_runner_execution([t("dom", selector="#x")])


def test_http_requires_url_method_status_and_names_each_missing_one():
    with pytest.raises(ModelStructureError) as e:
        check_runner_execution([t("http", url="/a")])
    assert "method" in str(e.value) and "expected_status" in str(e.value) and "execution.url" not in str(e.value).split("(missing:")[1]
    check_runner_execution([t("http", url="/a", method="GET", expected_status=200)])


def test_cli_requires_command_and_static_semantic_need_nothing():
    with pytest.raises(ModelStructureError, match="command"):
        check_runner_execution([t("cli")])
    check_runner_execution([t("cli", command="app --help"), t("static"), t("semantic")])


def test_model_object_definitions_are_accepted():
    from vmodel_contracts import AcceptanceTest
    d = t("dom", selector="#x")
    d.update(level="acceptance", criterion_id="AC-1", description="d")
    check_runner_execution([AcceptanceTest.model_validate(d).model_dump()])


def test_test_prompt_is_qa_framed_and_scoped():
    p = STAGE_PROMPTS["tests"].lower()
    assert "qa engineer" in p and "only what its one" in p and "observable" in p
    assert "selector" in p and "expected_status" in p


def test_review_prompt_drops_restated_requirement_text_and_scopes_tests():
    r = STAGE_PROMPTS["review"].lower()
    assert "restate" not in r
    assert "must cite a requirement or ac id" in r
    assert "contradicting its own criterion or spec" in r


def test_tool_loop_prompt_narrows_the_add_test_invitation():
    from spec_generator.prompts import TOOL_PROMPTS
    p = TOOL_PROMPTS["tests"].lower()
    assert "qa engineer" in p and "do not call add_test unless" in p
    assert "may propose additional tests" not in p
