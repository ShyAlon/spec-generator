"""QA-engineer test-generation prompt, runner/execution validator, review prompt scope."""
import pytest

from spec_generator.prompts import STAGE_PROMPTS
from spec_generator.skeleton import ModelStructureError, check_runner_execution


def t(runner, **ex):
    return {"test_id": "T-7", "runner": runner,
            "definition": {"preconditions": [], "steps": [], "expected_outcomes": [], "cleanup": [], "cases": [], "execution": ex}}


OK_OUT = ["The score screen shows 10 of 10."]


def tt(runner, outcomes=OK_OUT, steps=(), **ex):
    d = t(runner, **ex)
    d["definition"]["expected_outcomes"] = list(outcomes)
    d["definition"]["steps"] = list(steps)
    return d


def test_dom_is_renamed_to_ui():
    with pytest.raises(ModelStructureError, match=r"T-7.*'dom' is now 'ui'"):
        check_runner_execution([tt("dom")])
    check_runner_execution([tt("ui")])


def test_selector_in_execution_is_rejected_with_product_term_feedback():
    with pytest.raises(ModelStructureError, match=r"T-7.*selector is not allowed.*product terms"):
        check_runner_execution([tt("ui", selector="#score")])


@pytest.mark.parametrize("text", [
    "Locate [data-testid='score'] on the page.",
    "Click the element with data-testid=score.",
    "Find the node via a[href=x].",
    "Call document.querySelector on the score display.",
])
def test_markup_in_steps_or_outcomes_is_rejected(text):
    with pytest.raises(ModelStructureError, match=r"names UI markup"):
        check_runner_execution([tt("ui", steps=[text])])
    with pytest.raises(ModelStructureError, match=r"names UI markup"):
        check_runner_execution([tt("ui", outcomes=[text])])


def test_code_identifiers_in_prose_are_not_markup():
    check_runner_execution([tt("cli", outcomes=["loadCourse(raw) returns .toneAssetId for each lesson, in section #3.1."], command="x")])


@pytest.mark.parametrize("outcome", ["The page loads.", "no error", "No errors occur.", "The element is visible",
                                     "The app renders successfully", "", "   "])
def test_vacuous_outcomes_alone_are_rejected(outcome):
    with pytest.raises(ModelStructureError, match=r"states no checkable outcome"):
        check_runner_execution([tt("ui", outcomes=[outcome])])


def test_no_outcomes_at_all_is_rejected():
    with pytest.raises(ModelStructureError, match=r"states no checkable outcome"):
        check_runner_execution([tt("ui", outcomes=[])])


def test_one_real_outcome_beside_a_vacuous_one_passes():
    check_runner_execution([tt("ui", outcomes=["The page loads.", "The streak shows 3 days."])])


def test_case_outcomes_count():
    d = tt("ui", outcomes=[])
    d["definition"]["cases"] = [{"name": "a", "inputs": [], "expected_outcomes": ["The score shows 7."]}]
    check_runner_execution([d])


def test_http_requires_url_method_status_and_names_each_missing_one():
    with pytest.raises(ModelStructureError) as e:
        check_runner_execution([tt("http", url="/a")])
    assert "method" in str(e.value) and "expected_status" in str(e.value) and "execution.url" not in str(e.value).split("(missing:")[1]
    check_runner_execution([tt("http", url="/a", method="GET", expected_status=200)])


def test_cli_requires_command_and_static_semantic_need_nothing():
    with pytest.raises(ModelStructureError, match="command"):
        check_runner_execution([tt("cli")])
    check_runner_execution([tt("cli", command="app --help"), tt("static"), tt("semantic")])


def test_model_object_definitions_are_accepted():
    from vmodel_contracts import AcceptanceTest
    d = tt("ui")
    d.update(level="acceptance", criterion_id="AC-1", description="d")
    check_runner_execution([AcceptanceTest.model_validate(d).model_dump()])


def test_test_prompt_is_qa_framed_and_scoped():
    p = STAGE_PROMPTS["tests"].lower()
    assert "qa engineer" in p and "only what its one" in p and "observable" in p
    assert "expected_status" in p and "never write a selector" in p and "'ui'" in p
    assert "'dom'" not in p and "execution.selector" not in p


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


def test_agents_md_tells_the_developer_to_bind_ui_tests():
    from spec_generator.render import render_agents_md
    from vmodel_contracts import ArchitectureOverview, SpecManifest
    arch = ArchitectureOverview.model_construct(session_id="s", stack_entries=[], stack_entry_dispositions=[],
                                                components=[], decisions=[], non_functionals={})
    md = render_agents_md(arch, [], SpecManifest.model_construct(milestones=[]), [])
    assert "UI tests are yours to bind" in md and "ui-bindings.json" in md and "data-testid" in md
