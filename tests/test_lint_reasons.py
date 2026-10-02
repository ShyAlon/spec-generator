"""SRS lint reasons say what was checked, what was expected and what was found; duplicates merge."""
import json
from pathlib import Path

from vmodel_contracts.text_qa import srs_draft_findings
from spec_generator.lint_reasons import describe, group_findings

REQ = {"requirement_id": "R-11", "title": "Performance and Responsiveness",
       "statement": "The application shall stay responsive.",
       "acceptance": [{"criterion_id": "AC-11.1", "statement": "Each tap is acknowledged within 200 ms."}]}


def test_number_finding_is_one_readable_sentence_with_location():
    lint = srs_draft_findings({"functional": [REQ], "non_functional": []}, "intent only")
    groups = group_findings(lint, "R-11")
    assert len(groups) == 1                         # the number is one finding, not two
    text = describe("R-11", groups[0])
    assert "R-11 acceptance.AC-11.1" in text and "Checked:" in text and "Expected:" in text and "Found: '200'" in text
    assert "figure the interview never gave" in text


def test_same_span_from_two_rules_merges_and_names_both():
    lint = srs_draft_findings({"functional": [{"requirement_id": "R-1", "title": "t", "statement":
                               "The API shall return HTTP 201 on create.", "acceptance": []}], "non_functional": []}, "nothing")
    group = [g for g in group_findings(lint, "R-1") if g[0]["span"] == "201"][0]
    assert {f["rule"] for f in group} == {"http_status", "absent_from_interview"}
    text = describe("R-1", group)
    assert "HTTP status code" in text and "never gave" in text


def test_graph_uses_needs_and_conops_in_the_corpus_and_merged_reasons():
    import inspect
    from spec_generator.graph import build_graph
    src = inspect.getsource(build_graph)
    assert "json.dumps([state['intent_model'],state.get('needs'),state.get('conops')])" in src
    assert "describe(str(rid),group)" in src


def test_cli_does_not_stop_a_rewind_to_srs_on_the_stale_unresolved_review():
    import inspect
    from spec_generator import cli
    src = inspect.getsource(cli)
    assert "args.from_stage not in ('needs','srs') and (prior.get('semantic_reviews') or {}).get('srs')" in src
