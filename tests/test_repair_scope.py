from spec_generator.graph import _review_scope_requirements

STATE = {"srs": {"functional": [
    {"requirement_id": "R-1", "acceptance": [{"criterion_id": "AC-1"}]},
    {"requirement_id": "R-2", "acceptance": [{"criterion_id": "AC-2"}]},
    {"requirement_id": "R-3", "acceptance": [{"criterion_id": "AC-3"}]}]},
    "items": [{"spec_id": "SPEC-1", "requirement_ids": ["R-1", "R-2"]}, {"spec_id": "SPEC-2", "requirement_ids": ["R-3"]}],
    "tests": [{"test_id": "T-1", "criterion_id": "AC-3", "spec_id": "SPEC-2"}]}


def fx(scope, ids):
    return {"scope": scope, "target_ids": ids, "fix": "x"}


def test_requirement_repairs_scope_the_rereview_to_those_requirements():
    assert _review_scope_requirements(STATE, [fx("requirement", ["R-1"]), fx("requirement", ["R-3"])]) == ["R-1", "R-3"]


def test_spec_and_test_repairs_add_their_linked_requirements():
    assert _review_scope_requirements(STATE, [fx("spec", ["SPEC-1"])]) == ["R-1", "R-2"]
    assert _review_scope_requirements(STATE, [fx("test", ["T-1"])]) == ["R-3"]


def test_systemic_or_empty_means_review_everything():
    assert _review_scope_requirements(STATE, [fx("requirement", ["R-1"]), {"scope": "systemic", "target_ids": [], "fix": "x"}]) is None
    assert _review_scope_requirements(STATE, []) is None


def test_repair_proposal_lists_changed_and_unchanged_targets():
    from spec_generator.graph import _repair_proposal
    state = {"srs": {"functional": [{"requirement_id": "R-1", "statement": "old", "acceptance": []},
                                    {"requirement_id": "R-2", "statement": "same", "acceptance": []}]}}
    data = {"functional": [{"requirement_id": "R-1", "statement": "new", "acceptance": []},
                           {"requirement_id": "R-2", "statement": "same", "acceptance": []}]}
    p = _repair_proposal(state, data, [{"scope": "requirement", "target_ids": ["R-1", "R-2"], "fix": "x"}], ["F-1"])
    assert list(p["changed"]) == ["R-1"] and p["changed"]["R-1"]["before"]["statement"] == "old"
    assert p["unchanged_targets"] == ["R-2"] and p["srs"] is data
