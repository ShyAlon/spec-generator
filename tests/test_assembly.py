"""Assembly invariants: the deterministic tail catches what generation misses."""
from vmodel_contracts import (
    AcceptanceCriterion, AcceptanceTest, ItemSpec, Priority, Requirement, RequirementKind,
    RequirementSet, SpecItemKind, StakeholderNeed, StakeholderNeedsSet, VModelLevel,
)
from spec_generator.assembly import build_traceability, order_milestones, validate_traceability


def _needs():
    return StakeholderNeedsSet(session_id="s", needs=[
        StakeholderNeed(need_id="N-1", stakeholder="end user", statement="Book a lesson", kind=RequirementKind.FUNCTIONAL, intent_fields=["in_scope"], intent_need_ids=["SN-1"]),
    ])


def _reqs():
    return RequirementSet(session_id="s", requirements=[
        Requirement(requirement_id="R-1", title="Booking", statement="Users book lessons",
                    priority=Priority.MUST, kind=RequirementKind.FUNCTIONAL, need_ids=["N-1"], intent_fields=["in_scope"],
                    acceptance=[AcceptanceCriterion(criterion_id="AC-1", statement="POST /bookings returns 201",
                                                    executable_hint="http POST /bookings -> 201")]),
    ])


def test_clean_set_has_no_violations():
    items = [ItemSpec(spec_id="SPEC-1", kind=SpecItemKind.API_SURFACE, title="Booking API",
                      body_markdown="...", requirement_ids=["R-1"])]
    tests = [AcceptanceTest(test_id="T-1", level=VModelLevel.ACCEPTANCE, criterion_id="AC-1",
                            spec_id="SPEC-1", description="booking works", runner="http",
                            definition={"method": "POST", "url": "/bookings", "expect": 201})]
    assert validate_traceability(_needs(), _reqs(), items, tests) == []


def test_orphan_requirement_is_caught():
    violations = validate_traceability(_needs(), _reqs(), [], [])
    assert any("R-1" in v for v in violations)
    assert any("AC-1" in v for v in violations)


def test_unknown_need_reference_is_caught():
    reqs = _reqs()
    reqs.requirements[0].need_ids = ["N-99"]
    assert any("N-99" in v for v in validate_traceability(_needs(), reqs, [], []))


def test_milestones_respect_dependencies():
    a = ItemSpec(spec_id="SPEC-1", kind=SpecItemKind.DATA_MODEL, title="Model", body_markdown=".", requirement_ids=["R-1"])
    b = ItemSpec(spec_id="SPEC-2", kind=SpecItemKind.API_SURFACE, title="API", body_markdown=".", requirement_ids=["R-1"], depends_on=["SPEC-1"])
    ms = order_milestones([b, a], [])
    assert ms[0].spec_ids == ["SPEC-1"] and ms[1].spec_ids == ["SPEC-2"]


def test_traceability_links_requirement_to_test():
    items = [ItemSpec(spec_id="SPEC-1", kind=SpecItemKind.API_SURFACE, title="X", body_markdown=".", requirement_ids=["R-1"])]
    tests = [AcceptanceTest(test_id="T-1", level=VModelLevel.ACCEPTANCE, criterion_id="AC-1",
                            description="d", runner="http", definition={})]
    [entry] = build_traceability(_reqs(), items, tests)
    assert entry.spec_ids == ["SPEC-1"] and entry.test_ids == ["T-1"]

