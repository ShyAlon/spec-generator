import pytest
from spec_generator.errors import UserFacingError
from spec_generator.graph import _check_ac_ids_stable


def srs(*counts):
    n = 0
    reqs = []
    for i, c in enumerate(counts, 1):
        acc = []
        for _ in range(c):
            n += 1
            acc.append({"criterion_id": f"AC-{n}", "statement": "s"})
        reqs.append({"requirement_id": f"R-{i}", "acceptance": acc})
    return {"functional": reqs, "non_functional": []}


def test_same_criterion_ids_pass():
    _check_ac_ids_stable(srs(2, 1, 3), srs(2, 1, 3))


def test_added_or_removed_criterion_is_refused_with_evidence():
    with pytest.raises(UserFacingError) as e:
        _check_ac_ids_stable(srs(2, 1, 3), srs(3, 1, 3))      # R-1 gains a criterion: every later AC shifts
    assert "renumber" in e.value.what and "R-1" in e.value.cause


def test_a_requirement_added_or_removed_is_refused():
    with pytest.raises(UserFacingError):
        _check_ac_ids_stable(srs(2, 1), srs(2))
