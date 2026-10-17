"""An oversized one-requirement review cluster is split so each prompt fits the admission ceiling."""
import json

from spec_generator import graph
from spec_generator.graph import REVIEW_PROMPT_MAX_BYTES, _split_review_cluster


def _item(n, kb):
    return {"spec_id": f"SPEC-{n}", "body_markdown": "x" * (kb * 1000)}


def _test(n, spec, kb=1, crit=None):
    return {"test_id": f"T-{n}", "spec_id": spec, "criterion_id": crit, "steps": "y" * (kb * 1000)}


def _size(parts_item, fixed=3000):
    items, tests = parts_item
    return fixed + sum(len(json.dumps(x)) for x in items + tests)


def test_fits_is_returned_unchanged():
    items = [_item(1, 10)]
    tests = [_test(1, "SPEC-1")]
    parts = _split_review_cluster(3000, items, tests)
    assert parts == [(items, tests)]
    assert parts[0][0] is items and parts[0][1] is tests


def test_oversized_cluster_is_split_and_each_part_fits():
    items = [_item(i, 30) for i in range(1, 6)]          # 150KB of items
    tests = [_test(i, f"SPEC-{i}", 5) for i in range(1, 6)]
    parts = _split_review_cluster(3000, items, tests)
    assert len(parts) >= 2
    for part in parts:
        assert _size(part) <= REVIEW_PROMPT_MAX_BYTES
    # every item and every test appears, each exactly once
    assert sorted(i["spec_id"] for p in parts for i in p[0]) == [f"SPEC-{i}" for i in range(1, 6)]
    assert sorted(t["test_id"] for p in parts for t in p[1]) == [f"T-{i}" for i in range(1, 6)]


def test_tests_travel_with_their_spec():
    items = [_item(i, 40) for i in range(1, 4)]
    tests = [_test(i, f"SPEC-{i}", 2) for i in range(1, 4)]
    for part_items, part_tests in _split_review_cluster(3000, items, tests):
        ids = {i["spec_id"] for i in part_items}
        assert all(t["spec_id"] in ids for t in part_tests)


def test_criterion_only_tests_are_kept():
    items = [_item(i, 45) for i in range(1, 4)]
    orphan = _test(9, "SPEC-OTHER", 3, crit="AC-1")
    parts = _split_review_cluster(3000, items, [orphan])
    assert sum(1 for p in parts for t in p[1] if t["test_id"] == "T-9") == 1


def test_item_with_too_many_tests_is_repeated_with_subsets():
    items = [_item(1, 10)]
    tests = [_test(i, "SPEC-1", 30) for i in range(1, 8)]  # 210KB of tests for one item
    parts = _split_review_cluster(3000, items, tests)
    assert len(parts) >= 3
    assert all([i["spec_id"] for i in p[0]] == ["SPEC-1"] for p in parts)
    assert sorted(t["test_id"] for p in parts for t in p[1]) == sorted(t["test_id"] for t in tests)
    assert all(_size(p) <= REVIEW_PROMPT_MAX_BYTES for p in parts)


def test_single_item_over_budget_stays_alone_and_is_not_cut():
    items = [_item(1, 120), _item(2, 10)]
    parts = _split_review_cluster(3000, items, [])
    assert [i["spec_id"] for i in parts[0][0]] == ["SPEC-1"]
    assert len(parts[0][0][0]["body_markdown"]) == 120_000
    assert [i["spec_id"] for p in parts[1:] for i in p[0]] == ["SPEC-2"]


def test_split_is_deterministic():
    items = [_item(i, 30) for i in range(1, 6)]
    tests = [_test(i, f"SPEC-{i}", 5) for i in range(1, 6)]
    assert _split_review_cluster(3000, items, tests) == _split_review_cluster(3000, items, tests)


def test_resume5_review_item6_would_be_split():
    # the real shape: 114,988 byte prompt (fixed ~ 20KB of requirement, methodology and frame)
    items = [_item(i, 12) for i in range(1, 8)]          # ~84KB
    tests = [_test(i, f"SPEC-{i}", 1) for i in range(1, 8)]
    assert 20_000 + 84_000 + 7_000 > 100_000 - 1
    parts = _split_review_cluster(20_000, items, tests)
    assert len(parts) == 2
    assert all(_size(p, 20_000) <= REVIEW_PROMPT_MAX_BYTES for p in parts)


def test_ceiling_arithmetic_documented_in_constant():
    # prompt + 7,316 overhead + 8,192 output cap must stay under the 128,000 ceiling with margin
    assert REVIEW_PROMPT_MAX_BYTES + 7316 + 8192 <= 128000 - 10_000
    assert graph.REVIEW_CHUNK_MAX_FINDINGS == 32


# ---- the review node itself (model call replaced; no provider, no database) ----
import pytest  # noqa: E402

from spec_generator.telemetry import UsageLog  # noqa: E402


class _Partials:
    def __init__(self):
        self.d = {}

    def load(self, k):
        return self.d.get(k)

    def save(self, k, v):
        self.d[k] = json.loads(json.dumps(v))

    def clear(self, k):
        self.d.pop(k, None)


def _state(n_items, item_kb, test_kb=3):
    items = [{"spec_id": f"SPEC-{i}", "requirement_ids": ["R-1"], "body_markdown": "x" * (item_kb * 1000), "revision": "a"}
             for i in range(1, n_items + 1)]
    tests = [{"test_id": f"T-{i}", "spec_id": f"SPEC-{i}", "criterion_id": "AC-1", "steps": "y" * (test_kb * 1000)}
             for i in range(1, n_items + 1)]
    return {"srs": {"functional": [{"requirement_id": "R-1", "acceptance": [{"criterion_id": "AC-1"}], "statement": "s"}],
                    "non_functional": []},
            "items": items, "tests": tests, "repair_quick": True, "findings_ledger": graph.Ledger().to_dict()}


def _node(partials):
    return graph.build_graph(UsageLog("t"), partials=partials).nodes["review"].bound


def test_review_node_small_cluster_is_one_unchanged_call(monkeypatch):
    calls = []
    monkeypatch.setattr(graph, "_call_json", lambda stage, payload, log, progress, **kw: calls.append(kw["item_label"]) or [])
    _node(_Partials()).invoke(_state(2, 5))
    assert calls == ["review 1/1 (1 requirements)"]


def test_review_node_oversized_cluster_is_split_and_merged(monkeypatch):
    calls = []

    def fake(stage, payload, log, progress, **kw):
        calls.append((len(payload.encode()), kw["item_label"], payload))
        return [{"scope": "spec", "target_ids": ["SPEC-1"], "fix": f"Fix {len(calls)}."}]
    monkeypatch.setattr(graph, "_call_json", fake)
    out = _node(_Partials()).invoke(_state(5, 30))
    assert len(calls) >= 2
    assert all(size + len(graph.METHODOLOGY.encode()) <= REVIEW_PROMPT_MAX_BYTES for size, _, _ in calls)
    assert all(f"part {k}/{len(calls)}" in label for k, (_, label, _) in enumerate(calls, 1))
    assert all("PART " in payload and "Report findings only for what you see here." in payload for _, _, payload in calls)
    assert {f["fix"] for f in out["review_fixes"]} == {f"Fix {i}." for i in range(1, len(calls) + 1)}


def test_review_node_resume_skips_finished_parts(monkeypatch):
    partials = _Partials()
    seen = []

    def boom(stage, payload, log, progress, **kw):
        seen.append(kw["item_label"])
        if len(seen) == 2:
            raise RuntimeError("stop after the first part")
        return []
    monkeypatch.setattr(graph, "_call_json", boom)
    with pytest.raises(RuntimeError):
        _node(partials).invoke(_state(5, 30))
    assert len(seen) == 2
    again = []
    monkeypatch.setattr(graph, "_call_json", lambda stage, payload, log, progress, **kw: again.append(kw["item_label"]) or [])
    _node(partials).invoke(_state(5, 30))
    assert len(again) == 2  # 3 parts in all; part 1 was saved and is not asked again
    assert not any(label.startswith("review 1/1 part 1/") for label in again)
