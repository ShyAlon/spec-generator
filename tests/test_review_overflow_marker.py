from spec_generator.findings_ledger import Ledger, revision_of

REV = revision_of("body")


def _finding(ledger, n, round_n=1, summary=None):
    return ledger.add(rule_id="review.llm", rule_version="1", source=f"review:round-{round_n}", kind="semantic",
                      targets=[f"SPEC-{n}"], artifact_revision=REV, summary=summary or f"finding {n}",
                      evidence=[{"id": "review:finding", "sha256": revision_of(f"f{n}{round_n}")}])


def _overflowed(round_n=1, chunk=3, count=25):
    ledger = Ledger()
    ids = [_finding(ledger, n, round_n) for n in range(2)]
    ledger.mark_incomplete(f"review:round-{round_n}-chunk-{chunk}", f"chunk returned {count} findings (bound 32)", returned=count)
    return ledger, ids


def _dispose(ledger, fid):
    ledger.founder_dispose(fid, actor="founder", source="decision-page", rationale="accepted", artifact_revision=REV)


def test_overflow_marker_blocks_while_any_round_finding_is_open():
    ledger, ids = _overflowed()
    _dispose(ledger, ids[0])
    gate = ledger.gate()
    assert gate["blocked"] and any("incomplete required check" in r for r in gate["reasons"])
    assert [m["scope"] for m in gate["incomplete"]] == ["review:round-1-chunk-3"] and gate["overflow_cleared"] == []


def test_overflow_marker_clears_once_every_round_finding_is_decided():
    ledger, ids = _overflowed()
    for fid in ids:
        _dispose(ledger, fid)
    gate = ledger.gate()
    assert not gate["blocked"] and gate["incomplete"] == [] and gate["overflow_cleared"] == ["review:round-1-chunk-3"]
    assert ledger.incomplete, "the stored marker is kept as history; only the gate result changes"


def test_overflow_marker_does_not_clear_on_other_rounds_open_findings_only():
    ledger, ids = _overflowed(round_n=1)
    _finding(ledger, 9, round_n=2)
    for fid in ids:
        _dispose(ledger, fid)
    assert ledger.gate()["overflow_cleared"] == ["review:round-1-chunk-3"]
    assert ledger.gate()["blocked"]  # the round-2 finding still blocks on its own


def test_other_incomplete_markers_still_block_even_when_all_findings_are_decided():
    ledger, ids = _overflowed()
    for fid in ids:
        _dispose(ledger, fid)
    ledger.mark_incomplete("review:round-1-chunk-4", "chunk output truncated")
    ledger.mark_incomplete("grounding:SPEC-1", "grounding unverified: x")
    ledger.mark_incomplete("review:round-1-chunk-5", "chunk returned 25 findings (bound 32)")  # no returned count
    gate = ledger.gate()
    assert gate["blocked"] and len(gate["incomplete"]) == 3 and gate["overflow_cleared"] == ["review:round-1-chunk-3"]


def test_marker_roundtrips_unchanged():
    ledger, ids = _overflowed()
    again = Ledger.from_dict(ledger.to_dict())
    assert again.incomplete == ledger.incomplete
