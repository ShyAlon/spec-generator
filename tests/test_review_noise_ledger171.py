"""Noise classes back-tested on the 171-finding ledger from the cef8ed07 spec run (ids replaced by L-indexes,
hashes stripped). Labels are the analyst's class map; founder-kept classes must have ZERO matches."""
import json
from pathlib import Path

from spec_generator.findings_ledger import Ledger, revision_of
from spec_generator.review_noise import filter_review_noise, noise_class

FIX = json.loads((Path(__file__).parent / "fixtures" / "ledger171_noise_classes.json").read_text())
KEPT = {"storage-real", "REQ-leave-for-repair", "systemic-vague"}
REJECTED_BY_REGEX = {"G-restate"}


def _cls(o):
    return noise_class({"scope": o["scope"], "fix": o["text"]})


def test_fixture_is_the_171_ledger():
    assert len(FIX) == 171 and all(set(o) == {"id", "scope", "class", "text"} for o in FIX)


def test_zero_false_positives_on_founder_kept_findings():
    assert [o["id"] for o in FIX if o["class"] in KEPT and _cls(o)] == []


def test_class_g_catches_all_16_restating_findings():
    g = [o for o in FIX if o["class"] == "G-restate"]
    assert len(g) == 16 and all((_cls(o) or "").startswith("G") for o in g)


def test_widened_e_catches_most_tech_demand_findings_and_nothing_unlabelled_is_touched():
    aef = [o for o in FIX if o["class"] == "AEF-tech-demand" and _cls(o)]
    assert len(aef) >= 20
    others = [o["id"] for o in FIX if o["class"] not in ("AEF-tech-demand", "G-restate") and _cls(o)]
    assert others == []


def test_class_g_beats_the_rescue_phrase_and_requirement_scope_is_untouched():
    t = "Remove restated requirement text; reference requirement by ID only."
    assert noise_class({"scope": "spec", "fix": t}).startswith("G")
    assert noise_class({"scope": "requirement", "fix": t}) is None
    assert noise_class({"scope": "spec", "fix": "SPEC-2 contradicts R-4 and restates the requirement."}) is None


def test_class_h_is_defined_but_disabled():
    t = "Ensure all acceptance criteria are verifiable and observable."
    assert noise_class({"scope": "systemic", "fix": t}) is None


def test_refresh_does_not_reopen_a_noise_class_finding():
    ledger = Ledger()
    fid = ledger.add(rule_id="review.llm", rule_version="1", source="review:round-1", kind="semantic",
                     targets=["SPEC-1"], artifact_revision=revision_of("v1"),
                     summary="Ensure the spec references requirement R-9 without restating its text.",
                     evidence=[{"id": "x", "sha256": "aa"}], scope="spec",
                     detail={"scope": "spec", "fix": "Ensure the spec references requirement R-9 without restating its text."})
    ledger.founder_dispose(fid, actor="founder", source="decision-page", rationale="noise",
                           artifact_revision=revision_of("v1"))
    assert ledger.refresh({"SPEC-1": revision_of("v2")}) == []
    assert not ledger.open_findings()
    assert any(h["event"] == "refresh_filtered_noise" for h in ledger.history)


def test_filter_returns_g_class_for_systemic_scope_too():
    kept, filtered = filter_review_noise([{"scope": "systemic", "target_ids": [], "fix": "Avoid restating requirement text anywhere."}])
    assert not kept and filtered[0]["noise_class"].startswith("G")


def test_class_h_stays_disabled():
    # Class H (systemic-universal) matched founder-kept findings on the 171 ledger.
    # Shy: leave it off, it is technical debt. Do not enable without a new fixture gate.
    from spec_generator import review_noise
    assert review_noise.CLASS_H_ENABLED is False
