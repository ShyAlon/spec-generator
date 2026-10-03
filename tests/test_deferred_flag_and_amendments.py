"""IntentNeed.deferred drives deferral (marker match stays as fallback);
founder amendments render in the SRS scope with their source."""
from spec_generator.graph import srs_document_data
from spec_generator.skeleton import (build_srs_skeleton, deferred_needs, deferred_sn_ids,
                                     is_future_commitment)

INTENT_NEEDS = [
    {"need_id": "SN-1", "statement": "Track habits daily", "kind": "functional", "source_turns": [1]},
    {"need_id": "SN-2", "statement": "Community features", "kind": "functional", "source_turns": [2], "deferred": True},
]


def _intent(**over):
    base = {"session_id": "s", "problem_statement": "p", "target_users": ["u"], "core_value_proposition": "v",
            "tech_preferences": [], "stakeholder_needs": INTENT_NEEDS, "in_scope": ["a"], "out_of_scope": ["b"],
            "external_integrations": [], "constraints": [], "open_questions": [], "source_turns": {}}
    base.update(over)
    return base


def test_flag_without_marker_wording_is_deferred():
    assert is_future_commitment({"statement": "Community features", "deferred": True})
    assert not is_future_commitment({"statement": "Community features", "deferred": False})


def test_marker_fallback_still_works_for_old_payloads():
    assert is_future_commitment({"statement": "Eventually add sharing"})
    assert is_future_commitment("In the future, sharing")
    assert not is_future_commitment({"statement": "Track habits"})


def test_flag_found_through_intent_need_ids():
    intent = _intent()
    needs = {"needs": [{"need_id": "N-1", "statement": "Track habits daily", "intent_need_ids": ["SN-1"]},
                       {"need_id": "N-2", "statement": "Community features", "intent_need_ids": ["SN-2"]}]}
    assert deferred_sn_ids(intent) == {"SN-2"}
    assert [n["need_id"] for n in deferred_needs(needs, intent)] == ["N-2"]
    assert len(build_srs_skeleton(needs, intent)) == 1
    assert len(build_srs_skeleton(needs)) == 2  # no intent, no marker: unchanged behaviour


def test_scope_lists_deferred_flagged_need():
    d = srs_document_data(_intent(), {"scope_summary": "Habits only."})
    assert "Deferred to a future phase" in d["scope_summary"] and "Community features" in d["scope_summary"]


def test_amendments_render_with_source_and_stay_idempotent():
    intent = _intent(amendments=[{"amendment_id": "AM-1", "statement": "Songs are bundled with the app",
                                  "decided_on": "2026-10-01", "rationale": "licensing"}])
    d1 = srs_document_data(intent, {"scope_summary": "Habits only."})
    d2 = srs_document_data(intent, dict(d1))
    s = d1["scope_summary"]
    assert "AM-1: Songs are bundled with the app (source: founder-channel, decided 2026-10-01; rationale: licensing)" in s
    assert s.count("Founder amendments") == 1 and s.count("Deferred to a future phase") == 1
    assert s == d2["scope_summary"]


def test_no_amendments_leaves_scope_untouched():
    d = srs_document_data(_intent(stakeholder_needs=INTENT_NEEDS[:1]), {"scope_summary": "Habits only."})
    assert d["scope_summary"] == "Habits only."
