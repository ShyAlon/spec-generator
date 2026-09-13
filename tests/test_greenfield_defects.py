"""2026-09-13 greenfield regen defects (bundle artifacts-2026-09-12_20-18-37):
1. founder tech named in a NEED ('free resources like Supabase' = N-8) was
   invisible to founder-candidate extraction (it read tech_preferences only);
2. stack-entry hygiene: vendor in a strategy slot, same value under two keys;
3. open questions answered downstream carried stale in the SRS;
4. same-vendor concentration (Firebase 3x) unflagged."""
from datetime import date

from test_decision_drivers import _arch

from spec_generator.graph import (_founder_named_techs,
                                  _resolve_answered_open_questions)
from spec_generator.render import architecture_challenges


# ---------- 1. founder candidate from a need statement ----------

def test_founder_tech_named_in_need_is_extracted():
    intent = {"session_id": "s", "tech_preferences": [],
              "stakeholder_needs": [
                  {"need_id": "SN-8", "kind": "functional", "source_turns": [3],
                   "statement": "The app should use free resources like Supabase where possible."}]}
    assert _founder_named_techs(intent) == ["supabase"]


def test_need_mention_flows_into_candidate_grounding():
    # the gate that already exists for tech_preferences must fire for the
    # need-mention too: an architecture with no Supabase anywhere and no
    # rejection reason fails grounding
    from spec_generator.graph import _check_candidate_grounding
    arch = _arch({"session_id": "s",
                  "stack_entries": [{"key": "database_system", "value": "PostgreSQL"}],
                  "components": [{"name": "web", "responsibility": "UI",
                                  "requirement_ids": ["R-1"]}],
                  "decisions": [{"title": "DB", "decision": "PostgreSQL",
                                 "rationale": "default choice", "driver": "simplicity",
                                 "options_considered": []}],
                  "non_functionals": {}})
    intent = {"session_id": "s", "tech_preferences": [],
              "stakeholder_needs": [
                  {"need_id": "SN-8", "kind": "functional", "source_turns": [3],
                   "statement": "Use free resources like Supabase where possible."}]}
    try:
        _check_candidate_grounding(arch, intent)
        raised = False
    except Exception as exc:
        raised = True
        assert "supabase" in str(exc).lower()
    assert raised, "founder tech from a need must be gated exactly like a preference"


def test_founder_tech_rejected_with_specific_reason_passes():
    from spec_generator.graph import _check_candidate_grounding
    arch = _arch({"session_id": "s",
                  "stack_entries": [{"key": "database_system", "value": "PostgreSQL"}],
                  "components": [{"name": "web", "responsibility": "UI",
                                  "requirement_ids": ["R-1"]}],
                  "decisions": [{"title": "DB", "decision": "PostgreSQL",
                                 "rationale": "default choice", "driver": "simplicity",
                                 "options_considered": [
                                     {"option": "Supabase",
                                      "reason": "vendor-managed Postgres limits the "
                                                "custom point-aggregation functions we need"}]}],
                  "non_functionals": {}})
    intent = {"session_id": "s", "tech_preferences": [],
              "stakeholder_needs": [
                  {"need_id": "SN-8", "kind": "functional", "source_turns": [3],
                   "statement": "Use free resources like Supabase where possible."}]}
    _check_candidate_grounding(arch, intent)  # no raise


# ---------- 2 + 4. stack-entry hygiene + vendor concentration ----------

def _defect_arch():
    return _arch({"session_id": "s",
                  "stack_entries": [
                      {"key": "database_system", "value": "PostgreSQL"},
                      {"key": "data_persistence_service", "value": "PostgreSQL"},
                      {"key": "notification_service", "value": "Firebase Cloud Messaging"},
                      {"key": "permissions_management", "value": "Firebase Authentication"},
                      {"key": "resource_utilization_strategy", "value": "Firebase"}],
                  "components": [{"name": "web", "responsibility": "UI",
                                  "requirement_ids": ["R-1"]}],
                  "decisions": [{"title": "t", "decision": "d", "rationale": "r",
                                 "driver": "simplicity", "options_considered": []}],
                  "non_functionals": {}})


def test_vendor_in_strategy_slot_flagged():
    flags = architecture_challenges(_defect_arch())
    assert any("strategy slot answered with a vendor" in f
               and "resource_utilization_strategy" in f for f in flags)


def test_cross_key_duplicate_value_flagged():
    flags = architecture_challenges(_defect_arch())
    assert any("same technology under two keys" in f and "PostgreSQL" in f
               and "database_system" in f and "data_persistence_service" in f
               for f in flags)


def test_vendor_concentration_flagged():
    flags = architecture_challenges(_defect_arch())
    assert any("vendor concentration" in f and "firebase" in f.lower() for f in flags)


def test_clean_stack_flags_none_of_these():
    arch = _arch({"session_id": "s",
                  "stack_entries": [{"key": "database_system", "value": "PostgreSQL"},
                                    {"key": "framework", "value": "Django"}],
                  "components": [{"name": "web", "responsibility": "UI",
                                  "requirement_ids": ["R-1"]}],
                  "decisions": [{"title": "t", "decision": "d", "rationale": "r",
                                 "driver": "simplicity", "options_considered": []}],
                  "non_functionals": {}})
    flags = architecture_challenges(arch)
    assert not any("strategy slot" in f or "two keys" in f or "concentration" in f
                   for f in flags)


# ---------- 3. open-question resolution ----------

def test_answered_open_question_resolved_and_linked():
    state = {"srs": {"open_questions": [
                 "Specific data storage solution for persisting user points across devices",
                 "Specific UI elements or features for clarity and transparency about permissions",
                 "Examples of proven UI frameworks for similar applications"],
             "session_id": "s"},
             "architecture": {"stack_entries": [
                 {"key": "data_persistence_service", "value": "PostgreSQL"},
                 {"key": "frontend_framework", "value": "React"}]}}
    resolved = _resolve_answered_open_questions(state, persist=None)
    kept = state["srs"]["open_questions"]
    assert len(resolved) == 1
    assert "PostgreSQL" in resolved[0] and "data_persistence_service" in resolved[0]
    # the storage question is gone; the UI-feature question stays open
    assert not any("storage" in q for q in kept)
    assert any("UI elements" in q for q in kept)


def test_no_architecture_no_resolution():
    state = {"srs": {"open_questions": ["Specific data storage solution?"]},
             "architecture": {}}
    assert _resolve_answered_open_questions(state, persist=None) == []
    assert state["srs"]["open_questions"] == ["Specific data storage solution?"]
