"""Founder-named tech candidates + the dated curated candidate lists
(founder decision 2026-09-13): founder-named techs are injected with weight
and their verdict is ALWAYS in the record; vague rejections re-ask; off-list
picks need an ADR; the as-of date prints in the run log."""
import json

import pytest

from spec_generator import model_config
from spec_generator.graph import (ModelStructureError, _announce_candidate_lists,
                                  _candidate_block, _check_candidate_grounding,
                                  _founder_named_techs, _specific_reason)
from spec_generator.tech_candidates import (TECH_CANDIDATES, TECH_CANDIDATES_AS_OF,
                                            TECH_CANDIDATES_PATH, diff_candidates,
                                            validate_artifact)
from test_decision_drivers import _arch

INTENT_SUPABASE = {"session_id": "s", "tech_preferences": [
    {"area": "backend", "preference": "free resources like Supabase",
     "strength": "preferred", "source_turns": [3]}]}


def _arch_with(stack, options):
    from spec_generator.tech_candidates import resolve_technology
    sels=[]; entries=[]
    for k,v in stack:
        tid,row=resolve_technology(v)
        if row is None: tid="off-list:"+v.lower().replace(" ","-")
        sel={"slot_id":"slot:"+k.replace("_","-"),"technology_id":tid,"display_name":v}
        if tid.startswith("off-list:"): sel["off_list_justification"]="test fixture"
        sels.append(sel);entries.append({"key":k,"value":v,"technology_id":tid,"decision_id":"ADR-1"})
    return _arch({"session_id": "s", "stack_entries": entries,
                  "components": [{"name": "c", "responsibility": "r", "requirement_ids": ["R-1"]}],
                  "decisions": [{"decision_id":"ADR-1","title": "Database", "decision": stack[0][1],
                                 "rationale": "Fits the free-tier constraint and realtime sync.",
                                 "driver": "simplicity", "options_considered": options,
                                 "executes_concept_ids":[],"technology_selections":sels}],
                  "non_functionals": {}})


# --- artifact + freshness ---

def test_committed_artifact_is_valid_and_dated():
    data = json.loads(TECH_CANDIDATES_PATH.read_text())
    assert validate_artifact(data) == []
    assert data["as_of"] == TECH_CANDIDATES_AS_OF


def test_artifact_mobile_family_present():
    mobile = {n for n, _ in TECH_CANDIDATES["mobile_framework"]}
    assert {"React Native", "Flutter", "SwiftUI"} <= mobile
    health = {n for n, _ in TECH_CANDIDATES["health_data"]}
    assert "HealthKit" in health


def _row(name, kinds=("database",), platforms=("any",), aliases=(), note="n"):
    return {"name": name, "technology_id": "tech:" + name.lower().replace(" ", "-"), "kinds": list(kinds), "platforms": list(platforms),
            "runtime": None, "ecosystem": None,
            "aliases": sorted({name.lower(), *aliases}), "note": note}


def _artifact(rows, as_of="2026-09-13", kinds=("database",)):
    return {"as_of": as_of, "kinds": list(kinds), "technologies": rows}


def test_validate_artifact_refuses_garbage():
    assert validate_artifact({"as_of": "today", "kinds": ["database"], "technologies": []})
    # too few rows for a kind
    assert validate_artifact(_artifact([_row("PostgreSQL")]))
    # undeclared kind on a row
    assert validate_artifact(_artifact([_row("A"), _row("B"), _row("C", kinds=("nope",))]))
    # duplicate alias across rows - ambiguous derived lexicon
    assert validate_artifact(_artifact([_row("A", aliases=("dup",)), _row("B", aliases=("dup",)),
                                        _row("C")]))
    # self-alias missing
    bad = _row("X"); bad["aliases"] = ["other"]
    assert validate_artifact(_artifact([bad, _row("Y"), _row("Z")]))
    # a valid minimal artifact passes
    assert validate_artifact(_artifact([_row("A"), _row("B"), _row("C")])) == []


def test_diff_candidates_added_removed_and_platform_changes():
    old = _artifact([_row("PostgreSQL"), _row("SQLite")])
    new = _artifact([_row("PostgreSQL"), _row("TiDB")])
    assert diff_candidates(old, new) == ["database: +tidb -sqlite"]
    flipped = _artifact([_row("PostgreSQL"), _row("SQLite", platforms=("web",))])
    assert diff_candidates(old, flipped) == ["~sqlite: platforms ['any'] -> ['web']"]
    assert diff_candidates(old, old) == []


# --- extraction + prompt injection ---

def test_founder_named_tech_extracted_from_preference_text():
    assert _founder_named_techs(INTENT_SUPABASE) == ["supabase"]
    assert _founder_named_techs({"tech_preferences": [
        {"area": "backend", "preference": "whatever is cheapest", "strength": "preferred",
         "source_turns": [1]}]}) == []


def test_candidate_block_marks_founder_candidate_with_weight():
    block = _candidate_block("database", ["supabase"])
    assert TECH_CANDIDATES_AS_OF in block
    assert "FOUNDER-NAMED CANDIDATE" in block and "Supabase" in block
    assert "weigh them heavily" in block


def test_candidate_block_maps_mobile_slot_by_substring():
    block = _candidate_block("mobile_framework", [])
    assert "React Native" in block and "Flutter" in block
    assert "FOUNDER-NAMED" not in _candidate_block("mobile_framework", ["supabase"])  # wrong kind


# --- specificity ---

def test_vague_rejection_reasons_fail_specificity():
    assert not _specific_reason("not suitable")
    assert not _specific_reason("Better alternatives exist.")
    assert not _specific_reason("too heavy")
    assert _specific_reason("Supabase has no managed mobile push channel, which R-4 requires")


# --- the assembly gate ---

def test_founder_tech_dropped_from_record_fails():
    arch = _arch_with([("database", "Firebase Firestore")],
                      [{"option": "PostgreSQL", "reason": "No realtime sync needed client-side."},
                       {"option": "MongoDB", "reason": "Document model fits but self-hosting breaks the free tier."}])
    with pytest.raises(ModelStructureError, match="Supabase"):
        _check_candidate_grounding(arch, INTENT_SUPABASE)


def test_founder_tech_rejected_with_vague_reason_fails():
    arch = _arch_with([("database", "Firebase Firestore")],
                      [{"option": "Supabase", "reason": "not suitable"}])
    with pytest.raises(ModelStructureError, match="vague"):
        _check_candidate_grounding(arch, INTENT_SUPABASE)


def test_founder_tech_rejected_with_specific_reason_passes():
    arch = _arch_with([("database", "Firebase Firestore")],
                      [{"option": "Supabase",
                        "reason": "Supabase has no managed mobile push channel, which R-4 requires."}])
    _check_candidate_grounding(arch, INTENT_SUPABASE)


def test_founder_tech_adopted_needs_no_rejection_entry():
    arch = _arch_with([("database", "Supabase")], [])
    _check_candidate_grounding(arch, INTENT_SUPABASE)


def test_off_list_pick_without_structured_justification_is_unrepresentable():
    from pydantic import ValidationError
    from vmodel_contracts import TechnologySelection
    with pytest.raises(ValidationError, match="off_list_justification"):
        TechnologySelection(slot_id="slot:database", technology_id="off-list:zylith-9000-db", display_name="Zylith-9000 DB")
    selected = TechnologySelection(slot_id="slot:database", technology_id="off-list:zylith-9000-db",
                                   display_name="Zylith-9000 DB", off_list_justification="R-7 needs its conflict model")
    assert selected.off_list_justification


def test_on_list_and_lexicon_picks_need_no_extra_adr():
    arch = _arch_with([("database", "Firebase Firestore")], [])
    _check_candidate_grounding(arch, {"tech_preferences": []})


# --- the run-log announcement ---

def test_as_of_date_prints_in_run_log(monkeypatch):
    lines = []
    monkeypatch.setattr(model_config, "_console_line", lines.append)
    _announce_candidate_lists(["supabase"])
    assert any(TECH_CANDIDATES_AS_OF in ln for ln in lines)
    assert any("Supabase" in ln for ln in lines)


# --- the matrix ---

def test_matrix_rows_carry_platform_attributes():
    from spec_generator.tech_candidates import tech_row
    assert tech_row("React")["platforms"] == ["web"]          # the canonical regression case
    assert tech_row("react-native")["platforms"] == ["ios", "android"]
    assert tech_row("HealthKit")["platforms"] == ["ios"]
    assert tech_row("PostgreSQL")["platforms"] == ["any"]
    assert tech_row("novel-tech-xyz") is None                 # a matrix miss is not a verdict


def test_kind_lexicons_are_derived_from_the_matrix():
    from spec_generator.graph import _known_tech, _wrong_kind
    assert _known_tech("D3.js") and _known_tech("ts")
    assert _wrong_kind("data_visualization_library", "FastAPI") == "framework"
    assert _wrong_kind("data_visualization_library", "D3.js") is None


def test_product_platforms_derived_from_intent():
    from spec_generator.graph import _product_platforms
    assert _product_platforms({"problem_statement": "An iOS habit app"}) == {"ios"}
    assert _product_platforms({"problem_statement": "p",
                               "external_integrations": ["HealthKit"]}) == {"ios"}
    assert _product_platforms({"problem_statement": "a web app in the browser"}) == {"web"}
    assert _product_platforms({"problem_statement": "backend service"}) == set()


# --- gate 3 end to end: the React-for-mobile regression case ---

import pytest as _pytest
from langchain_core.messages import AIMessage as _AIMessage

from conftest import SYSTEM_BASELINE
from spec_generator import model_config as _mc
from spec_generator.errors import UserFacingError as _UFE
from spec_generator.graph import build_graph
from spec_generator.telemetry import UsageLog
from test_architecture_micro_pipeline import _MicroModel
from test_progress import INTENT

INTENT_IOS = {**INTENT, "problem_statement": "An iOS habit-tracking app",
              "external_integrations": ["HealthKit"]}


def _run(model_cls, **kw):
    calls = []
    mp = _pytest.MonkeyPatch()
    mp.setattr(_mc, "build_chat_model",
               lambda stage, **k: model_cls(stage, k.get("schema") or {}, calls, **kw))
    try:
        return build_graph(UsageLog("s")).invoke(
            {"intent_model": dict(INTENT_IOS), "system_baseline": SYSTEM_BASELINE},
            {"configurable": {"thread_id": model_cls.__name__}})
    finally:
        mp.undo()


class _ReactForMobile(_MicroModel):
    DEPART_VALUES = {**_MicroModel.DEPART_VALUES,
                     "mobile_framework": ("React Native", "Adopt React Native for mobile_framework")}

    def stream(self, messages):
        props = set(self.schema.get("properties", {}))
        if props == {"technology", "off_list_justification", "baseline_refs"} and any(
                "THE STACK KEY TO DECIDE: mobile_framework" in m.content for m in messages):
            if any("cannot answer this slot directly" in m.content for m in messages):
                yield _AIMessage(content=json.dumps({"technology": "React Native", "off_list_justification": None, "baseline_refs": []}))
            else:
                yield _AIMessage(content=json.dumps({"technology": "React", "off_list_justification": None, "baseline_refs": []}))
            return
        yield from super().stream(messages)


def test_react_rejected_from_a_mobile_frontend_slot_with_retry_feedback():
    r = _run(_ReactForMobile, depart=True, depart_key="mobile_framework",
             plan_keys=["language", "mobile_framework"])
    fw = next(e for e in r["architecture"]["stack_entries"] if e["key"] == "mobile_framework")
    assert fw["value"] == "React Native"  # the web-only candidate was rejected with feedback


class _InsistentReact(_MicroModel):
    DEPART_VALUES = {**_MicroModel.DEPART_VALUES,
                     "mobile_framework": ("React Native", "Adopt React Native for mobile_framework")}

    def stream(self, messages):
        props = set(self.schema.get("properties", {}))
        if props == {"technology", "off_list_justification", "baseline_refs"} and any(
                "THE STACK KEY TO DECIDE: mobile_framework" in m.content for m in messages):
            yield _AIMessage(content=json.dumps({"technology": "React", "off_list_justification": None, "baseline_refs": []}))
            return
        if "driver" in props and '"migration": true' not in "\n".join(str(m.content) for m in messages):
            yield _AIMessage(content=json.dumps({
                "title": "Frontend: React", "decision": "Use React",
                "rationale": "Web-first PWA is a deliberate product call: one codebase now, "
                             "the native wrapper ships later.",
                "driver": "simplicity", "options_considered": [], "baseline_refs": [],
                "evidence": [], "affected_repositories": [], "executes_concept_ids": []}))
            return
        yield from super().stream(messages)


def test_insistence_requires_the_platform_justification_in_an_adr():
    r = _run(_InsistentReact, depart=True, depart_key="mobile_framework",
             plan_keys=["language", "mobile_framework"])
    fw = next(e for e in r["architecture"]["stack_entries"] if e["key"] == "mobile_framework")
    assert fw["value"] == "React"  # contested, not silently blocked
    adr = next(d for d in r["architecture"]["decisions"] if d["title"] == "Frontend: React")
    assert "web" in adr["rationale"].lower()  # the platform trade-off is on record for veto


class _InsistentReactNoAdr(_MicroModel):
    DEPART_VALUES = {**_MicroModel.DEPART_VALUES,
                     "mobile_framework": ("React Native", "Adopt React Native for mobile_framework")}

    def stream(self, messages):
        props = set(self.schema.get("properties", {}))
        if props == {"technology", "off_list_justification", "baseline_refs"} and any(
                "THE STACK KEY TO DECIDE: mobile_framework" in m.content for m in messages):
            yield _AIMessage(content=json.dumps({"technology": "React", "off_list_justification": None, "baseline_refs": []}))
            return
        yield from super().stream(messages)


def test_insistence_without_an_adr_justification_fails_loudly():
    with _pytest.raises(_UFE, match="candidate-grounding"):
        _run(_InsistentReactNoAdr, depart=True, depart_key="mobile_framework",
             plan_keys=["language", "mobile_framework"])


def test_adr_call_rejects_bare_string_options_before_assembly():
    """The live DeepSeek shape must trigger call-level feedback, not stage assembly."""
    from pathlib import Path
    source=(Path(__file__).parents[1]/"src"/"spec_generator"/"graph.py").read_text()
    assert 'options_considered must contain' in source
    assert 'bare strings' in source
    # Deterministic conversion is deliberately unsafe: a string has no reason,
    # and inventing an empty reason would only defer the same contract failure.


def test_off_list_pick_credits_structured_stack_to_adr_link_not_exact_prose():
    from spec_generator.graph import _check_candidate_grounding
    from vmodel_contracts import ArchitectureOverview
    long_pick=("Per-insurer adapter behind a normalized claim-exchange boundary "
               "with the concrete channel per insurer pending research")
    data={"session_id":"s","stack_entries":[{"key":"insurer_integration","value":long_pick,
      "technology_id":"off-list:per-insurer-adapter","decision_id":"ADR-7"}],"components":[],
      "decisions":[{"decision_id":"ADR-7","title":"Insurer integration boundary",
      "decision":"Use a normalized correspondence port","rationale":"Insurers differ in channel and protocol",
      "driver":"simplicity","options_considered":[],"executes_concept_ids":[],
      "technology_selections":[{"slot_id":"slot:insurer-integration","technology_id":"off-list:per-insurer-adapter",
      "display_name":long_pick,"off_list_justification":"Keeps insurer-specific transport differences outside the claim domain while concrete channels remain research-bound."}]}],
      "non_functionals":{}}
    _check_candidate_grounding(ArchitectureOverview.model_validate(data),{"tech_preferences":[],"stakeholder_needs":[]})


def test_ui_framework_candidates_exclude_backend_substitutes():
    block = _candidate_block("ui_framework", [])
    assert "React" in block and "Vue" in block
    assert all(name not in block for name in ("Express:", "FastAPI:", "LangChain:", "Flask:", "NestJS:", "LangGraph:"))
