from conftest import SYSTEM_BASELINE
"""0.13.0 architecture micro-pipeline: the 2026-09-11 phi4 failure made
structural.

The failure: one giant call had to keep `stack` and `stack_baseline_refs`
key-synchronized; phi4 emitted a schema-legal bogus `stack_baseline_refs`
string key inside `stack` and the maps disagreed. The micro-pipeline makes
that class impossible: no call's schema carries two fields that must agree,
references are enum-constrained to code-computed truth, ids and the
departure -> ADR wiring are pipeline-owned, and assembly is deterministic.
"""
import json
import os

import pytest
from langchain_core.messages import AIMessage

from spec_generator import model_config
from spec_generator.cli import _make_partials
from spec_generator.errors import UserFacingError
from spec_generator.graph import (build_graph, _legacy_arch_fill, _merge_adr_slots,
                                  _required_adr_slots, _validate_adr_plan,
                                  _validate_component_plan, _validate_stack_plan)
from spec_generator.schemas import (arch_adr_plan_schema, arch_stack_entry_schema,
                                    arch_stack_plan_schema)
from spec_generator.skeleton import ModelStructureError
from spec_generator.telemetry import UsageLog
from test_progress import INTENT, PAYLOADS


def test_no_micro_schema_has_two_fields_that_must_agree():
    # the phi4 mistake is not representable: there is no stack_baseline_refs
    # anywhere a model writes, and every object is additionalProperties:false
    for sch in (arch_stack_plan_schema(), arch_stack_entry_schema(["repo:x"], ["tech:python"]),
                arch_adr_plan_schema()):
        assert "stack" not in json.dumps(sch).lower().replace("stack_keys", "")
    entry = arch_stack_entry_schema(["repo:x"], ["tech:python"])
    assert set(entry["properties"]) == {"technology", "off_list_justification", "baseline_refs"}
    assert len(entry["oneOf"]) == 3
    concrete, research, excluded = entry["oneOf"]
    assert concrete["additionalProperties"] is False
    assert set(concrete["properties"]) == {"technology", "off_list_justification", "baseline_refs"}
    assert concrete["properties"]["baseline_refs"]["items"] == {"enum": ["repo:x"]}
    assert research["properties"]["status"] == {"const": "research_required"}
    assert excluded["properties"]["status"] == {"const": "not_applicable"}
    plan = arch_adr_plan_schema()
    # the model cannot miss or mis-set a migration ADR: the plan schema has
    # no migration/authorizes fields at all - they are pipeline-owned
    assert set(plan["items"]["properties"]) == {"slug"}


def test_old_shape_whole_architecture_answer_fails_loudly():
    # a model answering a micro-call with the retired whole-artifact shape is
    # a retryable model error, never a silent pass
    with pytest.raises(ModelStructureError):
        _validate_stack_plan(PAYLOADS["architecture"])


def test_required_migration_slots_are_code_generated_not_model_planned():
    # the 2026-09-12 live phi4 failure, at the validator level: a departure
    # (stack:ci_tool) existed and the model's plan forgot to authorize it.
    # Required slots are now pipeline-owned, one per declared departure.
    required = _required_adr_slots(["stack:ci_tool", "component:billing"])
    assert required == [{"slug": "migrate-stack-ci-tool", "migration": True,
                         "authorizes": ["stack:ci_tool"]},
                        {"slug": "migrate-component-billing", "migration": True,
                         "authorizes": ["component:billing"]}]
    merged = _merge_adr_slots(required, [{"slug": "design-x", "migration": False,
                                          "authorizes": []}])
    assert [s["decision_id"] for s in merged] == ["ADR-1", "ADR-2", "ADR-3"]
    assert merged[0]["authorizes"] == ["stack:ci_tool"]  # required first; ids code-owned


def test_adr_plan_model_slots_cannot_touch_pipeline_owned_fields():
    taken = [s["slug"] for s in _required_adr_slots(["stack:ci_tool"])]
    ok = _validate_adr_plan([{"slug": "design-x"}], taken)
    assert ok == [{"slug": "design-x", "migration": False, "authorizes": []}]
    # the exact 2026-09-12 phi4 answer shape now fails loudly, with the fix named
    with pytest.raises(ModelStructureError, match="pipeline-owned"):
        _validate_adr_plan([{"slug": "ci_tool-absence", "migration": False, "authorizes": []}],
                           taken)
    # Echoing a pipeline-owned required slug is deterministic no-op noise.
    assert _validate_adr_plan([{"slug": "migrate-stack-ci-tool"}], taken) == []
    with pytest.raises(ModelStructureError, match="duplicate ADR slugs"):
        _validate_adr_plan([{"slug": "x"}, {"slug": "x"}], [])


def test_component_plan_rejects_duplicates_and_uncovered_requirements():
    with pytest.raises(ModelStructureError, match="duplicate component names"):
        _validate_component_plan([{"name": "a"}, {"name": "a"}], ["R-1"])
    with pytest.raises(ModelStructureError, match="covered by no component"):
        _validate_component_plan([{"name": "a", "requirement_ids": ["R-1"]}], ["R-1", "R-2"])


def test_legacy_fill_heals_equal_maps_and_never_fabricates_grounding():
    healed = _legacy_arch_fill({"stack": {"language": "Python"},
                                "stack_baseline_refs": {"language": ["repo:x"]},
                                "decisions": []})
    assert healed["stack_entries"] == [
        {"key": "language", "value": "Python", "baseline_refs": ["repo:x"],
         "migration_adr_id": None}]
    # the phi4 failure shape: mismatched maps heal with EMPTY refs - a
    # migration marker, never invented grounding
    healed = _legacy_arch_fill({"stack": {"language": "Python", "stack_baseline_refs": "x"},
                                "stack_baseline_refs": {"language": ["repo:x"]},
                                "decisions": []})
    assert all(e["baseline_refs"] == [] for e in healed["stack_entries"])


class _MicroModel:
    """Answers each micro-call from its schema; can be told to crash or to
    emit a departure, exercising assembly wiring + per-slot resume."""

    DEPART_VALUES = {"database": ("PostgreSQL", "Adopt PostgreSQL"),
                     "ci_tool": ("Jenkins", "Adopt Jenkins for ci_tool"),
                     "encryption": ("AES-256", "Adopt AES-256 for encryption"),
                     "framework": ("LangGraph", "Adopt LangGraph for framework"),
                     "sso_integration": ("Keycloak", "Adopt Keycloak for sso_integration"),
                     "data_visualization_library": ("D3.js", "Adopt D3.js for data_visualization_library")}

    def __init__(self, stage, schema, calls, crash_on=None, depart=False,
                 depart_key="database", adr_slugs=None, plan_keys=None):
        self.stage, self.schema, self.calls = stage, schema, calls
        self.crash_on, self.depart, self.depart_key = crash_on, depart, depart_key
        self.adr_slugs = adr_slugs
        self.plan_keys = plan_keys

    def stream(self, messages):
        self.calls.append(self.schema.get("properties", self.schema.get("type")))
        props = set(self.schema.get("properties", {}))
        if props == {"stack_keys"}:
            out = {"stack_keys": self.plan_keys or (["language", self.depart_key] if self.depart
                                  else ["language", "database"])}
        elif props == {"technology", "off_list_justification", "baseline_refs"}:
            prompt = "\n".join(str(m.content) for m in messages)
            if (f"THE STACK KEY TO DECIDE: {self.depart_key}" in prompt and self.depart):
                out = {"technology": self.DEPART_VALUES[self.depart_key][0], "off_list_justification": None, "baseline_refs": []}
            elif any(f"THE STACK KEY TO DECIDE: {k}" in m.content for k in ("database", "persistence")
                     for m in messages):
                # kind-correct default: the slot-kind gate (P3) rejects a language
                # value for a database slot
                out = {"technology": "PostgreSQL", "off_list_justification": None, "baseline_refs": ["persistence:spec-generator:1"]}
            else:
                out = {"technology": "python", "off_list_justification": None, "baseline_refs": ["runtime:spec-generator:1"]}
        elif props == {"responsibility"}:
            out = {"responsibility": "serve"}
        elif props == {"position"}:
            out = {"position": "fast"}
        elif "driver" in props:
            prompt = "\n".join(str(m.content) for m in messages)
            migration = '"migration": true' in prompt
            out = {"title": self.DEPART_VALUES[self.depart_key][1] if migration else "t",
                   "decision": "Adopt the replacement", "rationale": "shared transactional state",
                   "driver": "simplicity",
                   "options_considered": [], "baseline_refs": [],
                   "evidence": ["R-9 requires shared state"] if migration else [],
                   "affected_repositories": ["spec-generator"] if migration else [],
                   "executes_concept_ids": (self.schema["properties"]["executes_concept_ids"]["items"].get("enum") or []) if 'executes_concept_ids' in self.schema.get("properties",{}) else []}
        elif self.stage != "architecture":
            out = PAYLOADS[self.stage]
        elif self.schema.get("type") == "array":
            item_props = set(self.schema["items"].get("properties", {}))
            if "slug" in item_props:
                out = self.adr_slugs if self.adr_slugs is not None else [{"slug": "t"}]
            else:
                out = [{"name": "api", "requirement_ids": ["R-1"],
                        "baseline_refs": ["repo:spec-generator"], "migration": False}]
        else:
            raise AssertionError(f"unexpected architecture schema: {self.schema}")
        if self.crash_on and props == self.crash_on:
            raise RuntimeError("simulated crash")
        yield AIMessage(content=json.dumps(out))


def _run(monkeypatch, calls, partials=None, **kw):
    # conftest's SPEC_GENERATOR_BULK=1 stands: srs/items/tests take their bulk
    # paths (PAYLOADS-shaped); architecture is ALWAYS the micro-pipeline - it
    # has no bulk path, that was the whole single-giant-call failure mode.
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: _MicroModel(stage, k.get("schema") or {}, calls, **kw))
    g = build_graph(UsageLog("s"), partials=partials)
    return g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
                    {"configurable": {"thread_id": "micro"}})


def test_departure_is_wired_to_its_authorizing_adr_in_code(monkeypatch):
    r = _run(monkeypatch, [], depart=True)
    arch = r["architecture"]
    db = next(e for e in arch["stack_entries"] if e["key"] == "database")
    assert db["baseline_refs"] == [] and db["migration_adr_id"] == db["decision_id"]
    owner = next(d for d in arch["decisions"] if d["decision_id"] == db["decision_id"])
    assert owner["migration"] is True
    assert owner["affected_repositories"] == ["spec-generator"]


def test_ci_tool_departure_is_authorized_without_model_cooperation(monkeypatch):
    # the exact 2026-09-12 live failure, end to end: stack:ci_tool departed
    # from the baseline and the model's ADR plan authorized NOTHING (its
    # ci_tool-absence slot had migration:false). The pipeline now generates
    # the migration slot itself; the model only authors the body prose.
    r = _run(monkeypatch, [], depart=True, depart_key="ci_tool",
             adr_slugs=[{"slug": "ci_tool-absence"}])
    arch = r["architecture"]
    ci = next(e for e in arch["stack_entries"] if e["key"] == "ci_tool")
    assert ci["baseline_refs"] == [] and ci["migration_adr_id"] == ci["decision_id"]
    owner = next(d for d in arch["decisions"] if d["decision_id"] == ci["decision_id"])
    assert owner["migration"] is True
    assert owner["title"] == "Adopt Jenkins for ci_tool"  # names the departure
    assert owner["evidence"] and owner["affected_repositories"] == ["spec-generator"]
    # the model's optional slot survives as a plain non-migration ADR
    optional = next(d for d in arch["decisions"] if d["title"] == "t")
    assert optional["migration"] is False


def test_per_slot_resume_skips_completed_entries(monkeypatch, tmp_path):
    partials = _make_partials(tmp_path)
    with pytest.raises(RuntimeError):
        _run(monkeypatch, [], partials=partials, crash_on={"technology", "off_list_justification", "baseline_refs"})
    saved = partials.load("architecture")
    assert saved["stack_keys"] == ["language", "database"]
    assert saved.get("stack_entries", {}) == {}  # first entry call crashed before saving
    calls = []
    r = _run(monkeypatch, calls, partials=partials)
    entry_calls = [c for c in calls if isinstance(c, dict) and set(c) == {"technology", "off_list_justification", "baseline_refs"}]
    assert len(entry_calls) == 2  # both entries ran (nothing was saved)
    assert partials.load("architecture") is None  # cleared on completion
    # now crash AFTER the first entry completed: rerun skips it
    calls = []

    class CrashSecond(_MicroModel):
        def stream(self, messages):
            if "THE STACK KEY TO DECIDE: database" in "\n".join(str(m.content) for m in messages):
                raise RuntimeError("crash on second entry")
            yield from super().stream(messages)

    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: CrashSecond(stage, k.get("schema") or {}, calls))
    partials2 = _make_partials(tmp_path / "b")
    g = build_graph(UsageLog("s"), partials=partials2)
    with pytest.raises(RuntimeError):
        g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
                 {"configurable": {"thread_id": "micro2"}})
    saved = partials2.load("architecture")
    assert list(saved["stack_entries"]) == ["language"]  # first entry preserved
    calls = []
    _run(monkeypatch, calls, partials=partials2)
    entry_calls = [c for c in calls if isinstance(c, dict) and set(c) == {"technology", "off_list_justification", "baseline_refs"}]
    assert len(entry_calls) == 1  # only the crashed entry re-ran


def test_raw_capture_is_opt_in_and_byte_exact(monkeypatch, tmp_path):
    log = tmp_path / "llm-calls.jsonl"
    monkeypatch.setenv("VMODEL_LLM_LOG", str(log))
    monkeypatch.delenv("SPEC_GENERATOR_LOG_RAW", raising=False)
    _run(monkeypatch, [])
    assert not any("llm_call_raw" in l for l in log.read_text().splitlines())
    monkeypatch.setenv("SPEC_GENERATOR_LOG_RAW", "1")
    _run(monkeypatch, [])
    raws = [json.loads(l) for l in log.read_text().splitlines() if '"llm_call_raw"' in l]
    assert raws and all(r["raw_chars"] == len(r["raw"]) for r in raws)
    plan = next(r for r in raws if json.loads(r["raw"]).get("stack_keys"))
    assert json.loads(plan["raw"]) == {"stack_keys": ["language", "database"]}


def test_invented_entry_value_retries_only_that_slot(monkeypatch):
    # the exact 2026-09-12 Auth0 acceptance case: a value invented while
    # citing baseline refs. The per-slot check rejects it at the micro-call,
    # the semantic retry re-asks just that entry, and the stage completes.
    counters = {"db": 0}

    class Auth0First(_MicroModel):
        def stream(self, messages):
            if any("THE STACK KEY TO DECIDE: database" in m.content for m in messages):
                counters["db"] += 1
                if counters["db"] == 1:
                    yield AIMessage(content=json.dumps(
                        {"technology": "Auth0", "off_list_justification": None, "baseline_refs": ["repo:spec-generator"]}))
                    return
            yield from super().stream(messages)

    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: Auth0First(stage, k.get("schema") or {}, []))
    r = build_graph(UsageLog("s")).invoke(
        {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
        {"configurable": {"thread_id": "auth0"}})
    assert counters["db"] == 2  # rejected once, retried with feedback, accepted
    db = next(e for e in r["architecture"]["stack_entries"] if e["key"] == "database")
    assert db["value"] == "PostgreSQL" and db["baseline_refs"] == ["persistence:spec-generator:1"]


def test_persisted_entry_failing_current_contract_is_reasked(monkeypatch, tmp_path):
    # the live resume situation: a stale Auth0 entry persisted before the
    # per-slot check existed is discarded and re-asked - never trusted, and
    # the GOOD persisted entries are not re-run.
    partials = _make_partials(tmp_path)
    partials.save("architecture", {
        "stack_keys": ["language", "database"],
        "stack_entries": {"language": {"key": "language", "value": "ts",
                                       "baseline_refs": ["repo:spec-generator"]},
                          "database": {"key": "database", "value": "Auth0",
                                       "baseline_refs": ["repo:spec-generator"]}}})
    calls = []
    r = _run(monkeypatch, calls, partials=partials)
    entry_calls = [c for c in calls if isinstance(c, dict) and set(c) == {"technology", "off_list_justification", "baseline_refs"}]
    assert len(entry_calls) == 1  # only the bad slot was re-asked
    db = next(e for e in r["architecture"]["stack_entries"] if e["key"] == "database")
    assert db["value"] == "PostgreSQL" and db["baseline_refs"] == ["persistence:spec-generator:1"]


def test_gate_failure_is_human_readable_not_a_traceback(monkeypatch):
    # a slot problem the per-slot checks cannot see (here: an invented
    # component baseline ref, provider-side enum bypassed) still fails at the
    # whole-architecture gate - and surfaces as Error/Likely cause/Next,
    # never as a raw ModelStructureError traceback.
    from spec_generator.errors import UserFacingError

    class BadCompRef(_MicroModel):
        def stream(self, messages):
            if (self.schema.get("type") == "array"
                    and "name" in self.schema["items"].get("properties", {})):
                yield AIMessage(content=json.dumps(
                    [{"name": "api", "requirement_ids": ["R-1"],
                      "baseline_refs": ["repo:nope"], "migration": False}]))
                return
            yield from super().stream(messages)

    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: BadCompRef(stage, k.get("schema") or {}, []))
    with pytest.raises(UserFacingError) as ei:
        build_graph(UsageLog("s")).invoke(
            {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
            {"configurable": {"thread_id": "gate"}})
    rendered = ei.value.render()
    assert rendered.startswith("Error: stage 'architecture'")
    assert "Likely cause:" in rendered and "repo:nope" in rendered
    assert "Next:" in rendered


def test_empty_answer_gets_targeted_proposal_feedback(monkeypatch):
    # the exact 2026-09-12 sso_integration failure: {"technology": "", "off_list_justification": None, "baseline_refs": []}
    # on BOTH attempts. The blank is now rejected with feedback that NAMES the
    # legal move - propose a concrete technology; it becomes a migration
    # automatically, with the pipeline generating the authorizing ADR.
    seen = {"n": 0, "retry_feedback": None}

    class BlankFirst(_MicroModel):
        def stream(self, messages):
            if any("THE STACK KEY TO DECIDE: sso_integration" in m.content for m in messages):
                seen["n"] += 1
                if seen["n"] == 1:
                    yield AIMessage(content=json.dumps({"technology": "", "off_list_justification": None, "baseline_refs": []}))
                    return
                seen["retry_feedback"] = "\n".join(str(m.content) for m in messages)
                yield AIMessage(content=json.dumps({"technology": "Keycloak", "off_list_justification": None, "baseline_refs": []}))
                return
            yield from super().stream(messages)

    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: BlankFirst(stage, k.get("schema") or {}, [],
                                                      depart=True, depart_key="sso_integration"))
    r = build_graph(UsageLog("s")).invoke(
        {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
        {"configurable": {"thread_id": "blank"}})
    assert seen["n"] == 2  # blank rejected, retried once with feedback, accepted
    assert "empty value decides nothing" in seen["retry_feedback"]
    assert "proposed migration" in seen["retry_feedback"]
    sso = next(e for e in r["architecture"]["stack_entries"] if e["key"] == "sso_integration")
    assert sso["value"] == "Keycloak" and sso["baseline_refs"] == []
    assert sso["migration_adr_id"] == sso["decision_id"]  # pipeline-generated migration slot


def test_from_architecture_resume_merges_baseline_flag_into_old_checkpoint(monkeypatch, tmp_path):
    # the 2026-09-12 --from architecture crash (KeyError 'system_baseline'):
    # the original run's checkpoint predates the baseline feature and the CLI
    # never injected the --system-baseline flag into resumed state. Now: the
    # stage fails human-readably when absent, and the CLI merges the flag
    # value into the resumed checkpoint so the regen completes.
    import sqlite3
    import sys

    from langgraph.checkpoint.sqlite import SqliteSaver

    from spec_generator import cli
    from spec_generator.errors import UserFacingError
    from spec_generator.graph import checkpoint_serde

    intent = {**INTENT, "contract_version": "0.16.0"}
    intent_path = tmp_path / "intent.json"
    intent_path.write_text(json.dumps(intent))
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(SYSTEM_BASELINE))
    outdir = tmp_path / "artifacts"
    outdir.mkdir()
    sid = INTENT["session_id"]
    monkeypatch.setenv("SPEC_GENERATOR_FROM", "")
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: _MicroModel(stage, k.get("schema") or {}, []))

    # 1. an OLD checkpoint: state without system_baseline (pre-baseline
    # feature). needs+srs complete; architecture stops human-readably.
    conn = sqlite3.connect(str(outdir / "checkpoints.sqlite"), check_same_thread=False)
    g = build_graph(UsageLog(sid), checkpointer=SqliteSaver(conn, serde=checkpoint_serde()))
    with pytest.raises(UserFacingError, match="no system baseline"):
        g.invoke({"intent_model": dict(intent)}, {"configurable": {"thread_id": sid}})

    # 2. the fixed CLI path: --from architecture + --system-baseline merges the
    # flag into the resumed checkpoint and the full regen completes.
    monkeypatch.setattr(sys, "argv", ["prog", str(intent_path),
                                      "--system-baseline", str(baseline_path),
                                      "-o", str(outdir), "--from", "architecture"])
    cli.main()
    arch = json.loads((outdir / "architecture.json").read_text())
    assert arch["stack_entries"]  # regenerated under the current code
    assert arch["baseline_id"] == SYSTEM_BASELINE["baseline_id"]
    assert (outdir / "manifest.json").exists()  # downstream bundle assembled


def test_review_resume_after_accepted_srs_merges_baseline_without_rerunning_srs(monkeypatch, tmp_path):
    import sqlite3
    import sys
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command
    from spec_generator import cli
    from spec_generator.graph import checkpoint_serde

    intent={**INTENT,"contract_version":"0.16.0"}
    ip=tmp_path/"intent.json";ip.write_text(json.dumps(intent))
    bp=tmp_path/"baseline.json";bp.write_text(json.dumps(SYSTEM_BASELINE))
    out=tmp_path/"artifacts";out.mkdir();sid=intent["session_id"]
    monkeypatch.setattr(model_config,"build_chat_model",lambda stage,**k:_MicroModel(stage,k.get("schema") or {},[]))
    conn=sqlite3.connect(str(out/"checkpoints.sqlite"),check_same_thread=False)
    old=build_graph(UsageLog(sid),checkpointer=SqliteSaver(conn,serde=checkpoint_serde()),review=True)
    old.invoke({"intent_model":intent},{"configurable":{"thread_id":sid}})
    # accept the already-reviewed SRS; old architecture then fails because its
    # checkpoint has no baseline, reproducing the production lineage.
    with pytest.raises(UserFacingError,match="no system baseline"):
        old.invoke(Command(resume={"action":"accept"}),{"configurable":{"thread_id":sid}})
    calls=[]
    model=_MicroModel("architecture",{},[])
    def factory(stage,**k):
        calls.append(stage);return _MicroModel(stage,k.get("schema") or {},[])
    monkeypatch.setattr(model_config,"build_chat_model",factory)
    monkeypatch.setattr("builtins.input",lambda prompt:"a")
    monkeypatch.setattr(sys,"argv",["prog",str(ip),"--system-baseline",str(bp),"-o",str(out),"--review"])
    cli.main()
    assert "needs" not in calls and "srs" not in calls
    assert json.loads((out/"architecture.json").read_text())["baseline_id"]==SYSTEM_BASELINE["baseline_id"]


def test_verbatim_fact_text_is_grounded_not_rejected(monkeypatch):
    # the exact 2026-09-12 live rejection: the model COPIED the baseline fact
    # 'python -m spec_generator.cli' for stack:module_entry_points and the old
    # check rejected it - the vocab excluded entry-point/artifact facts. The
    # vocab now covers every citable fact class.
    from spec_generator.graph import _baseline_vocab, _value_grounded
    from vmodel_contracts import SystemBaseline

    vocab = _baseline_vocab(SystemBaseline.model_validate(SYSTEM_BASELINE))
    assert _value_grounded("python -m spec_generator.cli", vocab)  # verbatim fact
    assert _value_grounded("spec_generator.cli python", vocab)     # composed of fact tokens
    assert _value_grounded("ts", vocab)                            # language fact
    assert not _value_grounded("Auth0", vocab)                     # invented: still closed
    assert not _value_grounded("python -m auth0.cli", vocab)       # composed fabrication

    class LiveAnswer(_MicroModel):
        def stream(self, messages):
            props = set(self.schema.get("properties", {}))
            if props == {"stack_keys"}:  # the live run's plan included module_entry_points
                yield AIMessage(content=json.dumps({"stack_keys": ["language", "module_entry_points"]}))
                return
            if any("THE STACK KEY TO DECIDE: module_entry_points" in m.content for m in messages):
                yield AIMessage(content=json.dumps(
                    {"value": "python -m spec_generator.cli",
                     "baseline_refs": ["module:spec-generator:1"]}))
                return
            yield from super().stream(messages)

    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: LiveAnswer(stage, k.get("schema") or {}, []))
    r = build_graph(UsageLog("s")).invoke(
        {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
        {"configurable": {"thread_id": "verbatim"}})
    db = next(e for e in r["architecture"]["stack_entries"] if e["key"] == "module_entry_points")
    assert db["value"] == "python -m spec_generator.cli"
    assert db["baseline_refs"] == ["module:spec-generator:1"]
    assert db["migration_adr_id"] is None  # grounded, not a departure


def test_entry_point_shape_cannot_answer_a_technology_slot():
    # deterministic P3 layer 1 (2026-09-12 regen: 'python -m spec_generator.cli'
    # stuffed into framework + ci): shape rules, no model judgment needed
    from spec_generator.graph import _entry_point_shaped, _entry_point_slot
    assert _entry_point_shaped("python -m spec_generator.cli")
    assert _entry_point_shaped("Python -m spec_generator.cli")   # capital-P paraphrase
    assert _entry_point_shaped("scripts/batch-specs.sh")
    assert _entry_point_shaped("delivery_api.app:app via uvicorn")
    assert not _entry_point_shaped("LangGraph")
    assert not _entry_point_shaped("PostgreSQL 16")
    assert _entry_point_slot("module_entry_points")
    assert _entry_point_slot("package_entry_points")
    assert not _entry_point_slot("framework")

    class EntryPointAnswer(_MicroModel):
        def stream(self, messages):
            props = set(self.schema.get("properties", {}))
            if props == {"technology", "off_list_justification", "baseline_refs"} and any(
                    "THE STACK KEY TO DECIDE: framework" in m.content for m in messages):
                if any("cannot answer this slot" in m.content for m in messages):
                    yield AIMessage(content=json.dumps({"technology": "LangGraph", "off_list_justification": None, "baseline_refs": []}))
                else:
                    yield AIMessage(content=json.dumps(
                        {"value": "python -m spec_generator.cli",
                         "baseline_refs": ["module:spec-generator:1"]}))
                return
            yield from super().stream(messages)

    calls = []
    _run(monkeypatch := __import__("pytest").MonkeyPatch(), calls)
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: EntryPointAnswer(stage, k.get("schema") or {}, calls,
                                                            plan_keys=["language", "framework"],
                                                            depart_key="framework", depart=True))
    r = build_graph(UsageLog("s")).invoke(
        {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
        {"configurable": {"thread_id": "shape"}})
    fw = next(e for e in r["architecture"]["stack_entries"] if e["key"] == "framework")
    assert fw["value"] == "LangGraph"          # the command never lands in a framework slot
    assert fw["migration_adr_id"] is not None  # a departure, honestly labeled
    monkeypatch.undo()


def test_wrong_kind_value_rejected_deterministically(monkeypatch):
    # P3 layer 2, deterministic (the regen's encryption='Pydantic'): a value in a
    # KNOWN lexicon of the wrong kind cannot answer the slot. No model call.
    class PydanticAnswer(_MicroModel):
        def stream(self, messages):
            props = set(self.schema.get("properties", {}))
            if props == {"technology", "off_list_justification", "baseline_refs"} and any(
                    "THE STACK KEY TO DECIDE: encryption" in m.content for m in messages):
                if any("does not answer this" in m.content for m in messages):
                    yield AIMessage(content=json.dumps({"technology": "AES-256", "off_list_justification": None, "baseline_refs": []}))
                else:
                    yield AIMessage(content=json.dumps(
                        {"technology": "python", "off_list_justification": None, "baseline_refs": ["runtime:spec-generator:1"]}))
                return
            yield from super().stream(messages)

    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: PydanticAnswer(
                            stage, k.get("schema") or {}, [],
                            plan_keys=["language", "encryption"], depart_key="encryption", depart=True))
    r = build_graph(UsageLog("s")).invoke(
        {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
        {"configurable": {"thread_id": "wrongkind"}})
    enc = next(e for e in r["architecture"]["stack_entries"] if e["key"] == "encryption")
    assert enc["value"] == "AES-256"           # the wrong-kind candidate was rejected
    assert enc["migration_adr_id"] is not None


def test_founder_constraint_driver_needs_a_recorded_preference(monkeypatch):
    # P1 (the regen's ADR-1 'Auth0'): founder_constraint with NO founder tech
    # preferences on record is impossible, not merely unproven - per-ADR retry
    class OverclaimingAdr(_MicroModel):
        def stream(self, messages):
            props = set(self.schema.get("properties", {}))
            if "driver" in props:
                migration = any('"migration": true' in m.content for m in messages)
                driver = "founder_constraint"
                rat = "the founder mandated it"
                if any("driver founder_constraint requires" in m.content for m in messages):
                    driver, rat = "ecosystem_maturity", "mature ecosystem"
                yield AIMessage(content=json.dumps(
                    {"title": "Adopt PostgreSQL", "decision": "Use PostgreSQL",
                     "rationale": rat, "driver": driver,
                     "options_considered": [], "baseline_refs": [],
                     "evidence": ["R-9 requires shared state"] if migration else [],
                     "affected_repositories": ["spec-generator"] if migration else [],
                   "executes_concept_ids": (self.schema["properties"]["executes_concept_ids"]["items"].get("enum") or []) if 'executes_concept_ids' in self.schema.get("properties",{}) else []}))
                return
            yield from super().stream(messages)

    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: OverclaimingAdr(stage, k.get("schema") or {}, [], depart=True))
    r = build_graph(UsageLog("s")).invoke(
        {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},  # INTENT: no tech_preferences
        {"configurable": {"thread_id": "overclaim"}})
    drivers = [d["driver"] for d in r["architecture"]["decisions"]]
    assert "founder_constraint" not in drivers
    assert "ecosystem_maturity" in drivers


def test_mandate_language_under_another_driver_is_rejected(monkeypatch):
    # P1 consistency: claiming the founder's mandate in prose while the driver
    # says something else is the same lie with a quieter label
    class QuietlyOverclaimingAdr(_MicroModel):
        def stream(self, messages):
            props = set(self.schema.get("properties", {}))
            if "driver" in props:
                migration = any('"migration": true' in m.content for m in messages)
                if any("claims a founder mandate" in m.content for m in messages):
                    rat = "mature ecosystem"
                else:
                    rat = "chosen because it is mandated by the founder"
                yield AIMessage(content=json.dumps(
                    {"title": "Adopt PostgreSQL", "decision": "Use PostgreSQL",
                     "rationale": rat, "driver": "simplicity",
                     "options_considered": [], "baseline_refs": [],
                     "evidence": ["R-9 requires shared state"] if migration else [],
                     "affected_repositories": ["spec-generator"] if migration else [],
                   "executes_concept_ids": (self.schema["properties"]["executes_concept_ids"]["items"].get("enum") or []) if 'executes_concept_ids' in self.schema.get("properties",{}) else []}))
                return
            yield from super().stream(messages)

    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: QuietlyOverclaimingAdr(stage, k.get("schema") or {}, [], depart=True))
    r = build_graph(UsageLog("s")).invoke(
        {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
        {"configurable": {"thread_id": "quiet-overclaim"}})
    assert all("founder" not in d["rationale"] for d in r["architecture"]["decisions"])


def test_js_suffixed_technology_names_are_not_entry_points():
    # 2026-09-13 HIS LIVE RUN died here: greenfield stack slot
    # data_visualization_library='D3.js' rejected as "an entry point" - the
    # bare '.js' tripped the script-file shape check. Script detection now
    # requires invocation structure (a slash, 'python -m', module:attr), and
    # known-tech lexicon membership takes precedence over shape.
    from spec_generator.graph import (_entry_point_shaped, _known_tech,
                                      _wrong_kind)
    for name in ("D3.js", "Vue.js", "Node.js", "Chart.js", "Next.js",
                 "Express.js", "Three.js", "Nuxt.js", "TensorFlow.js"):
        assert not _entry_point_shaped(name), name
    # real entry points still caught
    assert _entry_point_shaped("scripts/batch-specs.sh")
    assert _entry_point_shaped("./deploy.sh")
    assert _entry_point_shaped("python -m spec_generator.cli")
    assert _entry_point_shaped("python3 run.py")
    assert _entry_point_shaped("delivery_api.app:app via uvicorn")
    # lexicon precedence + the new visualization kind
    assert _known_tech("D3.js") and _known_tech("Chart.js")
    assert _wrong_kind("data_visualization_library", "D3.js") is None
    assert _wrong_kind("data_visualization_library", "FastAPI") == "framework"


def test_d3js_accepted_for_a_visualization_slot_end_to_end(monkeypatch):
    # the exact answer his run died on: D3.js for data_visualization_library,
    # declared as a departure (empty refs) - must land with a migration ADR,
    # not a contract failure.
    class D3Answer(_MicroModel):
        def stream(self, messages):
            props = set(self.schema.get("properties", {}))
            if props == {"technology", "off_list_justification", "baseline_refs"} and any(
                    "THE STACK KEY TO DECIDE: data_visualization_library" in m.content
                    for m in messages):
                yield AIMessage(content=json.dumps({"technology": "D3.js", "off_list_justification": None, "baseline_refs": []}))
                return
            yield from super().stream(messages)

    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: D3Answer(
                            stage, k.get("schema") or {}, [],
                            plan_keys=["language", "data_visualization_library"],
                            depart_key="data_visualization_library", depart=True))
    r = build_graph(UsageLog("s")).invoke(
        {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
        {"configurable": {"thread_id": "d3"}})
    viz = next(e for e in r["architecture"]["stack_entries"]
               if e["key"] == "data_visualization_library")
    assert viz["value"] == "D3.js"
    assert viz["migration_adr_id"] is not None


GREENFIELD_BASELINE = {"contract_version": "0.16.0", "baseline_id": "gf-test",
                       "system_name": "New product", "mode": "greenfield",
                       "repositories": [], "immutable_decisions": []}


class _GreenfieldModel(_MicroModel):
    """Greenfield answers: empty refs everywhere, and - the poison from his
    2026-09-13 live run - components planned with migration=true because the
    old prompt said 'new component = migration'."""
    def stream(self, messages):
        props = set(self.schema.get("properties", {}))
        if self.stage == "architecture" and props == {"technology", "off_list_justification", "baseline_refs"}:
            yield AIMessage(content=json.dumps({"technology": "Swift", "off_list_justification": None, "baseline_refs": []}))
            return
        if self.stage == "architecture" and self.schema.get("type") == "array":
            item_props = set(self.schema["items"].get("properties", {}))
            if "slug" in item_props:
                yield AIMessage(content=json.dumps([{"slug": "offline-first"}]))
            else:
                yield AIMessage(content=json.dumps(
                    [{"name": "api", "requirement_ids": ["R-1"],
                      "baseline_refs": [], "migration": True}]))  # the old prompt's teaching
            return
        yield from super().stream(messages)


def _run_greenfield(monkeypatch, calls, partials=None):
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: _GreenfieldModel(
                            stage, k.get("schema") or {}, calls, plan_keys=["language"]))
    return build_graph(UsageLog("s"), partials=partials).invoke(
        {"intent_model": dict(INTENT), "system_baseline": GREENFIELD_BASELINE},
        {"configurable": {"thread_id": "gf"}})


def test_greenfield_component_plan_migration_is_code_owned():
    # the plan prompt taught 'new component = migration=true'; on an empty
    # baseline that makes the whole-arch gate reject a CORRECT run
    # ('greenfield architecture is new design, not a migration'). Code owns
    # the flag: greenfield components are always new design.
    from spec_generator.graph import _validate_component_plan
    slots = _validate_component_plan(
        [{"name": "api", "requirement_ids": ["R-1"], "baseline_refs": [], "migration": True}],
        ["R-1"], existing=False)
    assert slots[0]["migration"] is False
    # existing mode keeps the model's flag untouched
    slots = _validate_component_plan(
        [{"name": "api", "requirement_ids": ["R-1"], "baseline_refs": ["repo:x"], "migration": True}],
        ["R-1"], existing=True)
    assert slots[0]["migration"] is True


def test_greenfield_full_architecture_passes_the_gate(monkeypatch):
    r = _run_greenfield(monkeypatch, [])
    arch = r["architecture"]
    assert all(not c["migration"] and c["migration_adr_title"] is None
               for c in arch["components"])
    assert all(not d["migration"] for d in arch["decisions"])  # initial decisions, no migration framing
    assert all(not e["baseline_refs"] and e["migration_adr_id"] is None
               for e in arch["stack_entries"])


def test_greenfield_resume_normalizes_persisted_component_slots(monkeypatch, tmp_path):
    # his actual failure: slots persisted under the OLD prompt (migration=true)
    # made the resume re-evaluate the gate and die with ZERO model calls. The
    # persisted slots are normalized in place - no re-plan, no re-ask.
    partials = _make_partials(tmp_path)
    partials.save("architecture", {
        "stack_entries": {"language": {"key": "language", "value": "Swift",
                                       "baseline_refs": []}},
        "component_slots": [{"name": "api", "requirement_ids": ["R-1"],
                             "baseline_refs": [], "migration": True}],
    })
    calls = []
    r = _run_greenfield(monkeypatch, calls, partials=partials)
    assert all(not c["migration"] for c in r["architecture"]["components"])
    # the component plan was NOT re-asked (persisted slots kept, normalized)
    assert not any(c == {"stack_keys"} for c in calls)


def test_brownfield_still_requires_cite_or_migrate(monkeypatch):
    # unchanged: on an existing baseline, a component with no refs and no
    # migration is still rejected at the gate (existing tests pin the happy
    # path; this pins that the greenfield relaxation did not leak)
    from spec_generator.graph import _check_baseline_reconciliation
    from vmodel_contracts import ArchitectureOverview
    from conftest import SYSTEM_BASELINE
    arch = ArchitectureOverview.model_validate({
        "contract_version": "0.16.0", "session_id": "s", "baseline_id": "test-baseline",
        "stack_entries": [{"key": "language", "value": "ts", "technology_id": "tech:typescript", "decision_id": "ADR-1",
                           "baseline_refs": ["repo:spec-generator"], "migration_adr_id": None}],
        "components": [{"name": "api", "responsibility": "serve", "requirement_ids": ["R-1"],
                        "baseline_refs": [], "migration": False, "migration_adr_title": None}],
        "decisions": [{"decision_id":"ADR-1","title":"Language","decision":"TypeScript","rationale":"fit","driver":"simplicity","options_considered":[],"executes_concept_ids":[],"technology_selections":[{"slot_id":"slot:language","technology_id":"tech:typescript","display_name":"ts","baseline_refs":["repo:spec-generator"]}]}], "non_functionals": {}})
    from spec_generator.skeleton import ModelStructureError
    from vmodel_contracts import SystemBaseline
    try:
        _check_baseline_reconciliation(arch, SystemBaseline.model_validate(SYSTEM_BASELINE))
        assert False, "brownfield component without refs or migration must be rejected"
    except ModelStructureError as e:
        assert "requires baseline refs" in str(e)


def test_fresh_review_cli_carries_baseline_and_conops_into_architecture(monkeypatch,tmp_path,capsys):
    """Real CLI boundary regression: --review used to discard both fields."""
    import sys
    from spec_generator import cli
    intent={**INTENT,"contract_version":"0.16.0"}
    ip=tmp_path/"intent.json";ip.write_text(json.dumps(intent))
    bp=tmp_path/"baseline.json";bp.write_text(json.dumps(SYSTEM_BASELINE))
    # A minimal valid ConOps fixture from the contract, tied to this session.
    from vmodel_contracts import ConOpsDocument
    cp=tmp_path/"conops.json"
    cp.write_text(ConOpsDocument(session_id=intent["session_id"],stakeholder_context=[{"stakeholder":"u","role":"user","goals":["v"],"operating_environment":"web","source_turns":[0]}],operational_scenarios=[{"scenario_id":"OS-1","name":"use","trigger":"need","actors":["u"],"operational_flow":["act"],"success_outcome":"v","exceptions":[],"source_turns":[0]}],concept_options=[{"name":"ts","summary":"ts","advantages":["simple"],"disadvantages":["none"]},{"name":"native","summary":"native","advantages":["fast"],"disadvantages":["install"]}],decision={"selected_concept":"ts","rationale":"fit","founder_confirmed":True,"source_turns":[0]}).model_dump_json())
    seen=[]
    class Inspect(_MicroModel):
        def stream(self,messages):
            if self.stage=="architecture":
                text=" ".join(str(m.content) for m in messages)
                seen.append(SYSTEM_BASELINE["baseline_id"] in text)
            yield from super().stream(messages)
    monkeypatch.setattr(model_config,"build_chat_model",lambda stage,**k:Inspect(stage,k.get("schema") or {},[]))
    answers=iter(["a","a"]);monkeypatch.setattr("builtins.input",lambda prompt:next(answers))
    out=tmp_path/"fresh"
    monkeypatch.setattr(sys,"argv",["prog",str(ip),"--conops",str(cp),"--system-baseline",str(bp),"--review","-o",str(out),"--to","architecture"])
    cli.main()
    assert seen and all(seen)
    assert json.loads((out/"system-baseline.json").read_text())["baseline_id"]==SYSTEM_BASELINE["baseline_id"]


def _pwa_conops():
    return {"session_id":"s", "stakeholder_context":[], "operational_scenarios":[],
            "concept_options":[{"name":"Progressive Web Application (PWA)","summary":"web","advantages":["installable"],"disadvantages":["limits"]}],
            "decision":{"selected_concept":"Progressive Web Application (PWA)","rationale":"chosen","founder_confirmed":True,"source_turns":[1]}}


def test_conops_execution_gate_repairs_by_redrafting_adr(monkeypatch):
    attempts={"plans":0}
    class OmitsThenRepairs(_MicroModel):
        def stream(self,messages):
            props=set(self.schema.get("properties",{}))
            if self.schema.get("type")=="array" and "slug" in self.schema.get("items",{}).get("properties",{}):
                attempts["plans"]+=1
            if "driver" in props and attempts["plans"] >= 2:
                yield AIMessage(content=json.dumps({"title":"Execute PWA concept","decision":"Build the Progressive Web Application (PWA)","rationale":"Matches the confirmed operating concept","driver":"simplicity","options_considered":[],"baseline_refs":[],"evidence":[],"affected_repositories":[],"executes_concept_ids":["concept:pwa"]}));return
            yield from super().stream(messages)
    monkeypatch.setattr(model_config,"build_chat_model",lambda stage,**k: OmitsThenRepairs(stage,k.get("schema") or {},[]))
    result=build_graph(UsageLog("s")).invoke({"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE,"conops":_pwa_conops()},{"configurable":{"thread_id":"pwa-repair"}})
    assert attempts["plans"] >= 2
    assert any("concept:pwa" in d["executes_concept_ids"] for d in result["architecture"]["decisions"])


def test_conops_execution_slot_exhaustion_is_human_readable(monkeypatch):
    class OmitsConcept(_MicroModel):
        def stream(self, messages):
            for chunk in super().stream(messages):
                data=json.loads(chunk.content)
                if isinstance(data,dict) and "executes_concept_ids" in data:
                    data["executes_concept_ids"]=[]
                yield AIMessage(content=json.dumps(data))
    monkeypatch.setattr(model_config,"build_chat_model",lambda stage,**k: OmitsConcept(stage,k.get("schema") or {},[]))
    with pytest.raises(UserFacingError) as ei:
        build_graph(UsageLog("s")).invoke({"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE,"conops":_pwa_conops()},{"configurable":{"thread_id":"pwa-fail"}})
    rendered=ei.value.render()
    assert rendered.startswith("Error: stage 'architecture'")
    assert "model's output does not satisfy the contract" in rendered
    assert "Likely cause:" in rendered and "concept:pwa" in rendered
    assert "Next:" in rendered

def test_pwa_python_fastapi_adrs_do_not_dedupe_and_repair_reaches_body(monkeypatch):
    from spec_generator.graph import _dedupe_adr_slots
    slots=[{"decision_id":"ADR-1","slug":"adopt-pwa-concept","migration":False,"authorizes":[]},
           {"decision_id":"ADR-2","slug":"choose-python-for-backend","migration":False,"authorizes":[]},
           {"decision_id":"ADR-3","slug":"select-fastapi-framework","migration":False,"authorizes":[]}]
    def body(title,decision): return {"title":title,"decision":decision,"rationale":"fit","driver":"simplicity","options_considered":[],"baseline_refs":[],"evidence":[],"affected_repositories":[],"rejected":[]}
    bodies={"ADR-1":body("Adopt PWA concept","Implement the PWA with a Python FastAPI backend"),
            "ADR-2":body("Choose Python for backend","Use Python"),
            "ADR-3":body("Select FastAPI framework","Use FastAPI")}
    assert _dedupe_adr_slots(slots,bodies)==[]
    assert [s["slug"] for s in slots]==["adopt-pwa-concept","choose-python-for-backend","select-fastapi-framework"]


def test_conops_repair_feedback_is_in_adr_body_prompt(monkeypatch):
    seen=[]
    class CaptureRepair(_MicroModel):
        def stream(self,messages):
            props=set(self.schema.get("properties",{}));prompt="\n".join(str(m.content) for m in messages)
            if "driver" in props:
                seen.append(prompt)
                if "must declare these exact ids" in prompt:
                    yield AIMessage(content=json.dumps({"title":"Adopt PWA","decision":"Execute PWA","rationale":"confirmed","driver":"simplicity","options_considered":[],"baseline_refs":[],"evidence":[],"affected_repositories":[],"executes_concept_ids":["concept:pwa"]}));return
                yield AIMessage(content=json.dumps({"title":"Other","decision":"Other","rationale":"other","driver":"simplicity","options_considered":[],"baseline_refs":[],"evidence":[],"affected_repositories":[],"executes_concept_ids":[]}));return
            yield from super().stream(messages)
    monkeypatch.setattr(model_config,"build_chat_model",lambda stage,**k: CaptureRepair(stage,k.get("schema") or {},[]))
    result=build_graph(UsageLog("s")).invoke({"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE,"conops":_pwa_conops()},{"configurable":{"thread_id":"pwa-body-feedback"}})
    assert any("must declare these exact ids" in p for p in seen)
    assert any("concept:pwa" in d["executes_concept_ids"] for d in result["architecture"]["decisions"])

def test_advisory_subject_uses_each_fixed_slot_not_repeated_model_title():
    from spec_generator.graph import _dedupe_adr_slots
    slugs=['choose-python-for-backend','select-postgresql-database','select-fastapi-framework',
           'adopt-kafka-messaging','use-kubernetes-for-scaling','choose-firebase-auth','add-prometheus-monitoring']
    slots=[{"decision_id":f"ADR-{i}","slug":slug,"migration":False,"authorizes":[]} for i,slug in enumerate(slugs,1)]
    bodies={s['decision_id']:{"title":"Kubernetes for scalability strategy","decision":"generic","rationale":"generic","driver":"simplicity","options_considered":[],"baseline_refs":[]} for s in slots}
    assert _dedupe_adr_slots(slots,bodies)==[]
    assert [s['slug'] for s in slots]==slugs


def test_stack_is_exact_projection_and_repeated_titles_do_not_bind_identity():
    from pydantic import ValidationError
    from vmodel_contracts import ArchitectureOverview
    payload={"session_id":"s","stack_entries":[{"key":"database","value":"PostgreSQL","technology_id":"tech:postgresql","decision_id":"ADR-1"}],"components":[],"decisions":[{"decision_id":"ADR-1","title":"Repeated","decision":"PostgreSQL","rationale":"fit","driver":"simplicity","options_considered":[],"executes_concept_ids":[],"technology_selections":[{"slot_id":"slot:database","technology_id":"tech:postgresql","display_name":"PostgreSQL"}]},{"decision_id":"ADR-2","title":"Repeated","decision":"Other","rationale":"fit","driver":"simplicity","options_considered":[],"executes_concept_ids":[],"technology_selections":[]}],"non_functionals":{}}
    ArchitectureOverview.model_validate(payload)
    payload["stack_entries"][0]["decision_id"]="ADR-2"
    with pytest.raises(ValidationError,match="exact projection"):
        ArchitectureOverview.model_validate(payload)


def test_phi_blank_justification_is_absent_for_known_technology(monkeypatch):
    """Phi fills every schema field, using an empty string for not applicable."""
    class BlankJustificationForKnown(_MicroModel):
        def stream(self, messages):
            if any("THE STACK KEY TO DECIDE: language" in m.content for m in messages):
                yield AIMessage(content=json.dumps({
                    "technology": "Python",
                    "off_list_justification": "",
                    "baseline_refs": [],
                }))
                return
            yield from super().stream(messages)

    monkeypatch.setattr(
        model_config, "build_chat_model",
        lambda stage, **k: BlankJustificationForKnown(
            stage, k.get("schema") or {}, [], plan_keys=["language"]
        ),
    )
    result = build_graph(UsageLog("s")).invoke(
        {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
        {"configurable": {"thread_id": "phi-empty-justification"}},
    )
    entry = result["architecture"]["stack_entries"][0]
    assert entry["technology_id"] == "tech:python"
    owner = next(
        d for d in result["architecture"]["decisions"]
        if d["decision_id"] == entry["decision_id"]
    )
    assert owner["technology_selections"][0]["off_list_justification"] is None


def test_adr_body_strips_model_echoed_fixed_id(monkeypatch):
    """The model may echo the fixed slot id; stored title prose must not own it."""
    import inspect
    source = inspect.getsource(__import__('spec_generator.graph', fromlist=['x']))
    assert 'raw_title = str(d.get("title", "")).strip()' in source
    assert "_re.escape(slot['decision_id'])" in source


def test_phi_echo_of_all_required_adr_slugs_is_removed_without_model_retry(capsys):
    taken = ["migrate-stack-framework", "migrate-stack-notification-system",
             "migrate-stack-user-interface"]
    echoed = [{"slug": slug} for slug in taken]
    assert _validate_adr_plan(echoed, taken) == []
    err = capsys.readouterr().err
    assert "boundary=adr-slug-collision" in err
    assert "decision=remove-echoed count=3" in err


def test_self_referential_adr_body_is_reasked_locally(monkeypatch):
    seen=[]
    class SelfRefThenConcrete(_MicroModel):
        def stream(self, messages):
            prompt="\n".join(str(m.content) for m in messages)
            if "driver" in self.schema.get("properties", {}):
                seen.append(prompt)
                if "self-reference" not in prompt:
                    yield AIMessage(content=json.dumps({"title":"Consolidate ADR-1","decision":"Use ADR-1","rationale":"ADR-1 owns this","driver":"simplicity","options_considered":[],"baseline_refs":[],"evidence":[],"affected_repositories":[],"executes_concept_ids":[]})); return
                yield AIMessage(content=json.dumps({"title":"Choose Python","decision":"Use Python","rationale":"Matches the existing runtime","driver":"simplicity","options_considered":[],"baseline_refs":[],"evidence":[],"affected_repositories":[],"executes_concept_ids":[]})); return
            yield from super().stream(messages)
    monkeypatch.setattr(model_config,"build_chat_model",lambda stage,**k: SelfRefThenConcrete(stage,k.get("schema") or {},[],plan_keys=["language"]))
    result=build_graph(UsageLog("s")).invoke({"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE},{"configurable":{"thread_id":"self-ref"}})
    assert len(seen) >= 2 and "self-reference" in seen[1]
    owner = next(d for d in result["architecture"]["decisions"] if d["title"] == "Choose Python")
    assert "ADR-" not in owner["decision"]


def test_assembly_keeps_research_unresolved_and_materializes_every_nfr():
    from spec_generator.graph import _assemble_architecture
    merged=_assemble_architecture(session_id="s",baseline_id="b",entries={
      "language":{"key":"language","value":"Python","technology_id":"tech:python","baseline_refs":[]},
      "hosting":{"key":"hosting","status":"research_required","research_brief":"Compare current regional terms"}},
      comp_slots=[],comp_bodies={},adr_slots=[{"decision_id":"ADR-1","slug":"language","migration":False,"authorizes":["stack:language"]}],
      adr_bodies={"ADR-1":{"title":"Choose Python","decision":"Use Python","rationale":"fit","driver":"simplicity","options_considered":[],"baseline_refs":[],"evidence":[],"affected_repositories":[],"executes_concept_ids":[],"rejected":[]}},
      nfr={"R-NF-1":"p95 latency is measured and held under 300 ms"})
    assert [e["key"] for e in merged["stack_entries"]] == ["language"]
    assert merged["stack_entry_dispositions"] == [{"key":"hosting","status":"research_required","research_brief":"Compare current regional terms"}]
    assert merged["decisions"][0]["technology_selections"][0]["display_name"] == "Python"
    assert merged["non_functionals"] == {"R-NF-1":"p95 latency is measured and held under 300 ms"}
