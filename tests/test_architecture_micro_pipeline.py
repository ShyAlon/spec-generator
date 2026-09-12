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
    for sch in (arch_stack_plan_schema(), arch_stack_entry_schema(["repo:x"]),
                arch_adr_plan_schema()):
        assert "stack" not in json.dumps(sch).lower().replace("stack_keys", "")
    entry = arch_stack_entry_schema(["repo:x"])
    assert entry["additionalProperties"] is False
    assert set(entry["properties"]) == {"value", "baseline_refs"}
    assert entry["properties"]["baseline_refs"]["items"] == {"enum": ["repo:x"]}
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
    with pytest.raises(ModelStructureError, match="collide"):
        _validate_adr_plan([{"slug": "migrate-stack-ci-tool"}], taken)
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
                     "ci_tool": ("Jenkins", "Adopt Jenkins for ci_tool")}

    def __init__(self, stage, schema, calls, crash_on=None, depart=False,
                 depart_key="database", adr_slugs=None):
        self.stage, self.schema, self.calls = stage, schema, calls
        self.crash_on, self.depart, self.depart_key = crash_on, depart, depart_key
        self.adr_slugs = adr_slugs

    def stream(self, messages):
        self.calls.append(self.schema.get("properties", self.schema.get("type")))
        props = set(self.schema.get("properties", {}))
        if props == {"stack_keys"}:
            out = {"stack_keys": (["language", self.depart_key] if self.depart
                                  else ["language", "database"])}
        elif props == {"value", "baseline_refs"}:
            prompt = messages[-1].content
            if (f"THE STACK KEY TO DECIDE: {self.depart_key}" in prompt and self.depart):
                out = {"value": self.DEPART_VALUES[self.depart_key][0], "baseline_refs": []}
            else:
                out = {"value": "ts", "baseline_refs": ["repo:spec-generator"]}
        elif props == {"responsibility"}:
            out = {"responsibility": "serve"}
        elif props == {"position"}:
            out = {"position": "fast"}
        elif "driver" in props:
            prompt = messages[-1].content
            migration = '"migration": true' in prompt
            out = {"title": self.DEPART_VALUES[self.depart_key][1] if migration else "t",
                   "decision": "Adopt the replacement", "rationale": "shared transactional state",
                   "driver": "simplicity",
                   "options_considered": [], "baseline_refs": [],
                   "evidence": ["R-9 requires shared state"] if migration else [],
                   "affected_repositories": ["spec-generator"] if migration else []}
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
    assert db["baseline_refs"] == [] and db["migration_adr_id"] == "ADR-1"
    assert arch["decisions"][0]["migration"] is True
    assert arch["decisions"][0]["affected_repositories"] == ["spec-generator"]


def test_ci_tool_departure_is_authorized_without_model_cooperation(monkeypatch):
    # the exact 2026-09-12 live failure, end to end: stack:ci_tool departed
    # from the baseline and the model's ADR plan authorized NOTHING (its
    # ci_tool-absence slot had migration:false). The pipeline now generates
    # the migration slot itself; the model only authors the body prose.
    r = _run(monkeypatch, [], depart=True, depart_key="ci_tool",
             adr_slugs=[{"slug": "ci_tool-absence"}])
    arch = r["architecture"]
    ci = next(e for e in arch["stack_entries"] if e["key"] == "ci_tool")
    assert ci["baseline_refs"] == [] and ci["migration_adr_id"] == "ADR-1"
    first = arch["decisions"][0]
    assert first["decision_id"] == "ADR-1" and first["migration"] is True
    assert first["title"] == "Adopt Jenkins for ci_tool"  # names the departure
    assert first["evidence"] and first["affected_repositories"] == ["spec-generator"]
    # the model's optional slot survives as a plain non-migration ADR after it
    assert arch["decisions"][1]["decision_id"] == "ADR-2"
    assert arch["decisions"][1]["migration"] is False


def test_per_slot_resume_skips_completed_entries(monkeypatch, tmp_path):
    partials = _make_partials(tmp_path)
    with pytest.raises(RuntimeError):
        _run(monkeypatch, [], partials=partials, crash_on={"value", "baseline_refs"})
    saved = partials.load("architecture")
    assert saved["stack_keys"] == ["language", "database"]
    assert saved.get("stack_entries", {}) == {}  # first entry call crashed before saving
    calls = []
    r = _run(monkeypatch, calls, partials=partials)
    entry_calls = [c for c in calls if isinstance(c, dict) and set(c) == {"value", "baseline_refs"}]
    assert len(entry_calls) == 2  # both entries ran (nothing was saved)
    assert partials.load("architecture") is None  # cleared on completion
    # now crash AFTER the first entry completed: rerun skips it
    calls = []

    class CrashSecond(_MicroModel):
        def stream(self, messages):
            if "THE STACK KEY TO DECIDE: database" in messages[-1].content:
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
    entry_calls = [c for c in calls if isinstance(c, dict) and set(c) == {"value", "baseline_refs"}]
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
