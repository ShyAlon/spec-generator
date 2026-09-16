from conftest import SYSTEM_BASELINE, ModelFactoryLedger, SchemaRoutingArchitectureModel
"""--from <stage>: rewind a checkpoint so a stage and everything after it
re-runs under the CURRENT code (his 2026-09-09 need: the completed run held
pre-fix items with invented requirement ids; a plain resume re-reported the
old violations without re-running anything)."""
import json
import sqlite3
import sys

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from spec_generator import model_config
from spec_generator.cli import _invalidate_from, main
from spec_generator.errors import UserFacingError
from spec_generator.graph import build_graph
from spec_generator.telemetry import UsageLog
from test_progress import CONTRACT_SRS, INTENT, PAYLOADS, FakeModel
from vmodel_contracts import CONTRACT_VERSION


def _run_once(box, monkeypatch, db):
    ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        if stage == "architecture": return ledger.build(stage, kw.get("schema"))
        return box.setdefault(stage, FakeModel(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    box["architecture"] = type("Counter", (), {"calls": property(lambda _self: ledger.calls("architecture"))})()
    g = build_graph(UsageLog("s"), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    cfg = {"configurable": {"thread_id": "s"}}
    g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, cfg)
    return g, cfg


def test_from_items_reruns_downstream_only(monkeypatch, tmp_path):
    box = {}
    db = str(tmp_path / "cp.sqlite")
    g, cfg = _run_once(box, monkeypatch, db)
    counts = {k: v.calls for k, v in box.items()}
    assert counts == {"srs": 1, "architecture": 7, "items": 1, "tests": 1, "review": 1}  # architecture: 3 plans + 3 slot fills

    _invalidate_from(g, cfg, "items", review_mode=False)
    r = g.invoke(None, cfg)
    after = {k: v.calls for k, v in box.items()}
    assert "needs" not in after and after["srs"] == 1 and after["architecture"] == 7  # preserved
    assert after["items"] == 2 and after["tests"] == 2 and after["review"] == 2      # re-fired
    assert "bundle" in r and r["bundle"]["items"][0]["spec_id"] == "SPEC-1"


def test_review_loop_runs_fresh_after_rewind(monkeypatch, tmp_path):
    """review_round is reset, so the rewind doesn't land on the 'already 2
    passes' shortcut and skip the review."""
    box = {}
    db = str(tmp_path / "cp.sqlite")
    g, cfg = _run_once(box, monkeypatch, db)
    g.update_state(cfg, {"review_round": 2})  # simulate a completed two-pass loop
    _invalidate_from(g, cfg, "items", review_mode=False)
    g.invoke(None, cfg)
    assert box["review"].calls == 2  # ran again, not skipped by the stale round count


def test_from_review_reruns_only_review_and_assembly(monkeypatch, tmp_path):
    box = {}
    db = str(tmp_path / "cp.sqlite")
    g, cfg = _run_once(box, monkeypatch, db)
    _invalidate_from(g, cfg, "review", review_mode=False)
    g.invoke(None, cfg)
    after = {k: v.calls for k, v in box.items()}
    assert after["items"] == 1 and after["tests"] == 1 and after["review"] == 2


def test_from_unknown_stage_and_missing_checkpoint(monkeypatch, tmp_path):
    box = {}
    db = str(tmp_path / "cp.sqlite")
    g, cfg = _run_once(box, monkeypatch, db)
    with pytest.raises(UserFacingError, match="unknown stage"):
        _invalidate_from(g, cfg, "banana", review_mode=False)
    g2 = build_graph(UsageLog("other"), checkpointer=SqliteSaver(
        sqlite3.connect(str(tmp_path / "empty.sqlite"), check_same_thread=False)))
    with pytest.raises(UserFacingError, match="no checkpoint"):
        _invalidate_from(g2, {"configurable": {"thread_id": "nobody"}}, "items", review_mode=False)


def test_cli_from_items_end_to_end(monkeypatch, tmp_path, capsys):
    box = {}
    ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        if stage == "architecture": return ledger.build(stage, kw.get("schema"))
        return box.setdefault(stage, FakeModel(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    box["architecture"] = type("Counter", (), {"calls": property(lambda _self: ledger.calls("architecture"))})()
    intent_file = tmp_path / "intent_model.json"
    intent_file.write_text(json.dumps(dict(INTENT, contract_version=CONTRACT_VERSION)))
    baseline_file = tmp_path / "baseline.json"; baseline_file.write_text(json.dumps(SYSTEM_BASELINE))
    outdir = tmp_path / "bundle"
    monkeypatch.setattr(sys, "argv", ["cli", str(intent_file), "-o", str(outdir), "--system-baseline", str(baseline_file)])
    main()
    err = capsys.readouterr().err
    assert f"Bundle dir: {outdir.resolve()}" in err  # he watches folders mid-run - name where progress lands
    assert "*.partial.json" in err
    assert box["items"].calls == 1
    monkeypatch.setattr(sys, "argv", ["cli", str(intent_file), "-o", str(outdir), "--from", "items", "--system-baseline", str(baseline_file)])
    main()
    err = capsys.readouterr().err
    assert "Re-running from stage 'items'" in err
    assert "preserved: needs, srs, architecture" in err
    assert box["items"].calls == 2 and box["srs"].calls == 1
    assert json.loads((outdir / "items.json").read_text())[0]["spec_id"] == "SPEC-1"

def test_from_warns_before_discarding_preserved_partials(monkeypatch, tmp_path, capsys):
    """--from items invalidates downstream partials by design - but it must
    SAY so: a silent discard cost a user 36 finished test items (2026-09-10)."""
    box = {}
    ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        if stage == "architecture": return ledger.build(stage, kw.get("schema"))
        return box.setdefault(stage, FakeModel(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    box["architecture"] = type("Counter", (), {"calls": property(lambda _self: ledger.calls("architecture"))})()
    intent_file = tmp_path / "intent_model.json"
    intent_file.write_text(json.dumps(dict(INTENT, contract_version=CONTRACT_VERSION)))
    baseline_file = tmp_path / "baseline.json"; baseline_file.write_text(json.dumps(SYSTEM_BASELINE))
    outdir = tmp_path / "bundle"
    monkeypatch.setattr(sys, "argv", ["cli", str(intent_file), "-o", str(outdir), "--system-baseline", str(baseline_file)])
    main()
    (outdir / "tests_fills.partial.json").write_text(json.dumps(
        {"T-1": {"description": "d", "runner": "r", "definition": {"preconditions": [], "steps": ["Run x"], "expected_outcomes": ["passes"], "cleanup": [], "cases": [], "execution": {}}}}))
    monkeypatch.setattr(sys, "argv", ["cli", str(intent_file), "-o", str(outdir), "--from", "items", "--system-baseline", str(baseline_file)])
    main()
    err = capsys.readouterr().err
    assert "warning: --from items discards 1 preserved item(s) (tests_fills)" in err
    assert not (outdir / "tests_fills.partial.json").exists()


def test_default_output_dir_is_human_readable_and_stable(monkeypatch, tmp_path, capsys):
    """Default derives from the session timestamp: rerunning the same intent
    resumes the same checkpoint rather than creating a fresh directory. The
    directory lives in the workspace-SIBLING artifacts/ folder (2026-09-12
    founder decision: generated output stays out of the git trees)."""
    box = {}
    ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        if stage == "architecture": return ledger.build(stage, kw.get("schema"))
        return box.setdefault(stage, FakeModel(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    box["architecture"] = type("Counter", (), {"calls": property(lambda _self: ledger.calls("architecture"))})()
    repo = tmp_path / "require" / "spec-generator"; repo.mkdir(parents=True)
    monkeypatch.chdir(repo)
    monkeypatch.delenv("VMODEL_ARTIFACTS_DIR", raising=False)
    monkeypatch.setattr(model_config, "_DOTENV", {})
    intent_file = repo / "intent_model.json"
    intent = dict(INTENT, session_id="20260910-164100-ab12", contract_version=CONTRACT_VERSION)
    intent_file.write_text(json.dumps(intent))
    baseline_file = repo / "baseline.json"; baseline_file.write_text(json.dumps(SYSTEM_BASELINE))

    monkeypatch.setattr(sys, "argv", ["cli", str(intent_file), "--system-baseline", str(baseline_file)])
    main()
    outdir = tmp_path / "require" / "artifacts" / "20260910-164100-ab12" / "spec"
    assert (outdir / "tests.json").exists()
    first_counts = {k: v.calls for k, v in box.items()}

    monkeypatch.setattr(sys, "argv", ["cli", str(intent_file), "--system-baseline", str(baseline_file)])
    main()
    assert {k: v.calls for k, v in box.items()} == first_counts
    assert "Checkpoint found - resuming" in capsys.readouterr().err


def test_explicit_output_dir_is_unchanged(monkeypatch, tmp_path):
    box = {}
    ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        if stage == "architecture": return ledger.build(stage, kw.get("schema"))
        return box.setdefault(stage, FakeModel(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    box["architecture"] = type("Counter", (), {"calls": property(lambda _self: ledger.calls("architecture"))})()
    intent_file = tmp_path / "intent_model.json"
    intent_file.write_text(json.dumps(dict(INTENT, contract_version=CONTRACT_VERSION)))
    baseline_file = tmp_path / "baseline.json"; baseline_file.write_text(json.dumps(SYSTEM_BASELINE))
    chosen = tmp_path / "my-output"
    monkeypatch.setattr(sys, "argv", ["cli", str(intent_file), "-o", str(chosen), "--system-baseline", str(baseline_file)])
    main()
    assert (chosen / "tests.json").exists()


def test_only_stage_loads_exact_upstream_and_refuses_missing(tmp_path):
    from spec_generator.cli import _load_stage_state
    intent = {"session_id": "s"}
    (tmp_path / "srs.json").write_text(json.dumps({"functional": []}))
    (tmp_path / "architecture.json").write_text(json.dumps({"components": []}))
    state = _load_stage_state(tmp_path, intent, SYSTEM_BASELINE, "items")
    assert state == {"intent_model": intent, "system_baseline": SYSTEM_BASELINE,
                     "srs": {"functional": []}, "architecture": {"components": []}}
    with pytest.raises(UserFacingError) as err:
        _load_stage_state(tmp_path, intent, SYSTEM_BASELINE, "tests")
    assert "items.json" in err.value.render()


def test_only_stage_clones_to_new_dir_without_runtime_state(tmp_path):
    from spec_generator.cli import _prepare_stage_iteration
    source = tmp_path / "artifacts-source"; source.mkdir()
    (source / "srs.json").write_text("{}")
    (source / "checkpoints.sqlite").write_text("do not clone")
    (source / "tests_fills.partial.json").write_text("{}")
    (source / "llm-calls.jsonl").write_text("old")
    out = tmp_path / "artifacts-experiment"
    _prepare_stage_iteration(source, out)
    assert (out / "srs.json").exists() and (out / "specs").is_dir()
    assert not (out / "checkpoints.sqlite").exists()
    assert not (out / "tests_fills.partial.json").exists()
    assert not (out / "llm-calls.jsonl").exists()
    with pytest.raises(UserFacingError):
        _prepare_stage_iteration(source, source)


def test_only_stage_graph_ends_after_named_node(monkeypatch):
    from spec_generator.graph import build_graph
    from spec_generator.telemetry import UsageLog
    from test_progress import FakeModel, PAYLOADS
    box = {}
    ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        if stage == "architecture": return ledger.build(stage, kw.get("schema"))
        return box.setdefault(stage, FakeModel(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    box["architecture"] = type("Counter", (), {"calls": property(lambda _self: ledger.calls("architecture"))})()
    state = {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE, "needs": PAYLOADS["needs"],
             "srs": {"purpose": "p", "scope_summary": "s", "functional": [{
                 "requirement_id": "R-1", "need_ids": ["N-1"], "title": "Spec file",
                 "statement": "system shall emit a spec", "priority": "must",
                 "acceptance": [{"criterion_id": "AC-1", "statement": "spec exists"}],
             }], "non_functional": []},
             "architecture": PAYLOADS["architecture"]}
    result = build_graph(UsageLog("s"), only_stage="items").invoke(
        state, {"configurable": {"thread_id": "only-items"}})
    assert "items" in result
    assert set(box) == {"items"}


def test_default_outdir_lands_in_workspace_sibling_artifacts(tmp_path, monkeypatch):
    # 2026-09-12 founder decision: generated output lives OUTSIDE the git
    # trees, in the workspace-sibling artifacts/ folder (shared convention,
    # vmodel_contracts.runtime.artifacts_root).
    from spec_generator import cli, model_config

    repo = tmp_path / "require" / "spec-generator"; repo.mkdir(parents=True)
    monkeypatch.chdir(repo)
    monkeypatch.delenv("VMODEL_ARTIFACTS_DIR", raising=False)
    monkeypatch.setattr(model_config, "_DOTENV", {})  # isolate from any real ../.env
    intent = repo / "intent.json"; intent.write_text("{}")
    out = cli._default_outdir(intent, "20260910-164100-abc")
    assert out == (tmp_path / "require" / "artifacts" / "20260910-164100-abc" / "spec").resolve()


def test_default_outdir_env_override_and_explicit_resume_note(tmp_path, monkeypatch):
    from spec_generator import cli

    monkeypatch.setenv("VMODEL_ARTIFACTS_DIR", str(tmp_path / "custom"))
    intent = tmp_path / "intent.json"; intent.write_text("{}")
    out = cli._default_outdir(intent, "20260910-164100-abc")
    assert out == (tmp_path / "custom" / "20260910-164100-abc" / "spec").resolve()


def _run_cli(monkeypatch, intent_file, baseline_file, outdir):
    monkeypatch.setattr(sys, "argv", ["cli", str(intent_file), "-o", str(outdir),
                                      "--system-baseline", str(baseline_file)])
    main()


def test_baseline_change_invalidates_architecture_downstream(monkeypatch, tmp_path, capsys):
    # 2026-09-13: the poisoned overnight run (six-repo baseline) was resumed
    # with --greenfield; the checkpoint said all stages complete, NOTHING
    # re-ran, and the CLI printed "System baseline updated" - re-labeling the
    # wrong-system bundle as greenfield-anchored. A baseline change on resume
    # MUST invalidate architecture and everything downstream, never re-label.
    box = {}
    ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        if stage == "architecture": return ledger.build(stage, kw.get("schema"))
        return box.setdefault(stage, FakeModel(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    box["architecture"] = type("Counter", (), {"calls": property(lambda _self: ledger.calls("architecture"))})()
    intent_file = tmp_path / "intent_model.json"
    intent_file.write_text(json.dumps(dict(INTENT, contract_version=CONTRACT_VERSION)))
    b1 = tmp_path / "b1.json"; b1.write_text(json.dumps(SYSTEM_BASELINE))
    b2 = tmp_path / "b2.json"
    # a DIFFERENT system anchor (id differs; repo shape preserved so the
    # FakeModel's canned architecture still grounds against repo:spec-generator
    # - greenfield authoring itself is not what this regression targets)
    b2.write_text(json.dumps(dict(SYSTEM_BASELINE, baseline_id="other-system-v1")))
    outdir = tmp_path / "bundle"
    _run_cli(monkeypatch, intent_file, b1, outdir)
    capsys.readouterr()
    assert (outdir / "architecture.json").exists()

    _run_cli(monkeypatch, intent_file, b2, outdir)
    err = capsys.readouterr().err
    # both anchors named, invalidation explained, needs/srs preserved
    assert "WARNING: the --system-baseline for this run differs" in err
    assert "id=test-baseline" in err and "id=other-system-v1" in err
    assert "invalidating them" in err
    # architecture (6 slot calls) and everything downstream re-fired; needs/srs did not
    assert "needs" not in box and box["srs"].calls == 1
    assert box["architecture"].calls == 14 and box["items"].calls == 2
    assert box["tests"].calls == 2 and box["review"].calls == 2
    # the resumed state carries the NEW baseline
    assert json.loads((outdir / "system-baseline.json").read_text())["baseline_id"] == "other-system-v1"


def test_same_baseline_resume_relabels_nothing(monkeypatch, tmp_path, capsys):
    box = {}
    ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        if stage == "architecture": return ledger.build(stage, kw.get("schema"))
        return box.setdefault(stage, FakeModel(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    box["architecture"] = type("Counter", (), {"calls": property(lambda _self: ledger.calls("architecture"))})()
    intent_file = tmp_path / "intent_model.json"
    intent_file.write_text(json.dumps(dict(INTENT, contract_version=CONTRACT_VERSION)))
    b1 = tmp_path / "b1.json"; b1.write_text(json.dumps(SYSTEM_BASELINE))
    outdir = tmp_path / "bundle"
    _run_cli(monkeypatch, intent_file, b1, outdir)
    capsys.readouterr()
    _run_cli(monkeypatch, intent_file, b1, outdir)
    err = capsys.readouterr().err
    assert "WARNING: the --system-baseline" not in err
    assert box["architecture"].calls == 7  # unchanged: completed stages are NOT re-run
    assert box["items"].calls == 1


def test_from_architecture_real_node_persists_resolution_and_matrix_evidence(monkeypatch, tmp_path):
    """Regression for the 2026-09-14 acceptance bundle.

    Exercise the actual checkpoint rewind -> architecture node -> incremental
    persistence path. Unit helpers missed that the new architecture is not in
    LangGraph state until the node returns.
    """
    from spec_generator.cli import _make_persist

    question = "Specific data storage solution for persisting user points across devices"

    class AcceptanceModel(FakeModel):
        def invoke(self, messages):
            prompt = messages[-1].content
            if self.stage == "srs" and "Author ONE requirement fill" in prompt:
                from langchain_core.messages import AIMessage
                self.calls += 1
                return AIMessage(content=json.dumps(PAYLOADS["srs"]["fills"]["R-1"]))
            if self.stage == "items" and "Write the body of ONE item spec" in prompt:
                from langchain_core.messages import AIMessage
                self.calls += 1
                return AIMessage(content=json.dumps({"body_markdown": "# Persistence\nUse Supabase."}))
            if self.stage == "tests" and "TEST SLOT (fill its prose)" in prompt:
                from langchain_core.messages import AIMessage
                import re
                self.calls += 1
                tid = re.search(r'"test_id":\s*"([^"]+)"', prompt).group(1)
                fill = next(iter(PAYLOADS["tests"]["fills"].values()))
                return AIMessage(content=json.dumps({"fills": {tid: fill}}))
            return super().invoke(messages)

        def _architecture_answer(self, prompt: str):
            if "THE STACK KEY TO DECIDE" in prompt:
                return {"technology": "Supabase", "off_list_justification": None, "baseline_refs": []}
            if "THE COMPONENT SLOT" in prompt:
                return {"responsibility": "Persist points across devices"}
            if "THE ADR SLOT" in prompt:
                return {"title": "Adopt Supabase", "decision": "Use Supabase for persistence",
                        "rationale": "One managed backend.", "driver": "simplicity",
                        "options_considered": [{"option": "Firebase", "reason": "Second backend"}],
                        "baseline_refs": [], "evidence": [], "affected_repositories": [], "executes_concept_ids": []}
            if "THE NON-FUNCTIONAL REQUIREMENT" in prompt:
                return {"position": "No special position"}
            if "Plan the OPTIONAL architecture decision records" in prompt:
                return [{"slug": "adopt-supabase"}]
            if "PIPELINE-REQUIRED MIGRATION ADRS" in prompt:
                return []
            if "REQUIREMENT IDS TO COVER" in prompt:
                return [{"name": "persistence", "requirement_ids": ["R-1"],
                         "baseline_refs": [], "migration": False}]
            return {"stack_keys": ["data_persistence_service"]}

    box = {}
    monkeypatch.delenv("SPEC_GENERATOR_BULK", raising=False)
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: box.setdefault(stage, AcceptanceModel(stage)))
    db = str(tmp_path / "checkpoints.sqlite")
    outdir = tmp_path / "bundle"; outdir.mkdir(); (outdir / "specs").mkdir()
    intent = {**INTENT, "open_questions": [question]}
    g = build_graph(UsageLog("s"),
                    checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)),
                    persist=_make_persist(intent, outdir, greenfield=True))
    cfg = {"configurable": {"thread_id": "s"}}
    g.invoke({"intent_model": intent,
              "system_baseline": {**SYSTEM_BASELINE, "mode": "greenfield", "repositories": []}}, cfg)

    # Recreate the stale pre-fix checkpoint/artifact, then take the exact
    # production resume path requested by `--from architecture`.
    prior = g.get_state(cfg).values
    stale_srs = dict(prior["srs"]); stale_srs.pop("_resolved_open_questions", None)
    g.update_state(cfg, {"srs": stale_srs})
    _invalidate_from(g, cfg, "architecture", review_mode=False)
    result = g.invoke(None, cfg)

    stored = g.get_state(cfg).values
    assert stored["srs"]["_resolved_open_questions"] == [question]
    assert question not in json.loads((outdir / "srs.json").read_text())["open_questions"]
    architecture_md = (outdir / "architecture.md").read_text()
    assert "Driver: simplicity (model judgment - unverified; matrix: 2 sources" in architecture_md
    assert question not in result["bundle"]["_srs"]["open_questions"]


def test_to_architecture_stops_real_checkpoint_before_downstream(monkeypatch, tmp_path):
    """A live bounded experiment stops at architecture and later resumes."""
    box = {}
    ledger = ModelFactoryLedger(SchemaRoutingArchitectureModel)
    def factory(stage, **kw):
        if stage == "architecture": return ledger.build(stage, kw.get("schema"))
        return box.setdefault(stage, FakeModel(stage))
    monkeypatch.setattr(model_config, "build_chat_model", factory)
    box["architecture"] = type("Counter", (), {"calls": property(lambda _self: ledger.calls("architecture"))})()
    db = sqlite3.connect(str(tmp_path / "to.sqlite"), check_same_thread=False)
    g = build_graph(UsageLog("s"), checkpointer=SqliteSaver(db), stop_after="architecture")
    cfg = {"configurable": {"thread_id": "to-s"}}
    result = g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, cfg)
    assert result["architecture"]
    assert not result.get("items") and not result.get("tests") and not result.get("bundle")
    assert set(box) == {"srs", "architecture"}
    assert g.get_state(cfg).values["architecture"]


def test_reconcile_artifacts_copy_hashes_semantics_stales_and_audits(tmp_path):
    from spec_generator.cli import _reconcile_artifacts

    source = tmp_path / "run"; source.mkdir()
    question = "Specific data storage solution for persisting user points across devices"
    intent = {**INTENT, "open_questions": [question]}
    srs = {**CONTRACT_SRS, "open_questions": [question], "definitions": {},
           "constraints": [], "assumptions": []}
    architecture = {
        "session_id": "s", "baseline_id": "test-baseline",
        "stack_entries": [{"key": "data_persistence_service", "value": "Supabase", "technology_id":"tech:supabase", "decision_id":"ADR-1",
                           "baseline_refs": [], "migration_adr_id": None}],
        "components": [],
        "decisions": [{"decision_id": "ADR-1", "title": "Adopt Supabase",
                       "decision": "Use Supabase", "rationale": "Simple managed backend",
                       "driver": "simplicity", "options_considered": [], "baseline_refs": [],
                       "migration": False, "evidence": [], "affected_repositories": [], "executes_concept_ids":[], "technology_selections":[{"slot_id":"slot:data-persistence-service","technology_id":"tech:supabase","display_name":"Supabase"}], "rejected": []}],
        "non_functionals": {}}
    (source / "srs.json").write_text(json.dumps(srs))
    (source / "architecture.json").write_text(json.dumps(architecture))
    (source / "srs.md").write_text("old")
    (source / "architecture.md").write_text("old")
    (source / "items.json").write_text("[]")
    target = tmp_path / "generation-2"

    audit = _reconcile_artifacts(intent, SYSTEM_BASELINE, source, target)
    assert audit["semantic_changed"] == ["srs"] and audit["downstream"] == "stale"
    assert question not in json.loads((target / "srs.json").read_text())["open_questions"]
    assert "matrix: 2 sources" in (target / "architecture.md").read_text()
    status = json.loads((target / "artifact-status.json").read_text())
    assert status["artifacts"]["items.json"]["state"] == "stale"
    assert "semantic basis changed" in status["artifacts"]["items.json"]["cause"]
    assert json.loads((target / "audit.jsonl").read_text())["actor"] == "local user"
    assert (source / "srs.md").read_text() == "old", "comparison mode never mutates live bundle"


def test_reconcile_render_only_keeps_downstream_valid(tmp_path):
    from spec_generator.cli import _reconcile_artifacts, srs_document_data
    from spec_generator.render import render_srs
    from vmodel_contracts import SRSDocument

    source = tmp_path / "run"; source.mkdir()
    intent = dict(INTENT)
    srs = srs_document_data(intent, CONTRACT_SRS)
    arch = PAYLOADS["architecture"]
    (source / "srs.json").write_text(json.dumps(srs))
    (source / "architecture.json").write_text(json.dumps(arch))
    (source / "srs.md").write_text(render_srs(SRSDocument.model_validate(srs)))
    (source / "architecture.md").write_text("stale rendering")
    (source / "items.json").write_text("[]")
    audit = _reconcile_artifacts(intent, SYSTEM_BASELINE, source)
    assert audit["semantic_changed"] == [] and audit["downstream"] == "valid"
    status = json.loads((source / "artifact-status.json").read_text())
    assert status["artifacts"] == {}


def test_semantic_hash_equivalence_v1_ignores_evidence_but_not_open_questions():
    from spec_generator.cli import _semantic_hash

    base = {"decision": "Use Supabase", "evidence": ["old citation"], "open_questions": []}
    evidence_only = {**base, "evidence": ["new matrix citation"]}
    semantic = {**base, "open_questions": ["Which storage?"]}
    assert _semantic_hash(base) == _semantic_hash(evidence_only)
    assert _semantic_hash(base) != _semantic_hash(semantic)


def test_full_resume_reconnects_checkpoint_stopped_after_architecture(monkeypatch,tmp_path):
    """Exact production bug: --to architecture leaves no pending SQLite task; full resume must add the edge."""
    from spec_generator.cli import _resume_after_bounded_stop
    box={}
    monkeypatch.setattr(model_config,"build_chat_model",lambda stage,**kw:box.setdefault(stage,FakeModel(stage)))
    db_path=str(tmp_path/"bounded.sqlite"); cfg={"configurable":{"thread_id":"bounded"}}
    first=build_graph(UsageLog("bounded"),checkpointer=SqliteSaver(sqlite3.connect(db_path,check_same_thread=False)),stop_after="architecture")
    stopped=first.invoke({"intent_model":dict(INTENT),"system_baseline":SYSTEM_BASELINE},cfg)
    assert stopped.get("architecture") and not stopped.get("bundle") and not first.get_state(cfg).next
    full=build_graph(UsageLog("bounded"),checkpointer=SqliteSaver(sqlite3.connect(db_path,check_same_thread=False)))
    prior=full.get_state(cfg).values
    assert _resume_after_bounded_stop(full,cfg,prior)=="architecture"
    resumed=full.invoke(None,cfg)
    assert resumed.get("bundle") and resumed["items"] and resumed["tests"]


def test_render_diagnostics_do_not_blame_model_for_none_bundle():
    import inspect
    from spec_generator import cli
    source=inspect.getsource(cli.main)
    assert "assembly produced no bundle" in source
    assert "changing the model cannot fix a code exception" in source
    assert "artifact '{label}'" in source
