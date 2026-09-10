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
from test_progress import INTENT, PAYLOADS, FakeModel
from vmodel_contracts import CONTRACT_VERSION


def _run_once(box, monkeypatch, db):
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: box.setdefault(stage, FakeModel(stage)))
    g = build_graph(UsageLog("s"), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    cfg = {"configurable": {"thread_id": "s"}}
    g.invoke({"intent_model": dict(INTENT)}, cfg)
    return g, cfg


def test_from_items_reruns_downstream_only(monkeypatch, tmp_path):
    box = {}
    db = str(tmp_path / "cp.sqlite")
    g, cfg = _run_once(box, monkeypatch, db)
    counts = {k: v.calls for k, v in box.items()}
    assert counts == {"needs": 1, "srs": 1, "architecture": 1, "items": 1, "tests": 1, "review": 1}

    _invalidate_from(g, cfg, "items", review_mode=False)
    r = g.invoke(None, cfg)
    after = {k: v.calls for k, v in box.items()}
    assert after["needs"] == 1 and after["srs"] == 1 and after["architecture"] == 1  # preserved
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
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: box.setdefault(stage, FakeModel(stage)))
    intent_file = tmp_path / "intent_model.json"
    intent_file.write_text(json.dumps(dict(INTENT, contract_version=CONTRACT_VERSION)))
    outdir = tmp_path / "bundle"
    monkeypatch.setattr(sys, "argv", ["cli", str(intent_file), "-o", str(outdir)])
    main()
    err = capsys.readouterr().err
    assert f"Bundle dir: {outdir.resolve()}" in err  # he watches folders mid-run - name where progress lands
    assert "*.partial.json" in err
    assert box["items"].calls == 1
    monkeypatch.setattr(sys, "argv", ["cli", str(intent_file), "-o", str(outdir), "--from", "items"])
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
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kw: box.setdefault(stage, FakeModel(stage)))
    intent_file = tmp_path / "intent_model.json"
    intent_file.write_text(json.dumps(dict(INTENT, contract_version=CONTRACT_VERSION)))
    outdir = tmp_path / "bundle"
    monkeypatch.setattr(sys, "argv", ["cli", str(intent_file), "-o", str(outdir)])
    main()
    (outdir / "tests_fills.partial.json").write_text(json.dumps(
        {"T-1": {"description": "d", "runner": "r", "definition": {"preconditions": [], "steps": ["Run x"], "expected_outcomes": ["passes"], "cleanup": [], "cases": [], "execution": {}}}}))
    monkeypatch.setattr(sys, "argv", ["cli", str(intent_file), "-o", str(outdir), "--from", "items"])
    main()
    err = capsys.readouterr().err
    assert "warning: --from items discards 1 preserved item(s) (tests_fills)" in err
    assert not (outdir / "tests_fills.partial.json").exists()
