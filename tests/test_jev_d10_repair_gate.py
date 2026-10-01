"""JEV-D10: assembly gate, ledger-less checkpoints, repair ceiling, founder-only closure, CLI actions."""
import json
from argparse import Namespace

import pytest
from langgraph.checkpoint.memory import MemorySaver

from conftest import SYSTEM_BASELINE
from spec_generator.cli import _apply_ledger_actions
from spec_generator.errors import AssemblyBlocked, UserFacingError
from spec_generator.findings_ledger import Ledger
from spec_generator.graph import REPAIR_ATTEMPT_CEILING, _normalized_review_fixes, build_graph
from spec_generator.telemetry import UsageLog
from test_progress import INTENT
from test_review_fixes_and_persistence import FIXES, QueueModel, _request_repair, _run_to_block, queue_models  # noqa: F401


def _blocked(queue_models, tid="g1", fixes=FIXES):
    QueueModel.QUEUES["review"] = [fixes, [], [], []]
    g = build_graph(UsageLog("s"), checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": tid}}
    return g, cfg, _run_to_block(g, cfg)


def test_open_review_finding_blocks_assembly(queue_models):
    g, cfg, values = _blocked(queue_models)
    ledger = Ledger.from_dict(values["findings_ledger"])
    assert [f["rule_id"] for f in ledger.open_findings()] == ["review.llm"]
    assert "bundle" not in values


def test_clean_review_assembles(queue_models):
    QueueModel.QUEUES["review"] = [[]]
    g = build_graph(UsageLog("s"), checkpointer=MemorySaver())
    r = g.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, {"configurable": {"thread_id": "c"}})
    assert "bundle" in r


def test_ledger_less_checkpoint_is_unreviewed_not_clean(queue_models):
    g, cfg, values = _blocked(queue_models, "g2")
    g.update_state(cfg, {"findings_ledger": None}, as_node="review")
    with pytest.raises(AssemblyBlocked) as e:
        g.invoke(None, cfg)
    assert "never content-reviewed" in e.value.reasons[0]


def test_more_than_24_findings_survive(queue_models):
    many = [{"fix": f"R-1 problem number {i} needs a distinct fix"} for i in range(30)]
    g, cfg, values = _blocked(queue_models, "g3", many)
    ledger = Ledger.from_dict(values["findings_ledger"])
    assert len(ledger.open_findings()) == 30
    assert any("30 findings" in i["reason"] for i in ledger.incomplete)


def test_repair_does_not_close_and_ceiling_holds(queue_models):
    g, cfg, values = _blocked(queue_models, "g4")
    for attempt in range(REPAIR_ATTEMPT_CEILING + 1):
        ids = _request_repair(g, cfg, g.get_state(cfg).values)
        with pytest.raises(AssemblyBlocked):
            g.invoke(None, cfg)
    final = Ledger.from_dict(g.get_state(cfg).values["findings_ledger"])
    repairs = [h for h in final.history if h.get("event") == "repair"]
    assert len(repairs) <= REPAIR_ATTEMPT_CEILING          # the extra request was refused, not replayed
    assert [f for f in final.open_findings() if f["rule_id"] == "review.llm"]  # never self-closed
    assert queue_models["srs"].calls <= 1 + REPAIR_ATTEMPT_CEILING


def test_cli_dispose_closes_only_with_a_founder_script_and_unblocks(queue_models, tmp_path):
    g, cfg, values = _blocked(queue_models, "g5")
    f = Ledger.from_dict(values["findings_ledger"]).open_findings()[0]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 1, "authored_by": "agent", "dispositions": [{}]}))
    with pytest.raises(UserFacingError):
        _apply_ledger_actions(g, cfg, values, Namespace(dispose=bad, repair=None))
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"version": 1, "authored_by": "founder", "dispositions": [{
        "selector": {"rule_id": f["rule_id"], "target_ids": f["targets"]}, "action": "reject",
        "expected_artifact_sha256": f["artifact_revision"], "rationale": "acceptable as written"}]}))
    new = _apply_ledger_actions(g, cfg, values, Namespace(dispose=good, repair=None))
    assert not Ledger.from_dict(new["findings_ledger"]).open_findings()
    QueueModel.QUEUES["review"] = [[]]
    assert "bundle" in g.invoke(None, cfg)


def test_cli_repair_rejects_unknown_ids(queue_models):
    g, cfg, values = _blocked(queue_models, "g6")
    with pytest.raises(UserFacingError):
        _apply_ledger_actions(g, cfg, values, Namespace(dispose=None, repair="F-nope"))


def test_docs_show_the_findings_ledger():
    from spec_generator.docs import render_docs
    ledger = Ledger()
    ledger.add(rule_id="echo.directive", rule_version="1", source="rule", kind="semantic", targets=["SPEC-1"],
               artifact_revision="x", summary="gray echo")
    files = render_docs({"findings_ledger": ledger.to_dict()})
    assert "BLOCKED" in files["findings.md"] and "SPEC-1" in files["findings.md"]
