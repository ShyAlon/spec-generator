"""--quick: auto-approve the SRS repair, skip items/tests, scoped re-review, stop with evidence. Never assembles."""
import json
import sys
from argparse import Namespace

import pytest

from spec_generator import cli
from spec_generator.errors import UserFacingError
from spec_generator.findings_ledger import Ledger
from test_jev_d10_repair_gate import _blocked  # noqa: F401
from test_review_fixes_and_persistence import QueueModel, queue_models  # noqa: F401


def _quick_request(g, cfg, values):
    ids = [f["finding_id"] for f in Ledger.from_dict(values["findings_ledger"]).open_findings() if f["rule_id"] == "review.llm"]
    g.update_state(cfg, {"repair_request": {"finding_ids": ids, "requested_by": "founder"},
                         "repair_quick": True, "quick_stop": False}, as_node="review")
    return ids


def test_quick_repair_applies_srs_skips_items_reviews_and_stops(queue_models):
    g, cfg, before = _blocked(queue_models, "q1")
    items_calls = {k: m.calls for k, m in queue_models.items()}
    _quick_request(g, cfg, before)
    QueueModel.QUEUES["review"] = [[]]
    result = g.invoke(None, cfg)                       # no AssemblyBlocked, no proposal stop
    st = g.get_state(cfg).values
    assert not st.get("repair_proposal") and st["quick_stop"] and "bundle" not in result
    assert not g.get_state(cfg).next                   # graph ended at the review
    events = [h["event"] for h in Ledger.from_dict(st["findings_ledger"]).history]
    assert "repair_auto_approved" in events
    for stage in ("items", "tests"):                   # nothing downstream of the SRS was regenerated
        if stage in queue_models:
            assert queue_models[stage].calls == items_calls[stage]


def test_quick_summary_lists_closed_and_open(queue_models):
    g, cfg, before = _blocked(queue_models, "q2")
    _quick_request(g, cfg, before)
    QueueModel.QUEUES["review"] = [[]]
    g.invoke(None, cfg)
    st = g.get_state(cfg).values
    from pathlib import Path
    text = cli._quick_summary(before["findings_ledger"], st["findings_ledger"], Path("out"))
    assert "QUICK REPAIR RESULT" in text and "Closed by repair" in text and "Still OPEN" in text


def test_plain_run_after_quick_is_not_stopped_again(queue_models):
    g, cfg, before = _blocked(queue_models, "q3")
    _quick_request(g, cfg, before)
    QueueModel.QUEUES["review"] = [[]]
    g.invoke(None, cfg)
    # what main() does for a later non-quick run: clear the stop and resume at items
    g.update_state(cfg, {"quick_stop": False, "repair_quick": False}, as_node="apply_fixes")
    assert "items" in g.get_state(cfg).next


