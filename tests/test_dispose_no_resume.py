"""Bare --dispose records the disposition and stops; it must not clear quick_stop (which would
re-point the checkpoint at items and make the next plain run regenerate everything)."""
import json
from argparse import Namespace

from spec_generator.cli import _pre_run_actions
from spec_generator.findings_ledger import Ledger
from test_jev_d10_repair_gate import _blocked  # noqa: F401
from test_review_fixes_and_persistence import QueueModel, queue_models  # noqa: F401


def _script(tmp_path, f):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"version": 1, "authored_by": "founder", "dispositions": [{
        "selector": {"rule_id": f["rule_id"], "target_ids": f["targets"]}, "action": "reject",
        "expected_artifact_sha256": f["artifact_revision"], "rationale": "acceptable as written"}]}))
    return p


def test_bare_dispose_keeps_quick_stop_and_reports_dispose_only(queue_models, tmp_path):
    g, cfg, values = _blocked(queue_models, "d1")
    g.update_state(cfg, {"quick_stop": True, "repair_quick": True}, as_node="review")
    prior = g.get_state(cfg).values
    f = Ledger.from_dict(prior["findings_ledger"]).open_findings()[0]
    new, dispose_only = _pre_run_actions(g, cfg, prior, Namespace(dispose=_script(tmp_path, f), repair=None, quick=False))
    assert dispose_only is True
    assert new["quick_stop"] is True and new["repair_quick"] is True
    assert not Ledger.from_dict(new["findings_ledger"]).open_findings()


def test_plain_run_after_quick_stop_still_clears_it(queue_models):
    g, cfg, values = _blocked(queue_models, "d2")
    g.update_state(cfg, {"quick_stop": True, "repair_quick": True}, as_node="review")
    new, dispose_only = _pre_run_actions(g, cfg, g.get_state(cfg).values,
                                         Namespace(dispose=None, repair=None, quick=False))
    assert dispose_only is False and not new.get("quick_stop")


def test_dispose_with_repair_is_not_dispose_only(queue_models, tmp_path):
    g, cfg, values = _blocked(queue_models, "d3")
    f = Ledger.from_dict(values["findings_ledger"]).open_findings()[0]
    _, dispose_only = _pre_run_actions(g, cfg, values, Namespace(dispose=_script(tmp_path, f), repair=f["finding_id"], quick=False))
    assert dispose_only is False
