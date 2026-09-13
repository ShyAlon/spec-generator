"""MEDIUM experiment tier (his 2026-09-13 three-speed design): deterministic
scope caps (VMODEL_CAP_*) and the per-item quality report that makes two
experiment runs comparable."""
import json

import pytest

from spec_generator import model_config
from spec_generator.graph import _experiment_cap
from spec_generator.skeleton import ModelStructureError
from spec_generator.experiment_report import build_report, render


def test_cap_truncates_and_preserves(monkeypatch):
    monkeypatch.setenv("VMODEL_CAP_NEEDS", "2")
    assert _experiment_cap("NEEDS", [1, 2, 3, 4], "needs") == [1, 2]
    assert _experiment_cap("NEEDS", [1], "needs") == [1]  # under the cap: untouched


def test_cap_unset_or_invalid(monkeypatch):
    monkeypatch.delenv("VMODEL_CAP_NEEDS", raising=False)
    assert _experiment_cap("NEEDS", [1, 2, 3], "needs") == [1, 2, 3]
    monkeypatch.setenv("VMODEL_CAP_NEEDS", "lots")
    with pytest.raises(ModelStructureError, match="not an integer"):
        _experiment_cap("NEEDS", [1, 2, 3], "needs")


def test_caps_flow_through_a_run(monkeypatch):
    # end-to-end: the stack-slot cap truncates the architecture plan, so only
    # the capped slots are filled and land in the assembly
    from test_architecture_micro_pipeline import _MicroModel, SYSTEM_BASELINE
    from spec_generator.graph import build_graph
    from spec_generator.telemetry import UsageLog
    from test_progress import INTENT
    monkeypatch.setenv("VMODEL_CAP_STACK_SLOTS", "1")
    calls = []
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **k: _MicroModel(
                            stage, k.get("schema") or {}, calls,
                            plan_keys=["language", "database", "ci_tool"]))
    r = build_graph(UsageLog("s")).invoke(
        {"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE},
        {"configurable": {"thread_id": "caps"}})
    assert [e["key"] for e in r["architecture"]["stack_entries"]] == ["language"]


def _fixture(tmp_path):
    (tmp_path / "llm-calls.jsonl").write_text("\n".join([
        json.dumps({"event": "llm_call_end", "stage": "architecture", "item": "stack:language",
                    "attempt": 1, "latency_s": 7.0}),
        json.dumps({"event": "llm_call_end", "stage": "architecture", "item": "stack:language",
                    "attempt": 2, "retry_reason": "contract", "latency_s": 7.0}),
        json.dumps({"event": "llm_call_end", "stage": "architecture", "item": "stack:database",
                    "attempt": 1, "latency_s": 6.5}),
        json.dumps({"event": "llm_call_error", "stage": "tests", "item": "T-1",
                    "error": {"kind": "timeout"}}),
    ]))
    (tmp_path / "usage.jsonl").write_text(json.dumps(
        {"stage": "architecture", "prompt_tokens": 100, "completion_tokens": 50,
         "price_per_mtok_prompt_usd": 0.0, "price_per_mtok_completion_usd": 0.0}))
    (tmp_path / "needs.json").write_text(json.dumps({"needs": [{"need_id": "N-1"}]}))
    return tmp_path


def test_report_aggregates_quality_signal(tmp_path):
    rep = build_report(_fixture(tmp_path))
    assert rep["items"]["total"] == 2
    assert rep["items"]["first_pass"] == 1
    assert rep["items"]["retried"][0]["item"] == "stack:language"
    assert rep["items"]["retried"][0]["contract_retries"] == 1
    assert rep["provider_errors"] == [{"stage": "tests", "item": "T-1", "error": "timeout"}]
    assert rep["counts"]["needs"] == 1
    assert rep["tokens"] == {"prompt": 100, "completion": 50}
    text = render(rep)
    assert "first-pass" in text and "RETRY architecture:stack:language" in text
    assert "ERROR tests:T-1 - timeout" in text


def test_report_handles_an_empty_bundle(tmp_path):
    rep = build_report(tmp_path)
    assert rep["items"]["total"] == 0 and rep["items"]["first_pass_rate"] is None
    assert render(rep)  # renders without crashing
