"""A contract rejection gets two repair attempts (three calls); malformed JSON keeps one retry; the retry feedback
names the runner/components coupling of the test-fill gates."""
import pytest
from langchain_core.messages import AIMessage

from spec_generator import model_config
from spec_generator.errors import UserFacingError
from spec_generator.graph import _call_json, _contract_repair_hint
from spec_generator.skeleton import ModelStructureError
from spec_generator.telemetry import UsageLog


class Scripted:
    def __init__(self, replies):
        self.replies, self.calls, self.feedback = replies, 0, []

    def stream(self, messages):
        self.feedback.append(" ".join(str(getattr(m, "content", "")) for m in messages))
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        yield AIMessage(content=reply)


def _gate(errors):
    """Validator that rejects every reply whose text is in `errors` (reply -> message), accepts the rest."""
    def check(d):
        msg = errors.get(d.get("v"))
        if msg:
            raise ModelStructureError(msg)
        return d
    return check


OWN = "T-9: components ['UI Layer'] do not own the requirement this test traces to; choose from ['Retry Service']"
EXE = "T-9: runner 'http' requires definition.execution.url, method, expected_status (missing: url, method, expected_status)"


def test_two_contract_rejections_then_valid_succeeds_on_the_third_call(monkeypatch, tmp_path):
    monkeypatch.setenv("VMODEL_LLM_LOG", str(tmp_path / "llm.jsonl"))
    m = Scripted(['{"v": "own"}', '{"v": "exe"}', '{"v": "ok"}'])
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: m)
    data = _call_json("tests", "p", UsageLog("s"), None, validate=_gate({"own": OWN, "exe": EXE}), item_label="T-9", contract_attempts=3)
    assert data == {"v": "ok"} and m.calls == 3
    assert "change ONLY the components list" in m.feedback[1] and "Retry Service" in m.feedback[1]
    assert "Do not switch the runner" in m.feedback[1]
    assert "runner 'ui' or 'static' with execution {}" in m.feedback[2] and "definition.execution.url, method and expected_status" in m.feedback[2]


def test_three_contract_rejections_still_end_in_the_named_error(monkeypatch, tmp_path):
    monkeypatch.setenv("VMODEL_LLM_LOG", str(tmp_path / "llm.jsonl"))
    m = Scripted(['{"v": "own"}'])
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: m)
    with pytest.raises(UserFacingError) as ei:
        _call_json("tests", "p", UsageLog("s"), None, validate=_gate({"own": OWN}), item_label="T-9", contract_attempts=3)
    assert m.calls == 3 and "does not satisfy the contract" in ei.value.render()


def test_malformed_json_keeps_a_single_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("VMODEL_LLM_LOG", str(tmp_path / "llm.jsonl"))
    m = Scripted(["not json at all"])
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: m)
    with pytest.raises(UserFacingError):
        _call_json("tests", "p", UsageLog("s"), None, item_label="T-9")
    assert m.calls == 2


def test_hint_is_empty_for_other_errors_and_never_names_the_pipeline_note():
    assert _contract_repair_hint("functional.0.title: Field required") == ""
    assert "PIPELINE" not in _contract_repair_hint(OWN + EXE)


def test_other_stages_keep_exactly_one_repair(monkeypatch, tmp_path):
    monkeypatch.setenv("VMODEL_LLM_LOG", str(tmp_path / "llm.jsonl"))
    m = Scripted(['{"v": "own"}'])
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: m)
    with pytest.raises(UserFacingError):
        _call_json("srs", "p", UsageLog("s"), None, validate=_gate({"own": OWN}), item_label="S-1")
    assert m.calls == 2


def test_the_tests_stage_asks_for_three_attempts():
    import inspect
    from spec_generator import graph
    src = inspect.getsource(graph)
    assert src.count("contract_attempts=3") == 2
