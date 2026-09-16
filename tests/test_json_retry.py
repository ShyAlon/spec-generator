"""Transient-JSON-flake recovery (his live 'tests' stage failure): tolerant
extraction + ONE automatic retry with feedback on every authoring stage."""
import json

import pytest
from langchain_core.messages import AIMessage

from spec_generator import model_config
from spec_generator.errors import UserFacingError
from spec_generator.graph import _call_json, _extract_json
from spec_generator.progress import ProgressBus
from spec_generator.telemetry import UsageLog


def test_extract_plain_and_fenced():
    assert _extract_json('{"a": 1}') == '{"a": 1}'
    assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_prose_wrapped_object_and_array():
    assert json.loads(_extract_json('Sure! Here is the result:\n{"a": {"b": [1, 2]}}\nHope that helps')) == {"a": {"b": [1, 2]}}
    assert json.loads(_extract_json('Output: [1, {"x": "y"}] done')) == [1, {"x": "y"}]


def test_extract_ignores_braces_in_strings():
    assert json.loads(_extract_json('{"s": "a } brace { here"}')) == {"s": "a } brace { here"}


def test_extract_truncated_raises():
    with pytest.raises(ValueError):
        _extract_json('{"a": {"b": [1, 2')  # truncated: no balanced close


class FlakyModel:
    """Garbage once, then valid JSON."""
    def __init__(self):
        self.calls = 0
        self.last_messages = None

    def stream(self, messages):
        self.calls += 1
        self.last_messages = messages
        yield AIMessage(content='Sure, here you go: prose only, no JSON' if self.calls == 1 else '{"ok": true}')


class AlwaysBadModel:
    def stream(self, messages):
        yield AIMessage(content='not json at all')


def test_transient_flake_auto_recovers(monkeypatch):
    m = FlakyModel()
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: m)
    bus = ProgressBus()
    seen = []
    bus.subscribe(seen.append)
    log = UsageLog("s")
    data = _call_json("tests", "payload", log, bus)
    assert data == {"ok": True}
    assert m.calls == 2  # one automatic retry, no human involved
    assert any("retrying once" in (e.detail or "") for e in seen)
    # the retry carries the feedback as an extra message
    assert any("not valid JSON" in str(getattr(msg, "content", "")) for msg in m.last_messages)
    assert len(log.records) == 2  # both calls logged


def test_persistent_garbage_gets_friendly_error_with_snippet(monkeypatch):
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: AlwaysBadModel())
    with pytest.raises(UserFacingError) as ei:
        _call_json("tests", "payload", UsageLog("s"), None)
    rendered = ei.value.render()
    assert "automatic retry did not fix it" in rendered
    assert "not json at all" in rendered  # raw snippet for diagnosis
    assert "SPEC_GENERATOR_MODEL_TESTS" in rendered


def test_valid_first_try_means_one_call(monkeypatch):
    class Clean:
        def __init__(self):
            self.calls = 0

        def stream(self, messages):
            self.calls += 1
            yield AIMessage(content='[{"n": 1}]')

    m = Clean()
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: m)
    assert _call_json("needs", "p", UsageLog("s"), None) == [{"n": 1}]
    assert m.calls == 1  # no behavioral change on the happy path



def _titles_missing_once_then_valid():
    from test_progress import PAYLOADS

    class SchemaBlindModel:
        """The live failure shape: valid JSON, but the fill is missing the
        contract-required 'title' - then a corrected reply."""
        def __init__(self):
            self.calls = 0
            self.last_messages = None

        def stream(self, messages):
            import copy
            self.calls += 1
            self.last_messages = messages
            if self.calls == 1:
                bad = copy.deepcopy(PAYLOADS["srs"])
                del bad["fills"]["R-1"]["title"]
                yield AIMessage(content=json.dumps(bad))
            else:
                yield AIMessage(content=json.dumps(PAYLOADS["srs"]))

    return SchemaBlindModel()


def _srs_validate(d):
    from test_progress import INTENT, PAYLOADS
    from vmodel_contracts import SRSDocument
    from spec_generator.graph import srs_document_data
    from spec_generator.skeleton import build_srs_skeleton, merge_srs_fills
    merged = merge_srs_fills(build_srs_skeleton(PAYLOADS["needs"]), d)
    body = {**merged, "session_id": "s", "intent_model_ref": "s"}
    SRSDocument.model_validate(srs_document_data(INTENT, body))
    return merged


def test_contract_drift_auto_recovers_with_feedback(monkeypatch):
    """Structured output guarantees shape, not contract constraints - a draft
    that parses but fails the contract gets the same one-retry treatment, with
    the contract errors as feedback."""
    m = _titles_missing_once_then_valid()
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: m)
    bus = ProgressBus()
    seen = []
    bus.subscribe(seen.append)
    data = _call_json("srs", "payload", UsageLog("s"), bus, validate=_srs_validate)
    assert data["functional"][0]["title"] == "Spec file"  # the validator's merged body is returned
    assert data["functional"][0]["acceptance"][0]["criterion_id"] == "AC-1"  # code-assigned
    assert m.calls == 2
    assert any("contract revision requested" in (e.detail or "") for e in seen)
    fb = " ".join(str(getattr(msg, "content", "")) for msg in m.last_messages)
    assert "title" in fb  # the model is told WHAT to fix


def test_persistent_contract_drift_gets_friendly_error(monkeypatch):
    class AlwaysDrifting:
        def stream(self, messages):
            import copy
            from test_progress import PAYLOADS
            bad = copy.deepcopy(PAYLOADS["srs"])
            del bad["fills"]["R-1"]["title"]
            yield AIMessage(content=json.dumps(bad))

    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: AlwaysDrifting())
    with pytest.raises(UserFacingError) as ei:
        _call_json("srs", "payload", UsageLog("s"), None, validate=_srs_validate)
    rendered = ei.value.render()
    assert "does not satisfy the contract" in rendered
    assert "title" in rendered
    assert "without a model upgrade" in rendered
