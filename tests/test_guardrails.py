"""Guardrails around every LLM call: run-wide call budget, per-call wall-clock
deadline, stream stall limit - informative kills, never silent hangs. Plus the
per-item start/end instrumentation in the progress stream."""
import re
import time

import pytest

import spec_generator.model_config as mc
from spec_generator.errors import UserFacingError
from spec_generator.progress import ProgressBus, ProgressEvent, stderr_printer
from spec_generator.stream_items import ItemReporter


class FakeModel:
    def __init__(self, text="ok", invoke_sleep_s=0.0, chunks=(), stall_s=0.0):
        self._text, self._invoke_sleep = text, invoke_sleep_s
        self._chunks, self._stall = chunks, stall_s

    def invoke(self, messages):
        time.sleep(self._invoke_sleep)
        return self._text

    def stream(self, messages):
        for c in self._chunks:
            yield c
        if self._stall:
            time.sleep(self._stall)
            yield type("C", (), {"content": "tail"})()


@pytest.fixture(autouse=True)
def _clean_guard_env(monkeypatch):
    monkeypatch.setattr(mc, "_CALLS_USED", 0)
    for v in ("VMODEL_LLM_CALL_TIMEOUT_S", "VMODEL_LLM_STALL_TIMEOUT_S", "VMODEL_LLM_MAX_CALLS",
              "VMODEL_LLM_MAX_TOKENS", "VMODEL_LLM_LOG", "VMODEL_LLM_HEARTBEAT_S", "VMODEL_OLLAMA_THINK", "VMODEL_OLLAMA_NUM_CTX", "VMODEL_OLLAMA_KEEP_ALIVE", "SPEC_GENERATOR_FROM"):
        monkeypatch.delenv(v, raising=False)


def test_invoke_under_deadline_returns_normally():
    assert mc.invoke_checked(FakeModel("done"), [], "srs") == "done"


def test_slow_invoke_is_killed_with_an_informative_error(monkeypatch):
    monkeypatch.setenv("VMODEL_LLM_CALL_TIMEOUT_S", "1")
    with pytest.raises(UserFacingError) as ei:
        mc.invoke_checked(FakeModel(invoke_sleep_s=5), [], "srs")
    err = ei.value.render()
    assert "killed" in err and "'srs'" in err
    assert "ollama" in err and "qwen2.5-coder:14b" in err  # stage + provider + model named
    assert "rerun the same command" in err


def test_run_call_budget_trips_with_an_informative_error(monkeypatch):
    monkeypatch.setenv("VMODEL_LLM_MAX_CALLS", "2")
    mc.invoke_checked(FakeModel(), [], "srs")
    mc.invoke_checked(FakeModel(), [], "srs")
    with pytest.raises(UserFacingError) as ei:
        mc.invoke_checked(FakeModel(), [], "srs")
    err = ei.value.render()
    assert "budget" in err and "2 calls" in err and "VMODEL_LLM_MAX_CALLS" in err


def test_stream_passthrough_unchanged():
    chunks = [type("C", (), {"content": t})() for t in ('{"a": ', "1}")]
    out = "".join(c.content for c in mc.stream_checked(FakeModel(chunks=chunks), [], "srs"))
    assert out == '{"a": 1}'


def test_stalled_stream_is_killed_with_an_informative_error(monkeypatch):
    monkeypatch.setenv("VMODEL_LLM_STALL_TIMEOUT_S", "1")
    monkeypatch.setenv("VMODEL_LLM_CALL_TIMEOUT_S", "60")
    chunks = [type("C", (), {"content": '{"a":'})()]
    gen = mc.stream_checked(FakeModel(chunks=chunks, stall_s=5), [], "items")
    assert next(gen).content == '{"a":'  # first chunk flows through
    with pytest.raises(UserFacingError) as ei:
        next(gen)
    err = ei.value.render()
    assert "killed" in err and "stall" in err and "'items'" in err


def test_silent_prefill_is_not_a_stall(monkeypatch):
    """No chunks BEFORE the first token = prefill (healthy, bounded by the
    call budget) - the stall guard only applies mid-stream (2026-09-10: a
    healthy T-1 fill was killed at 180s while ollama prefilled SRS+ITEMS)."""
    monkeypatch.setenv("VMODEL_LLM_STALL_TIMEOUT_S", "1")
    monkeypatch.setenv("VMODEL_LLM_CALL_TIMEOUT_S", "30")

    class SlowPrefill:
        def stream(self, messages):
            time.sleep(3)  # 3x the stall limit, before ANY chunk
            yield type("C", (), {"content": '{"a": 1}'})()

    out = list(mc.stream_checked(SlowPrefill(), [], "items"))
    assert "".join(c.content for c in out) == '{"a": 1}'


def test_midstream_silence_is_still_a_stall(monkeypatch):
    """A stream that produced a chunk and then goes quiet IS hung."""
    monkeypatch.setenv("VMODEL_LLM_STALL_TIMEOUT_S", "1")
    monkeypatch.setenv("VMODEL_LLM_CALL_TIMEOUT_S", "30")

    class MidStall:
        def stream(self, messages):
            yield type("C", (), {"content": '{"a":'})()
            time.sleep(3)
            yield type("C", (), {"content": ' 1}'})()

    gen = mc.stream_checked(MidStall(), [], "items")
    next(gen)
    with pytest.raises(UserFacingError) as ei:
        next(gen)
    assert "stalled mid-call" in ei.value.render()


def test_item_reporter_carries_start_and_end():
    bus = ProgressBus()
    events = []
    bus.subscribe(events.append)
    r = ItemReporter(bus, "needs")
    time.sleep(0.05)
    r.feed('[{"need_id": "R-1", "title": "first"}, ')
    time.sleep(0.05)
    r.feed('[{"need_id": "R-1", "title": "first"}, {"need_id": "R-2", "title": "second"}]')
    items = [e for e in events if e.status == "item"]
    assert len(items) == 2
    assert all(e.elapsed_s is not None and e.elapsed_s >= 0 for e in items)
    assert items[0].ts <= items[1].ts
    assert items[1].ts - items[1].elapsed_s >= items[0].ts - 1e-6  # item 2 starts when item 1 ended


def test_printer_renders_item_timestamps(capsys):
    bus = ProgressBus()
    bus.subscribe(stderr_printer())
    end = time.time()
    bus.emit(ProgressEvent("items", "", "item", end, 3.2, "SPEC-5 (gateway) Set up Kong API gateway"))
    out = capsys.readouterr().err
    assert re.search(r"\[\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}\] SPEC-5 \(gateway\) Set up Kong API gateway", out)
    # events without timing (retry notices) keep the old plain line
    bus.emit(ProgressEvent("srs", "", "item", end, None, "model output was not valid JSON - retrying once"))
    assert "- model output was not valid JSON - retrying once" in capsys.readouterr().err


class _Resp:
    def __init__(self, content, reason=None):
        self.content = content
        self.response_metadata = {} if reason is None else {"done_reason": reason}


def test_generation_length_ceiling_defaults_to_the_stage_ceiling(monkeypatch):
    assert mc._max_tokens_limit(8192) == 8192
    monkeypatch.setenv("VMODEL_LLM_MAX_TOKENS", "1024")
    assert mc._max_tokens_limit(8192) == 1024
    monkeypatch.setenv("VMODEL_LLM_MAX_TOKENS", "junk")
    assert mc._max_tokens_limit(8192) == 8192  # garbage falls back


def test_truncated_generation_is_named_not_silently_passed():
    class M:
        def invoke(self, messages):
            return _Resp('{"partial":', "length")  # ollama done_reason / openai finish_reason

    with pytest.raises(UserFacingError) as ei:
        mc.invoke_checked(M(), [], "srs")
    err = ei.value.render()
    assert "generation-length ceiling" in err and "8192 tokens" in err
    assert "VMODEL_LLM_MAX_TOKENS" in err and "'srs'" in err


def test_clean_finish_reason_passes():
    class M:
        def invoke(self, messages):
            return _Resp("{}", "stop")

    assert mc.invoke_checked(M(), [], "srs").content == "{}"


def test_stream_cut_at_token_ceiling_is_named_not_silently_passed():
    chunks = [_Resp('{"a":', None), _Resp("1}", "length")]
    gen = mc.stream_checked(FakeModel(chunks=chunks), [], "items")
    assert next(gen).content == '{"a":'  # chunks still flow through
    with pytest.raises(UserFacingError) as ei:
        list(gen)
    err = ei.value.render()
    assert "generation-length ceiling" in err and "VMODEL_LLM_MAX_TOKENS" in err

def test_kill_guidance_drops_from_when_the_failing_command_carried_it(monkeypatch):
    """2026-09-10: a killed run whose command carried --from items told the
    user to "rerun the same command" - which forcibly re-ran the stage and
    discarded 36 finished items. The guidance must name the --from drop."""
    monkeypatch.setenv("VMODEL_LLM_CALL_TIMEOUT_S", "1")
    monkeypatch.setenv("SPEC_GENERATOR_FROM", "items")
    with pytest.raises(UserFacingError) as ei:
        mc.invoke_checked(FakeModel(invoke_sleep_s=5), [], "tests")
    err = ei.value.render()
    assert "WITHOUT '--from items'" in err
    assert "discards finished per-item work" in err
    assert "checkpoint resumes at this stage" in err


def test_truncation_guidance_also_drops_from(monkeypatch):
    monkeypatch.setenv("SPEC_GENERATOR_FROM", "items")

    class M:
        def invoke(self, messages):
            return _Resp('{"partial":', "length")

    with pytest.raises(UserFacingError) as ei:
        mc.invoke_checked(M(), [], "tests")
    assert "WITHOUT '--from items'" in ei.value.render()


class RateLimitError(Exception):
    """Mimics the openai SDK exception shape (status/body/response.headers)."""

    def __init__(self, message, err_type, code=None, headers=None):
        super().__init__(message)
        self.status_code = 429
        self.body = {"error": {"message": message, "type": err_type, "code": code}}
        self.response = type("R", (), {"headers": headers or {}})()


def _read_log(path):
    import json
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_llm_logging_off_by_default(tmp_path):
    log = tmp_path / "llm.jsonl"
    mc.invoke_checked(type("M", (), {"invoke": lambda self, m: type("R", (), {})()})(), [], "srs")
    assert not log.exists()


def test_llm_logging_file_sink_start_end_usage(tmp_path, monkeypatch):
    log = tmp_path / "llm.jsonl"
    monkeypatch.setenv("VMODEL_LLM_LOG", str(log))

    class M:
        def invoke(self, messages):
            return type("R", (), {"usage_metadata": {"input_tokens": 5, "output_tokens": 3,
                                                     "total_tokens": 8}})()

    mc.invoke_checked(M(), [], "srs", item="IT-1")
    recs = _read_log(log)
    assert [r["event"] for r in recs] == ["llm_call_start", "llm_call_end"]
    assert recs[0]["provider"] == "ollama" and recs[0]["stage"] == "srs" and recs[0]["item"] == "IT-1"
    assert recs[1]["usage"]["total_tokens"] == 8 and recs[1]["latency_s"] >= 0
    assert "messages" not in recs[0] and "prompt" not in recs[0]  # no prompt content logged


def test_llm_logging_stderr_sink(monkeypatch, capsys):
    monkeypatch.setenv("VMODEL_LLM_LOG", "stderr")
    mc.invoke_checked(type("M", (), {"invoke": lambda self, m: type("R", (), {})()})(), [], "srs")
    assert '"llm_call_start"' in capsys.readouterr().err


def test_llm_logging_error_record_masks_keys(tmp_path, monkeypatch):
    log = tmp_path / "llm.jsonl"
    monkeypatch.setenv("VMODEL_LLM_LOG", str(log))
    secret = "sk-abcdVERYSECRETKEY12345"
    exc = RateLimitError(f"quota exceeded for key {secret}", "insufficient_quota",
                              code="insufficient_quota",
                              headers={"x-request-id": "req_test", "retry-after": "20"})

    class M:
        def invoke(self, messages):
            raise exc

    with pytest.raises(UserFacingError):
        mc.invoke_checked(M(), [], "srs")
    rec = [r for r in _read_log(log) if r["event"] == "llm_call_error"][0]
    assert rec["error"]["http_status"] == 429
    assert rec["error"]["code"] == "insufficient_quota"
    assert rec["error"]["request_id"] == "req_test"
    assert rec["error"]["retry_after_s"] == "20"
    assert secret not in log.read_text()  # masked


def test_insufficient_quota_is_billing_not_throughput():
    from spec_generator.errors import translate_llm_error

    exc = RateLimitError("You exceeded your current quota, please check your plan and billing details",
                              "insufficient_quota", code="insufficient_quota")
    err = translate_llm_error(exc, "srs", "openai", "gpt-x", None)
    r = err.render()
    assert "no usable quota" in r and "billing" in r and "waiting changes nothing" in r
    assert "switch-model.sh ollama" in r


def test_true_rate_limit_names_throughput_and_retry_after():
    from spec_generator.errors import translate_llm_error

    exc = RateLimitError("Rate limit reached for gpt-x on requests per min (RPM): max 50",
                              "rate_limit_exceeded", headers={"retry-after": "20"})
    err = translate_llm_error(exc, "srs", "openai", "gpt-x", None)
    r = err.render()
    assert "throughput" in r and "20s" in r


def test_generic_429_points_at_raw_log():
    from spec_generator.errors import translate_llm_error

    exc = RateLimitError("some other 429 situation", "some_type")
    err = translate_llm_error(exc, "srs", "openai", "gpt-x", None)
    assert "VMODEL_LLM_LOG" in err.render()


def test_403_speaks_of_tier_access():
    from spec_generator.errors import translate_llm_error

    exc = type("PermissionDeniedError", (Exception,), {"status_code": 403})("error code: 403 - denied")
    err = translate_llm_error(exc, "srs", "openai", "gpt-x", None)
    r = err.render()
    assert "denied access (403)" in r and "tier" in r


def test_stream_calls_are_logged(tmp_path, monkeypatch):
    log = tmp_path / "llm.jsonl"
    monkeypatch.setenv("VMODEL_LLM_LOG", str(log))

    class M:
        def stream(self, messages):
            yield type("C", (), {"content": "x"})()

    list(mc.stream_checked(M(), [], "items", item="ITEM-1"))
    recs = _read_log(log)
    events = [r["event"] for r in recs]
    assert events == ["llm_call_start", "llm_call_end"]
    assert recs[0]["item"] == "ITEM-1"
    assert recs[1]["ttft_s"] is not None and recs[1]["ttft_s"] <= recs[1]["latency_s"]  # prefill cost visible


def test_heartbeat_prints_during_silent_call_then_stops(monkeypatch, capsys):
    monkeypatch.setenv("VMODEL_LLM_HEARTBEAT_S", "0.05")

    class M:
        def invoke(self, messages):
            time.sleep(0.2)
            return type("R", (), {})()

    mc.invoke_checked(M(), [], "srs")
    err = capsys.readouterr().err
    assert "still working - stage 'srs' call" in err
    time.sleep(0.15)  # ~3 more intervals: no further beats once the call returned
    assert "still working" not in capsys.readouterr().err


def test_heartbeat_disabled_with_zero(monkeypatch, capsys):
    monkeypatch.setenv("VMODEL_LLM_HEARTBEAT_S", "0")

    class M:
        def invoke(self, messages):
            time.sleep(0.1)
            return type("R", (), {})()

    mc.invoke_checked(M(), [], "srs")
    assert "still working" not in capsys.readouterr().err


def test_describe_models_names_resolved_provider_and_model():
    s = mc.describe_models()
    assert "ollama/qwen2.5-coder:14b" in s and "srs=" in s


def test_ollama_thinking_off_by_default(monkeypatch):
    monkeypatch.delenv("VMODEL_OLLAMA_THINK", raising=False)
    m = mc.build_chat_model("srs")
    assert m.reasoning is False  # qwen3 <think> dead time


def test_ollama_thinking_knob_reenables(monkeypatch):
    monkeypatch.setenv("VMODEL_OLLAMA_THINK", "1")
    m = mc.build_chat_model("srs")
    assert m.reasoning is True


def test_ollama_num_ctx_default_and_knob(monkeypatch):
    monkeypatch.delenv("VMODEL_OLLAMA_NUM_CTX", raising=False)
    m = mc.build_chat_model("srs")
    assert m.num_ctx == 32768  # ollama's 4096 default silently truncates stage prompts
    monkeypatch.setenv("VMODEL_OLLAMA_NUM_CTX", "16384")
    m = mc.build_chat_model("srs")
    assert m.num_ctx == 16384


def test_ollama_keep_alive_pinned(monkeypatch):
    monkeypatch.delenv("VMODEL_OLLAMA_KEEP_ALIVE", raising=False)
    m = mc.build_chat_model("srs")
    assert m.keep_alive == "30m"  # idle timer must not unload mid-run
    monkeypatch.setenv("VMODEL_OLLAMA_KEEP_ALIVE", "1h")
    m = mc.build_chat_model("srs")
    assert m.keep_alive == "1h"


def test_ollama_native_metrics_and_prompt_fingerprint_are_logged(tmp_path, monkeypatch):
    log = tmp_path / "native.jsonl"
    monkeypatch.setenv("VMODEL_LLM_LOG", str(log))

    class M:
        def stream(self, messages):
            yield type("C", (), {"content": "{}", "usage_metadata": {}, "response_metadata": {
                "prompt_eval_count": 700, "eval_count": 80,
                "prompt_eval_duration": 2_000_000_000, "eval_duration": 4_000_000_000,
            }})()

    list(mc.stream_checked(M(), [type("Msg", (), {"content": "hello"})()], "tests",
                           item="T-1", attempt=2, retry_reason="contract"))
    end = _read_log(log)[-1]
    assert end["usage"]["input_tokens"] == 700 and end["usage"]["output_tokens"] == 80
    assert end["prompt_eval_duration"] == 2_000_000_000
    assert end["eval_duration"] == 4_000_000_000
    assert end["prompt_bytes"] == 5 and len(end["prompt_sha256"]) == 16
    assert end["attempt"] == 2 and end["retry_reason"] == "contract"
