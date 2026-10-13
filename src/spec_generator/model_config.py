"""Per-stage model routing for generation (model-agnostic layer, ADR-0003).
LOCAL-FIRST: every stage defaults to Ollama ($0, offline, no key). DeepSeek
(or openai/anthropic) is OPT-IN per stage via SPEC_GENERATOR_MODEL_<STAGE>=
deepseek:deepseek-chat - route authoring stages (needs, srs, review) to a
strong provider for anything founder-facing: generic requirements are the
free competitors' product. Assembly stages don't appear here: deterministic,
no model.
"""
from __future__ import annotations

import hashlib
import json
import os
import re as _re
import sys
import threading
import time
from dataclasses import dataclass

_DOTENV: dict[str, str] | None = None


def _dotenv() -> dict[str, str]:
    """Config files, parsed once per process, most-shared first:

    1. <require-root>/.env - the SHARED config one directory up
       (vmodel-spec-app/scripts/switch-model.sh writes there; one file drives
       every process). Present only in the sibling-repo layout; absent in a
       standalone checkout and nothing changes.
    2. ./.env - repo-local overrides for the rare per-repo deviation.

    Real environment variables always win over both files (see _env), and
    os.environ is never mutated - no dependency, no side effects."""
    global _DOTENV
    if _DOTENV is not None:
        return _DOTENV
    _DOTENV = {}
    from pathlib import Path

    for p in (Path("..") / ".env", Path(".env")):
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            _DOTENV[key.strip()] = value.strip().strip('"').strip("'")
    return _DOTENV


def _env(key: str) -> str | None:
    """One setting lookup: the real environment wins, the repo-local .env
    fills the gap. Every external config read goes through here so a .env
    written by switch-model.sh works for API keys too, not just routing."""
    from .worker_runtime import current_runtime
    runtime=current_runtime()
    if runtime is not None:
        if key in runtime.settings:return runtime.settings[key]
        if key.startswith("VMODEL_CAP_") or key in ("SPEC_GENERATOR_FROM","SPEC_GENERATOR_BULK","VMODEL_REPLAY_DIR","SPEC_GENERATOR_LOG_RAW","VMODEL_LLM_DEBUG_HTTP"):
            return None
    return os.environ.get(key) or _dotenv().get(key)


# ---------------------------------------------------------------- guardrails
# Every LLM call passes through these limits. A hung ollama generation costs a
# hang at full GPU; the same hang on a paid API is a bill. All four knobs are
# env/.env-overridable; the defaults match "demo on a 16GB laptop, or cents".
_CALL_TIMEOUT_S = 600   # wall-clock budget per call, client retries included
_STALL_TIMEOUT_S = 180  # a stream this quiet is hung, not thinking
_MAX_CALLS = 200        # what ONE process run may spend, in LLM calls
_CALLS_USED = 0

# Eight times a same-stage running median catches a 10x runaway while leaving
# wide room for legitimate item-size variance. Four minutes protects cold
# starts and prevents one unusually quick early item from shrinking the limit.
from vmodel_contracts.runtime import AdaptiveCallWatchdog, AdaptiveWatchdogTimeout
_ADAPTIVE_WATCHDOG = AdaptiveCallWatchdog(multiplier=8.0, floor_s=240.0)


_MAX_TOKENS_ENV = "VMODEL_LLM_MAX_TOKENS"  # per-call generation ceiling override (unset: stage default)


def _max_tokens_limit(stage_default: int) -> int:
    """Per-call generation-length ceiling. Each stage ships its own default
    (StageModel.max_tokens); VMODEL_LLM_MAX_TOKENS overrides it globally - a
    looping generation hits a length ceiling, not only a clock ceiling."""
    raw = _env(_MAX_TOKENS_ENV)
    try:
        v = int(raw) if raw else stage_default
    except ValueError:
        v = stage_default
    return v if v > 0 else stage_default


def _truncation_reason(resp) -> str | None:
    meta = getattr(resp, "response_metadata", None) or {}
    for key in ("done_reason", "finish_reason", "stop_reason"):  # ollama / openai-compat / anthropic
        if meta.get(key) in ("length", "max_tokens"):
            return str(meta[key])
    return None


def _check_truncated(resp, stage: str, cfg) -> None:
    """A generation cut at the token ceiling is a failure, not an answer: name
    it here, or it surfaces downstream as a confusing parse/validation error."""
    if _truncation_reason(resp) is None:
        return
    from .errors import UserFacingError

    ceiling = _max_tokens_limit(cfg.max_tokens)
    _log_llm({"event": "llm_call_error", "provider": cfg.provider, "model": cfg.model, "stage": stage,
              "error": {"kind": "token_ceiling", "ceiling_tokens": ceiling}})
    raise UserFacingError(
        f"stage '{stage}': the {cfg.provider}/{cfg.model} call hit the generation-length "
        f"ceiling ({ceiling} tokens) and its output was cut off.",
        "the model had more to emit than one call allows - the payload is truncated.",
        f"{_rerun_guidance(stage)}. If it repeats, raise "
        f"{_MAX_TOKENS_ENV} only deliberately, or route the stage to another provider with "
        "vmodel-spec-app/scripts/switch-model.sh.",
    )


def _guard_limits() -> tuple[int, int, int]:
    def _num(name: str, default: int) -> int:
        raw = _env(name)
        try:
            v = int(raw) if raw else default
        except ValueError:
            v = default
        return v if v > 0 else default

    return (_num("VMODEL_LLM_CALL_TIMEOUT_S", _CALL_TIMEOUT_S),
            _num("VMODEL_LLM_STALL_TIMEOUT_S", _STALL_TIMEOUT_S),
            _num("VMODEL_LLM_MAX_CALLS", _MAX_CALLS))


def _rerun_guidance(stage: str) -> str:
    """The rerun instruction that is TRUE for this invocation. An explicit
    --from <stage> forces that stage to re-execute AND discards preserved
    per-item work downstream, so after a failure the right command DROPS
    --from and lets the checkpoint + per-item resume do their job (2026-09-10:
    the old blanket "rerun the same command" cost a user a forced re-run of
    every finished item)."""
    frm = _env("SPEC_GENERATOR_FROM")
    if frm:
        return (f"rerun the command WITHOUT '--from {frm}' ('--from' forces a re-run from '{frm}' and "
                "discards finished per-item work) - the checkpoint resumes at this stage and finished "
                "items are skipped")
    return "rerun the same command - the checkpoint resumes at this stage"


def _llm_log_sink() -> str | None:
    """VMODEL_LLM_LOG: unset = off (default, zero cost); 'stderr' = console;
    any other value = a file path, one JSON object per line."""
    from .worker_runtime import current_runtime
    runtime=current_runtime()
    return runtime.llm_log if runtime is not None else _env("VMODEL_LLM_LOG")


def _mask_secrets(value: str) -> str:
    return _re.sub(r"(sk-[A-Za-z0-9_-]{4})[A-Za-z0-9_-]+", r"\1...", str(value))


def raw_capture_enabled() -> bool:
    """SPEC_GENERATOR_LOG_RAW=1 (opt-in, off by default): append each raw
    completion to the LLM log as an llm_call_raw event. Completions can carry
    founder context, so this is a deliberate debugging act - it exists so a
    structural failure leaves a BYTE-EXACT artifact for a regression fixture
    (the 2026-09-11 phi4 architecture failure was unrecoverable as a fixture
    because only a 200-character snippet survived)."""
    return _env("SPEC_GENERATOR_LOG_RAW") == "1"


def log_raw_completion(stage: str, item: str | None, attempt: int, raw: str) -> None:
    from .stop_fence import guarded_publish
    return guarded_publish(lambda:_log_raw_completion(stage,item,attempt,raw))


def _log_raw_completion(stage: str, item: str | None, attempt: int, raw: str) -> None:
    """The only path that ever logs message content - gated on the explicit
    opt-in above, and written through the same sink as the metadata events."""
    from .stop_fence import check_execution
    check_execution()
    if raw_capture_enabled():
        _log_llm({"event": "llm_call_raw", "stage": stage,
                  **({"item": item} if item else {}),
                  "attempt": attempt, "raw_chars": len(raw), "raw": raw})



def call_diagnostic(stage: str, item: str | None, attempt: int) -> dict:
    cfg = for_stage(stage)
    return dict(_call_diagnostics().get((stage, item, attempt), {
        "attempt": attempt, "provider": cfg.provider, "model": cfg.model,
    }))


def capture_rejected_completion(stage: str, item: str | None, attempt: int,
                                raw: str, reason: str) -> str:
    from .stop_fence import guarded_publish
    return guarded_publish(lambda:_capture_rejected_completion(stage,item,attempt,raw,reason))


def _capture_rejected_completion(stage: str, item: str | None, attempt: int,
                                raw: str, reason: str) -> str:
    """Persist every rejected output byte-for-byte plus a readable manifest."""
    from .stop_fence import check_execution
    check_execution()
    from pathlib import Path
    sink = _llm_log_sink()
    root = Path(sink).expanduser().resolve().parent if sink and sink != "stderr" else Path.cwd()
    rejected = root / "rejected-llm-outputs"
    rejected.mkdir(parents=True, exist_ok=True)
    safe_item = _re.sub(r"[^A-Za-z0-9_.-]+", "-", item or "call").strip("-")
    safe_reason = _re.sub(r"[^A-Za-z0-9_.-]+", "-", reason).strip("-")[:60] or "rejected"
    stem = f"{stage}--{safe_item}--attempt-{attempt}--{safe_reason}"
    path = rejected / f"{stem}.raw.txt"
    suffix = 2
    while path.exists():
        path = rejected / f"{stem}--{suffix}.raw.txt"; suffix += 1
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(raw); os.replace(tmp, path)
    diag = call_diagnostic(stage, item, attempt)
    manifest = rejected / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"path": str(path), "stage": stage, "item": item,
                             "attempt": attempt, "reason": reason,
                             "raw_bytes": len(raw.encode()), **diag}, default=str) + "\n")
    _console_line(f"     - rejected model output saved: {path}")
    return str(path)


def capture_failed_completion(stage: str, item: str | None, raw: str) -> str:
    """Compatibility wrapper for callers that only know terminal failure."""
    return capture_rejected_completion(stage, item, 2, raw, "terminal-contract-failure")

def _log_llm(event: dict) -> None:
    """One JSON line per LLM call event. Never logs message content or keys -
    metadata, token usage, and provider error payloads only, plus raw
    completions when (and only when) SPEC_GENERATOR_LOG_RAW=1 opts in.
    Logging must never break a run."""
    sink = _llm_log_sink()
    if not sink:
        return
    line = json.dumps({**event, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, default=str)
    try:
        if sink == "stderr":
            _console_line(line)
        else:
            with open(sink, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError:
        pass


_CONSOLE_LOCK = threading.Lock()
_OUTPUT_MODE_ANNOUNCED: set[tuple[str, str]] = set()
_DEBUG_HTTP_ANNOUNCED: set[str] = set()
_CALL_DIAGNOSTICS: dict[tuple[str, str | None, int], dict] = {}


def _console_level(msg: str) -> str | None:
    """Conservative severity classification for shared terminal colors."""
    lower = msg.lower()
    if any(token in lower for token in ("error", "failed", "failure", "!!")):
        return "error"
    if any(token in lower for token in ("warning", "fallback", "retry", "still working", "simulated")):
        return "warning"
    if " ok " in f" {lower} " or "success" in lower or "approved" in lower:
        return "success"
    return None


def _console_line(msg: str) -> None:
    """One stderr line, atomically - the heartbeat thread and the progress
    printer share this lock so lines never interleave mid-call."""
    from .worker_runtime import current_runtime
    runtime=current_runtime()
    if runtime is not None:
        from .stop_fence import guarded_publish,StopRequested
        try:guarded_publish(lambda:runtime.console(msg))
        except StopRequested:pass
        return
    from vmodel_contracts.runtime import format_console_line

    with _CONSOLE_LOCK:
        sys.stderr.write(format_console_line(msg, level=_console_level(msg), stream=sys.stderr) + "\n")
        sys.stderr.flush()


def _debug_http_path() -> str | None:
    """Current payload-bearing debug sink, only when VMODEL_DEBUG_HTTP opts in."""
    from vmodel_contracts.runtime import debug_http_enabled, debug_http_path
    return str(debug_http_path(_llm_log_sink())) if debug_http_enabled() else None


def log_malformed_response(stage: str, kind: str, resp, *, snippet: str | None = None) -> None:
    """Make a malformed provider reply visible immediately, without prompts."""
    response_id = getattr(resp, "id", None) or (getattr(resp, "response_metadata", None) or {}).get("id") or "unknown"
    text = f"  ERROR malformed response [{stage}]: {kind}; response_id={response_id}"
    if snippet:
        text += f"; reply began: {snippet.strip()[:160]!r}"
    debug_path = _debug_http_path()
    if debug_path:
        text += f"; full HTTP exchange: {debug_path}"
    _console_line(text)


def _heartbeat_interval() -> float:
    """VMODEL_LLM_HEARTBEAT_S: seconds between 'still working' lines while a
    call runs without producing output. Default 30; 0 disables."""
    try:
        return float(_env("VMODEL_LLM_HEARTBEAT_S") or "30")
    except ValueError:
        return 30.0


def _start_heartbeat(label: str):
    """Print '  ... still working - <label>, Ns elapsed' every
    VMODEL_LLM_HEARTBEAT_S seconds until the returned stop() is called.
    Daemon thread; zero threads when disabled."""
    interval = _heartbeat_interval()
    if interval <= 0:
        return lambda: None
    t0 = time.monotonic()
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(interval):
            if stop.is_set():  # a beat already past the wait must not print after the call returned
                break
            elapsed=time.monotonic()-t0
            _console_line(f"  ... still working - {label}, {elapsed:.0f}s elapsed")
            from . import structured_progress
            from .worker_runtime import current_runtime
            runtime=current_runtime()
            sink=runtime.progress if runtime is not None else structured_progress.active_sink
            if sink is not None and sink._enabled:
                try:
                    from .stop_fence import guarded_publish
                    guarded_publish(lambda:sink.emit(label,label,"heartbeat",f"Model call still running, {elapsed:.0f}s elapsed"))
                except Exception:pass

    import contextvars
    context=contextvars.copy_context()
    threading.Thread(target=lambda:context.run(_beat), daemon=True).start()
    return stop.set


def _provider_error_detail(exc) -> dict:
    """Best-effort raw provider error for logs and the error taxonomy: HTTP
    status, error type/code/message, request id, retry-after. The openai SDK
    shape (openai, deepseek) carries .status_code/.body/.response; local
    ollama errors degrade to class + message. Secret-shaped strings masked."""
    out = {"class": type(exc).__name__}
    status = getattr(exc, "status_code", None)
    if status is not None:
        out["http_status"] = status
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for h, key in (("x-request-id", "request_id"), ("retry-after", "retry_after_s")):
        try:
            v = headers.get(h)
        except Exception:
            v = None
        if v:
            out[key] = v
    body = getattr(exc, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        for k in ("type", "code", "param", "message"):
            if err.get(k) is not None:
                out[k] = _mask_secrets(str(err[k]))[:600]
    if "message" not in out:
        out["message"] = _mask_secrets(str(exc))[:600]
    return out


def _usage_of(resp) -> dict:
    """Normalize LangChain usage and Ollama-native eval metadata."""
    usage = dict(getattr(resp, "usage_metadata", None) or {})
    meta = getattr(resp, "response_metadata", None) or {}
    aliases = {"prompt_eval_count": "input_tokens", "eval_count": "output_tokens"}
    for source, target in aliases.items():
        if meta.get(source) is not None and target not in usage:
            usage[target] = meta[source]
    if "total_tokens" not in usage and ("input_tokens" in usage or "output_tokens" in usage):
        usage["total_tokens"] = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
    for key in ("prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration"):
        if meta.get(key) is not None:
            usage[key] = meta[key]
    return usage


def _prompt_metrics(messages) -> dict:
    raw = "\n".join(str(getattr(m, "content", "")) for m in messages).encode()
    return {"prompt_bytes": len(raw), "prompt_sha256": hashlib.sha256(raw).hexdigest()[:16]}


def _guard_enter(stage: str) -> None:
    """One LLM call is starting: charge it against the run budget."""
    global _CALLS_USED
    from .worker_runtime import current_runtime
    runtime=current_runtime()
    if runtime is not None:
        with runtime.lock:
            runtime.calls+=1;calls=runtime.calls
    else:
        _CALLS_USED+=1;calls=_CALLS_USED
    if calls > _guard_limits()[2]:
        cfg = for_stage(stage)
        from .errors import UserFacingError

        raise UserFacingError(
            f"stage '{stage}': this run hit its LLM call budget ({_guard_limits()[2]} calls) and was stopped.",
            "the budget bounds what one run can cost and catches retry loops; the "
            f"{cfg.provider}/{cfg.model} calls before it all completed.",
            f"{_rerun_guidance(stage)}. If the run keeps "
            "hitting the budget a stage is looping: report it. Raise the ceiling with "
            "VMODEL_LLM_MAX_CALLS only deliberately.",
        )


def _guard_killed(stage: str, kind: str, waited: float, limit: int):
    cfg = for_stage(stage)
    from .errors import UserFacingError

    return UserFacingError(
        f"stage '{stage}': the {cfg.provider}/{cfg.model} call was killed - {kind} "
        f"(limit {limit}s, waited {waited:.0f}s).",
        "a runaway generation or a hung connection; the call was making no useful progress.",
        f"{_rerun_guidance(stage)}. If it repeats, check "
        "the provider is healthy (for ollama: ollama ps + server logs), raise "
        "VMODEL_LLM_CALL_TIMEOUT_S / VMODEL_LLM_STALL_TIMEOUT_S, or route the stage to another "
        "provider with vmodel-spec-app/scripts/switch-model.sh.",
    )


def _run_with_deadline(fn, timeout_s: int, stage: str):
    """Wall-clock deadline around one blocking call. The worker thread cannot
    be force-killed in Python, so the client ALSO gets the same timeout (see
    build_chat_model) - the socket dies, the thread unwinds, and the caller
    gets an informative error instead of a hang."""
    import concurrent.futures

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    import contextvars
    context=contextvars.copy_context()
    from .stop_fence import track_inflight_worker
    release = track_inflight_worker()
    def run():
        try: return context.run(fn)
        finally: release()
    try: future = pool.submit(run)
    except BaseException:
        release();pool.shutdown(wait=False);raise
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        if future.cancel(): release()
        raise TimeoutError from None
    finally:
        # Never let Executor.__exit__ turn a watchdog deadline back into a
        # blocking wait for the runaway worker. Provider socket timeouts below
        # are the actual cancellation rail; this is the Python-side escape.
        pool.shutdown(wait=False, cancel_futures=True)


def _set_model_timeout(chat_model, timeout_s: float) -> None:
    """Apply this call's adaptive deadline to each provider's live socket.

    LangChain wrappers expose different clients. Best effort is intentional:
    the outer wall-clock deadline still returns control for test doubles and
    future providers, while these known clients stop Ollama/paid generation.
    """
    model = getattr(chat_model, "runnable", chat_model)
    try:  # ChatOpenAI / DeepSeek / OpenAI-compatible Ollama
        model.request_timeout = timeout_s
        root = getattr(model, "root_client", None)
        if root is not None:
            root.timeout = timeout_s
            if getattr(root, "_client", None) is not None:
                import httpx
                root._client.timeout = httpx.Timeout(timeout_s)
    except Exception:
        pass
    try:  # native ChatOllama
        import httpx
        model._client._client.timeout = httpx.Timeout(timeout_s)
    except Exception:
        pass


@dataclass(frozen=True)
class StageModel:
    provider: str
    model: str
    temperature: float = 0.0  # ADR-0022: reproducibility - failures replay exactly
    max_tokens: int = 8192
    seed: int = 42            # ADR-0022: fixed seed (SPEC_GENERATOR_SEED overrides)


_STAGE_DEFAULTS: dict[str, StageModel] = {
    "needs":        StageModel("ollama", "qwen2.5-coder:14b"),
    "srs":          StageModel("ollama", "qwen2.5-coder:14b"),
    "architecture": StageModel("ollama", "qwen2.5-coder:14b"),
    "items":        StageModel("ollama", "qwen2.5-coder:14b"),
    "tests":        StageModel("ollama", "qwen2.5-coder:14b", max_tokens=12288),  # largest payload: truncation headroom
    "validation":   StageModel("ollama", "qwen2.5-coder:14b"),
    "review":       StageModel("ollama", "qwen2.5-coder:14b"),
    "expert_opinion": StageModel("ollama", "qwen2.5-coder:14b"),
}

_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": None,
    "ollama": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
}
_API_KEY_ENV = {"deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "ollama": None}


def for_stage(stage: str) -> StageModel:
    from .worker_runtime import current_runtime
    runtime=current_runtime()
    if runtime is not None:
        if stage not in runtime.models:raise KeyError(f'job model absent for {stage}')
        return runtime.models[stage]
    if stage not in _STAGE_DEFAULTS:
        raise KeyError(f"unknown stage {stage!r}")
    cfg = _STAGE_DEFAULTS[stage]
    seed = int(os.environ.get("SPEC_GENERATOR_SEED", str(cfg.seed)))
    override = _env(f"SPEC_GENERATOR_MODEL_{stage.upper()}")
    if override:
        provider, _, model = override.partition(":")
        return StageModel(provider, model, cfg.temperature, cfg.max_tokens, seed)
    return StageModel(cfg.provider, cfg.model, cfg.temperature, cfg.max_tokens, seed)


def bulk_fills_enabled() -> bool:
    """ADR-0022: per-slot micro-calls are the default for srs fills and item
    specs (a bad fill fails one cheap call, never a long composite payload).
    The old one-bulk-call-per-stage path stays behind SPEC_GENERATOR_BULK=1
    in case the wall-time trade proves bad on the founder's hardware."""
    return _env("SPEC_GENERATOR_BULK") == "1"


def _preflight(cfg: "StageModel", stage: str) -> None:
    from .errors import UserFacingError

    env = _API_KEY_ENV.get(cfg.provider)
    if env and not _env(env):
        raise UserFacingError(
            f"stage '{stage}' is routed to provider '{cfg.provider}' (model {cfg.model}), but {env} is not set.",
            "no API key in the environment.",
            f"export {env}=... - remote providers are opt-in (the default is local Ollama); "
            f"or route this stage back to local: SPEC_GENERATOR_MODEL_{stage.upper()}.=ollama:qwen2.5-coder:14b.",
        )


def describe_models() -> str:
    """What will actually run, resolved per stage after env/.env layering -
    printed at startup so the log always records which provider/model ran."""
    return ", ".join(f"{s}={for_stage(s).provider}/{for_stage(s).model}" for s in _STAGE_DEFAULTS)


def invoke_checked(chat_model, messages, stage: str, item: str | None = None):
    """Invoke under guardrails, retrying one provider-empty completion.

    DeepSeek documents occasional empty content in JSON mode. Treat that as a
    retryable provider signal, log the response envelope, and never let it
    masquerade as a downstream contract failure.
    """
    from .errors import UserFacingError, translate_llm_error
    from .stop_fence import check_execution
    check_execution()

    cfg = for_stage(stage)
    call_timeout = _guard_limits()[0]
    for attempt in (1, 2):
        check_execution()
        adaptive_deadline, stage_median = _watchdog().deadline(stage)
        effective_timeout = min(call_timeout, adaptive_deadline)
        _guard_enter(stage)
        base = {"provider": cfg.provider, "model": cfg.model, "stage": stage,
                "attempt": attempt, **({"item": item} if item else {}),
                **({"retry_reason": "empty_response"} if attempt == 2 else {})}
        _log_llm({"event": "llm_call_start", **base})
        stop_heartbeat = _start_heartbeat(f"stage '{stage}'" + (f" [{item}]" if item else "") + " call")
        t0 = time.monotonic()
        try:
            _set_model_timeout(chat_model, effective_timeout)
            resp = _run_with_deadline(lambda: __import__("spec_generator.attempt_admission",fromlist=["checked_model_invoke"]).checked_model_invoke(chat_model,messages,stage=stage,provider=cfg.provider,model=cfg.model,max_output_tokens=_max_tokens_limit(cfg.max_tokens)), effective_timeout, stage)
            elapsed = time.monotonic() - t0
            _watchdog().complete(stage, elapsed)
            _check_truncated(resp, stage, cfg)
            content = getattr(resp, "content", None)
            if isinstance(content, str) and not content.strip():
                log_malformed_response(stage, "empty content", resp)
                _log_llm({"event": "llm_call_error", **base, "latency_s": round(time.monotonic() - t0, 2),
                          "error": {"kind": "empty_response", "raw_payload": __import__("vmodel_contracts.runtime", fromlist=["empty_llm_response_detail"]).empty_llm_response_detail(resp, _mask_secrets)}})
                if attempt == 1:
                    continue
                guidance = (_rerun_guidance(stage) + "; DeepSeek documents occasional empty JSON-mode "
                            "completions, so a fresh call may succeed" if cfg.provider == "deepseek" else
                            _rerun_guidance(stage) + "; if it repeats, check the selected provider/model")
                raise UserFacingError(
                    f"stage '{stage}': '{cfg.provider}/{cfg.model}' returned empty content twice.",
                    "the provider completed both requests without an answer; the raw response envelopes "
                    "are recorded in llm-calls.jsonl.", guidance)
            _log_llm({"event": "llm_call_end", **base, "latency_s": round(time.monotonic() - t0, 2),
                      "usage": _usage_of(resp)})
            return resp
        except TimeoutError:
            elapsed = time.monotonic() - t0
            adaptive = effective_timeout < call_timeout or adaptive_deadline <= call_timeout
            _log_llm({"event": "llm_call_error", **base, "latency_s": round(elapsed, 2),
                      "error": {"kind": "adaptive_watchdog" if adaptive else "timeout", "limit_s": effective_timeout,
                                "stage_median_s": stage_median}})
            if adaptive and attempt == 1:
                ratio = elapsed / stage_median if stage_median else None
                basis = f"{ratio:.1f}x stage median" if ratio is not None else "cold-start floor"
                _console_line(f"  WARNING boundary=watchdog stage={stage} item={item or '(none)'} "
                              f"elapsed={elapsed:.1f}s median={stage_median if stage_median is not None else 'none'}s "
                              f"multiplier={getattr(_watchdog(), 'multiplier', 8):g} floor={getattr(_watchdog(), 'floor_s', 240):g}s "
                              f"absolute_cap={call_timeout}s deadline={effective_timeout:.1f}s "
                              f"decision=kill-retry reason={basis}")
                continue
            if adaptive:
                raise AdaptiveWatchdogTimeout(stage, elapsed, effective_timeout, stage_median) from None
            raise _guard_killed(stage, "no completion within the call budget", elapsed, call_timeout) from None
        except UserFacingError:
            raise
        except Exception as e:  # noqa: BLE001
            _log_llm({"event": "llm_call_error", **base, "latency_s": round(time.monotonic() - t0, 2),
                      "error": _provider_error_detail(e)})
            friendly = translate_llm_error(e, stage, cfg.provider, cfg.model, _BASE_URLS.get(cfg.provider))
            if friendly is not None:
                raise friendly from e
            raise
        finally:
            stop_heartbeat()


def stream_checked(chat_model, messages, stage: str, item: str | None = None,
                   attempt: int = 1, retry_reason: str | None = None):
    """Stream the same call invoke_checked would make, under the guardrails:
    the whole call has a wall-clock budget, and a MID-STREAM stall limit (a
    stream that goes quiet between chunks is hung). Silence BEFORE the first
    chunk is prefill - healthy prompt evaluation, potentially minutes on a
    local model with a large context (2026-09-10: a T-1 fill died to the
    stall guard at 180s while ollama was happily prefilling SRS+ITEMS) - so
    only the wall-clock budget bounds time-to-first-chunk. The concatenated
    chunk contents are the completion itself - the output contract is
    unchanged, the consumer just sees tokens as they land. Logged like
    invoke_checked."""
    import queue
    import threading

    from .errors import UserFacingError, translate_llm_error
    from .stop_fence import check_execution
    check_execution()

    cfg = for_stage(stage)
    _guard_enter(stage)
    call_timeout, stall_timeout, _ = _guard_limits()
    adaptive_deadline, stage_median = _watchdog().deadline(stage)
    effective_timeout = min(call_timeout, adaptive_deadline)
    base = {"provider": cfg.provider, "model": cfg.model, "stage": stage,
            "attempt": attempt, **_prompt_metrics(messages),
            **({"item": item} if item else {}),
            **({"retry_reason": retry_reason} if retry_reason else {})}
    q: queue.Queue = queue.Queue()

    def _pump() -> None:
        try:
            for chunk in __import__("spec_generator.attempt_admission",fromlist=["checked_model_stream"]).checked_model_stream(chat_model,messages,stage=stage,provider=cfg.provider,model=cfg.model,max_output_tokens=_max_tokens_limit(cfg.max_tokens)):
                q.put(("chunk", chunk))
            q.put(("done", None))
        except Exception as e:  # noqa: BLE001 - delivered to the consumer below
            q.put(("error", e))

    _set_model_timeout(chat_model, effective_timeout)
    _log_llm({"event": "llm_call_start", **base})
    stop_heartbeat = _start_heartbeat(f"stage '{stage}'" + (f" [{item}]" if item else "") + " call")
    t0 = time.monotonic()
    import contextvars
    stream_context=contextvars.copy_context()
    from .stop_fence import track_inflight_worker
    release = track_inflight_worker()
    def run_pump():
        try: stream_context.run(_pump)
        finally: release()
    try: threading.Thread(target=run_pump, daemon=True).start()
    except BaseException:
        release();raise
    last = None  # final chunk carries the truncation metadata (done_reason/finish_reason)
    first_chunk_seen = False  # prefill silence is healthy; only mid-stream quiet is a stall
    accumulated_usage: dict = {}
    accumulated_meta: dict = {}
    ttft_s: float | None = None  # time to first token - the prefill cost, made visible
    try:
        while True:
            elapsed = time.monotonic() - t0
            if elapsed > effective_timeout:
                adaptive = adaptive_deadline <= call_timeout
                _log_llm({"event": "llm_call_error", **base, "latency_s": round(elapsed, 2),
                          "error": {"kind": "adaptive_watchdog" if adaptive else "timeout",
                                    "limit_s": effective_timeout, "stage_median_s": stage_median}})
                if adaptive:
                    raise AdaptiveWatchdogTimeout(stage, elapsed, effective_timeout, stage_median)
                raise _guard_killed(stage, "no completion within the call budget", elapsed, call_timeout)
            wait = stall_timeout if first_chunk_seen else min(5.0, effective_timeout - elapsed)
            try:
                kind, payload = q.get(timeout=max(0.05, min(wait, effective_timeout - elapsed)))
            except queue.Empty:
                if time.monotonic() - t0 > effective_timeout:
                    _log_llm({"event": "llm_call_error", **base, "latency_s": round(time.monotonic() - t0, 2),
                              "error": {"kind": "adaptive_watchdog", "limit_s": effective_timeout, "stage_median_s": stage_median}})
                    raise AdaptiveWatchdogTimeout(stage, time.monotonic() - t0,
                                                        effective_timeout, stage_median) from None
                if first_chunk_seen:
                    _log_llm({"event": "llm_call_error", **base, "latency_s": round(time.monotonic() - t0, 2),
                              "error": {"kind": "stall", "limit_s": stall_timeout}})
                    raise _guard_killed(stage, "the stream stalled mid-call (output stopped after starting)",
                                        stall_timeout, stall_timeout) from None
                continue  # prefill: no chunks yet, still inside the wall budget
            if kind == "chunk":
                if not first_chunk_seen:
                    ttft_s = round(time.monotonic() - t0, 2)
                first_chunk_seen = True
                last = payload
                chunk_usage = dict(getattr(payload, "usage_metadata", None) or {})
                if chunk_usage:
                    accumulated_usage.update(chunk_usage)
                chunk_meta = dict(getattr(payload, "response_metadata", None) or {})
                if chunk_meta:
                    accumulated_meta.update(chunk_meta)
                yield payload
            elif kind == "done":
                _check_truncated(last, stage, cfg)
                usage = dict(accumulated_usage)
                aliases = {"prompt_eval_count": "input_tokens", "eval_count": "output_tokens"}
                for source, target in aliases.items():
                    if accumulated_meta.get(source) is not None and target not in usage:
                        usage[target] = accumulated_meta[source]
                if "total_tokens" not in usage and ("input_tokens" in usage or "output_tokens" in usage):
                    usage["total_tokens"] = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
                for key in ("prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration"):
                    if accumulated_meta.get(key) is not None:
                        usage[key] = accumulated_meta[key]
                if not usage:
                    usage = _usage_of(last)
                durations = {k: usage.get(k) for k in ("prompt_eval_duration", "eval_duration")
                             if usage.get(k) is not None}
                elapsed = time.monotonic() - t0
                _watchdog().complete(stage, elapsed)
                _call_diagnostics()[(stage, item, attempt)] = {
                    "attempt": attempt, "duration_s": round(elapsed, 2),
                    "tokens": usage.get("total_tokens", usage.get("output_tokens", 0)),
                    "provider": cfg.provider, "model": cfg.model,
                }
                _log_llm({"event": "llm_call_end", **base, "latency_s": round(elapsed, 2),
                          **({"ttft_s": ttft_s} if ttft_s is not None else {}),
                          "usage": usage, **durations})
                return
            else:
                _log_llm({"event": "llm_call_error", **base, "latency_s": round(time.monotonic() - t0, 2),
                          "error": _provider_error_detail(payload)})
                raise payload
    except UserFacingError:
        raise
    except Exception as e:  # noqa: BLE001
        friendly = translate_llm_error(e, stage, cfg.provider, cfg.model, _BASE_URLS.get(cfg.provider))
        if friendly is not None:
            raise friendly from e
        raise
    finally:
        stop_heartbeat()


_OLLAMA_STRUCTURED_MIN = (0, 5, 0)
_OLLAMA_CAPABILITY_CACHE: dict[str, tuple[bool, str]] = {}


def _ollama_structured_support(base_url: str | None = None) -> tuple[bool, str]:
    """Ask the running Ollama server whether JSON-schema output is supported.

    Structured outputs shipped in Ollama 0.5.0. Detection is cached per server
    for this process, but the selected mode is still printed for every model
    call so a constrained/fallback difference is never silent.
    """
    import urllib.request

    root = (base_url or _BASE_URLS["ollama"]).removesuffix("/v1").rstrip("/")
    if root in _OLLAMA_CAPABILITY_CACHE:
        return _OLLAMA_CAPABILITY_CACHE[root]
    try:
        with urllib.request.urlopen(root + "/api/version", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        raw = str(payload.get("version") or "")
        match = _re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", raw)
        if not match:
            result = (False, f"server returned unrecognized version {raw!r}")
        else:
            version = tuple(int(x or 0) for x in match.groups())
            result = (version >= _OLLAMA_STRUCTURED_MIN,
                      f"Ollama {raw}" + (" supports JSON schema" if version >= _OLLAMA_STRUCTURED_MIN
                                          else " is older than 0.5.0"))
    except Exception as exc:  # detection must not turn an old server into a startup failure
        result = (False, f"support check failed: {type(exc).__name__}")
    _OLLAMA_CAPABILITY_CACHE[root] = result
    return result


def _report_output_mode(stage: str, constrained: bool, reason: str) -> None:
    """The expected constrained path is quiet; fallback is a visible warning."""
    if not constrained:
        _console_line(f"  WARNING output mode [{stage}]: fallback (unconstrained) - {reason}")


def _structured_enabled() -> bool:
    """Constrained decoding (schema-enforced output) is the default for local
    Ollama; SPEC_GENERATOR_STRUCTURED=0 opts out (older Ollama < 0.5, or a
    quality regression the founder wants off)."""
    return _env("SPEC_GENERATOR_STRUCTURED") != "0"


def _ollama_keep_alive() -> str:
    """Pin the model between calls: VMODEL_OLLAMA_KEEP_ALIVE, default 30m.
    Without it the idle timer (5m default) can unload the model out from
    under a long in-flight request - and a model stuck 'Stopping...' hangs
    the client until the wall-clock budget (ollama#14364/#12179/#8178;
    2026-09-10: T-1 hung 6+ minutes while `ollama ps` showed 'Stopping...')."""
    return _env("VMODEL_OLLAMA_KEEP_ALIVE") or "30m"


def _ollama_num_ctx() -> int:
    """Explicit context window for ollama calls, default 32768. Ollama's
    server default is 4096 and it silently TRUNCATES prompts past that
    (2026-09-10: `ollama ps` showed CONTEXT 4096 while stage prompts run
    larger - the model never saw whole requirements, which surfaced as
    contract failures/retries). Override with VMODEL_OLLAMA_NUM_CTX.
    Note: the native /api/chat path honors this per request; the
    OpenAI-compat endpoint only forwards num_ctx once ollama PR #16825
    lands - until then the server-side lever for that path is
    OLLAMA_CONTEXT_LENGTH=32768 on `ollama serve`."""
    try:
        return int(_env("VMODEL_OLLAMA_NUM_CTX") or "32768")
    except ValueError:
        return 32768


def _ollama_think() -> bool:
    """qwen3-class ollama models think by default - minutes of silent <think>
    before any output, dead time for structured generation (2026-09-10: a
    qwen3:8b items-plan call sat silent 4+ minutes and read as hung). Off by
    default; VMODEL_OLLAMA_THINK=1 restores provider-default thinking."""
    return _env("VMODEL_OLLAMA_THINK") == "1"



def _tool_payload_schema(schema):
    """Preserve document-local refs when embedding a schema in tool payload.

    Fragment pointers resolve from the tool-parameter document root, not the
    nested payload. Validate before sending and never mutate the contract schema.
    """
    import copy
    result = copy.deepcopy(schema)
    def walk(node):
        if isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                if ref != "#" and not ref.startswith("#/"):
                    raise ValueError("tool schema requires local JSON pointer references")
                target = schema
                try:
                    for part in ref[2:].split("/") if ref != "#" else []:
                        part = part.replace("~1", "/").replace("~0", "~")
                        target = target[int(part)] if isinstance(target, list) else target[part]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise ValueError("unresolved tool schema reference") from exc
                node["$ref"] = "#/properties/payload" + ref[1:]
            for value in node.values():
                walk(value)
    walk(result)
    return result


class _DeepSeekToolOutputAdapter:
    """Expose one forced DeepSeek function call as ordinary JSON content.

    Function parameters must be an object, while our artifact schemas can also
    be arrays, so every schema is wrapped in a code-owned ``payload`` property.
    The rest of the pipeline continues to parse and validate plain JSON.
    """
    def __init__(self, model, schema: dict, name: str):
        self.name = name
        tool = {"type": "function", "function": {"name": name,
                "description": "Return the requested structured artifact.",
                "parameters": {"type": "object", "properties": {"payload": _tool_payload_schema(schema)},
                               "required": ["payload"], "additionalProperties": False}}}
        self.model = model
        self.tool = tool
        self.runnable = model.bind_tools([tool], tool_choice=name)

    def __getattr__(self, name):
        # Preserve observability/test access to the underlying transport.
        return getattr(self.model, name)

    def _content(self, message):
        import json
        calls = getattr(message, "tool_calls", None) or []
        if len(calls) != 1 or calls[0].get("name") != self.name:
            # Empty content enters the existing bounded human-readable retry;
            # never disguise a missing/wrong tool call as a valid artifact.
            return ""
        args = calls[0].get("args") or {}
        if "payload" not in args:
            return ""
        return json.dumps(args["payload"])

    def invoke(self, messages):
        message = self.runnable.invoke(messages)
        message.content = self._content(message)
        return message

    def stream(self, messages):
        # LangChain assembles streamed tool-call argument fragments via chunk
        # addition. Emit one normalized final chunk so downstream sees exactly
        # one JSON artifact, never partial function syntax.
        from langchain_core.messages import AIMessageChunk
        combined = None
        for chunk in self.runnable.stream(messages):
            combined = chunk if combined is None else combined + chunk
        if combined is None:
            yield AIMessageChunk(content="")
            return
        yield AIMessageChunk(content=self._content(combined),
                             response_metadata=getattr(combined, "response_metadata", {}) or {},
                             usage_metadata=getattr(combined, "usage_metadata", None))

def build_chat_model(stage: str, structured: bool = True, schema: dict | None = None):
    """Build a live model, or a recorded-response replay when explicitly set.

    structured=False builds WITHOUT the grammar format for tool calling.
    schema= overrides the stage's static per-skeleton grammar.
    """
    replay = _env("VMODEL_REPLAY_DIR")
    if replay:
        from .replay import ReplayChatModel
        return ReplayChatModel(stage, replay)
    cfg = for_stage(stage)
    from .wire_binding import validate_provider
    validate_provider(cfg.provider)
    _preflight(cfg, stage)
    if cfg.provider == "ollama" and structured and _structured_enabled():
        supported, reason = _ollama_structured_support()
        if supported:
            from langchain_ollama import ChatOllama

            from .schemas import for_stage as _schema_for_stage

            # native /api/chat: format receives the exact per-call schema.
            # This enforces shape only; parsing, semantic validation, and the
            # existing contract retry remain the safety net.
            selected_schema = schema or _schema_for_stage(stage)
            _report_output_mode(stage, True, reason)
            return ChatOllama(model=cfg.model, temperature=cfg.temperature,
                              reasoning=_ollama_think(), num_ctx=_ollama_num_ctx(),
                              keep_alive=_ollama_keep_alive(),
                timeout=_guard_limits()[0],
            **({"max_retries":0} if os.environ.get("VMODEL_TENANT_OWNER") is not None or os.environ.get("INTERVIEW_ENGINE_TENANTS")=="1" else {}),  # socket dies with guardrail deadline
                              seed=cfg.seed,  # ADR-0022
                              num_predict=_max_tokens_limit(cfg.max_tokens),
                              base_url=_BASE_URLS["ollama"].removesuffix("/v1"),
                              format=selected_schema)
        _report_output_mode(stage, False, reason)
    elif cfg.provider == "ollama" and structured:
        _report_output_mode(stage, False, "SPEC_GENERATOR_STRUCTURED=0")
    if cfg.provider in ("deepseek", "openai", "ollama"):
        from langchain_openai import ChatOpenAI

        env = _API_KEY_ENV[cfg.provider]
        extra = {}
        deepseek_tool_schema = schema if cfg.provider == "deepseek" and structured and schema is not None else None
        if cfg.provider == "deepseek":
            extra["extra_body"] = {"thinking": {"type": "disabled"}}
        if cfg.provider == "deepseek" and structured:
            announcement = ("deepseek", "forced-tool")
            if announcement not in _OUTPUT_MODE_ANNOUNCED:
                _OUTPUT_MODE_ANNOUNCED.add(announcement)
                _console_line("  output mode: DeepSeek forced function call; field shape provider-constrained, semantic validation+retry retained")
        if cfg.provider == "ollama":
            body = {"num_ctx": _ollama_num_ctx(),  # honored once ollama PR #16825 lands
                    "keep_alive": _ollama_keep_alive()}
            if not _ollama_think():
                body["think"] = False
            extra = {"extra_body": body}
        from .wire_binding import model_clients,register_model
        installed_clients=model_clients(stage=stage,provider=cfg.provider,model=cfg.model)
        if installed_clients is not None:extra.update(installed_clients)
        if cfg.provider in ("deepseek", "openai") and installed_clients is None:
            from vmodel_contracts.runtime import debug_http_client
            client = debug_http_client(llm_log=_llm_log_sink(), mask=_mask_secrets)
            if client is not None:
                extra["http_client"] = client
                debug_path = _debug_http_path()
                if debug_path not in _DEBUG_HTTP_ANNOUNCED:
                    _DEBUG_HTTP_ANNOUNCED.add(debug_path)
                    _console_line(f"  WARNING full LLM HTTP debug enabled: {debug_path} (request messages included; no authorization headers)")
        model = ChatOpenAI(model=cfg.model, temperature=cfg.temperature,
            timeout=_guard_limits()[0],
            **({"max_retries":0} if installed_clients is not None or os.environ.get("VMODEL_TENANT_OWNER") is not None or os.environ.get("INTERVIEW_ENGINE_TENANTS")=="1" else {}), seed=cfg.seed,
                          max_tokens=_max_tokens_limit(cfg.max_tokens),
                          base_url=_BASE_URLS[cfg.provider],
                          api_key=(_env(env) or "" if env else "") or "ollama", **extra)
        if deepseek_tool_schema is not None:
            model=_DeepSeekToolOutputAdapter(model, deepseek_tool_schema, f"return_{stage}_artifact")
        return register_model(model,stage=stage,provider=cfg.provider,model=cfg.model)
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # ADR-0022 note: anthropic exposes no seed parameter - temperature 0
        # is the only reproducibility lever on that provider (documented in the ADR).
        return ChatAnthropic(model=cfg.model, temperature=cfg.temperature,
            timeout=_guard_limits()[0],
            **({"max_retries":0} if os.environ.get("VMODEL_TENANT_OWNER") is not None or os.environ.get("INTERVIEW_ENGINE_TENANTS")=="1" else {}), max_tokens=_max_tokens_limit(cfg.max_tokens),
                             api_key=_env(_API_KEY_ENV["anthropic"]) or "")
    raise ValueError(cfg.provider)




def _watchdog():
    from .worker_runtime import current_runtime
    runtime=current_runtime()
    return runtime.watchdog if runtime is not None else _ADAPTIVE_WATCHDOG


def _call_diagnostics():
    from .worker_runtime import current_runtime
    runtime=current_runtime()
    return runtime.diagnostics if runtime is not None else _CALL_DIAGNOSTICS
