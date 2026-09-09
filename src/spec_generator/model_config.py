"""Per-stage model routing for generation (model-agnostic layer, ADR-0003).
LOCAL-FIRST: every stage defaults to Ollama ($0, offline, no key). DeepSeek
(or openai/anthropic) is OPT-IN per stage via SPEC_GENERATOR_MODEL_<STAGE>=
deepseek:deepseek-chat - route authoring stages (needs, srs, review) to a
strong provider for anything founder-facing: generic requirements are the
free competitors' product. Assembly stages don't appear here: deterministic,
no model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class StageModel:
    provider: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 8192


_STAGE_DEFAULTS: dict[str, StageModel] = {
    "needs":        StageModel("ollama", "qwen2.5-coder:14b"),
    "srs":          StageModel("ollama", "qwen2.5-coder:14b"),
    "architecture": StageModel("ollama", "qwen2.5-coder:14b"),
    "items":        StageModel("ollama", "qwen2.5-coder:14b"),
    "tests":        StageModel("ollama", "qwen2.5-coder:14b"),
    "review":       StageModel("ollama", "qwen2.5-coder:14b"),
}

_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": None,
    "ollama": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
}
_API_KEY_ENV = {"deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "ollama": None}


def for_stage(stage: str) -> StageModel:
    if stage not in _STAGE_DEFAULTS:
        raise KeyError(f"unknown stage {stage!r}")
    cfg = _STAGE_DEFAULTS[stage]
    override = os.environ.get(f"SPEC_GENERATOR_MODEL_{stage.upper()}")
    if override:
        provider, _, model = override.partition(":")
        return StageModel(provider, model, cfg.temperature, cfg.max_tokens)
    return cfg


def _preflight(cfg: "StageModel", stage: str) -> None:
    from .errors import UserFacingError

    env = _API_KEY_ENV.get(cfg.provider)
    if env and not os.environ.get(env):
        raise UserFacingError(
            f"stage '{stage}' is routed to provider '{cfg.provider}' (model {cfg.model}), but {env} is not set.",
            "no API key in the environment.",
            f"export {env}=... - remote providers are opt-in (the default is local Ollama); "
            f"or route this stage back to local: SPEC_GENERATOR_MODEL_{stage.upper()}.=ollama:qwen2.5-coder:14b.",
        )


def invoke_checked(chat_model, messages, stage: str):
    from .errors import translate_llm_error

    cfg = for_stage(stage)
    try:
        return chat_model.invoke(messages)
    except Exception as e:  # noqa: BLE001
        friendly = translate_llm_error(e, stage, cfg.provider, cfg.model, _BASE_URLS.get(cfg.provider))
        if friendly is not None:
            raise friendly from e
        raise


def build_chat_model(stage: str):
    cfg = for_stage(stage)
    _preflight(cfg, stage)
    if cfg.provider in ("deepseek", "openai", "ollama"):
        from langchain_openai import ChatOpenAI

        env = _API_KEY_ENV[cfg.provider]
        return ChatOpenAI(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                          base_url=_BASE_URLS[cfg.provider],
                          api_key=(os.environ.get(env, "") if env else "") or "ollama")
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                             api_key=os.environ.get(_API_KEY_ENV["anthropic"], ""))
    raise ValueError(cfg.provider)
