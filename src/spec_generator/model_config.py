"""Per-stage model routing for generation (model-agnostic layer, ADR-0003).

Authoring stages (needs, requirements, review) keep a strong model - generic
requirements are what free competitors produce. Mechanical rephrasing can drop
a tier. Assembly stages don't appear here: they are deterministic, no model.
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
    "needs":        StageModel("deepseek", "deepseek-chat"),
    "requirements": StageModel("deepseek", "deepseek-chat"),
    "architecture": StageModel("deepseek", "deepseek-chat"),
    "items":        StageModel("deepseek", "deepseek-chat"),
    "tests":        StageModel("deepseek", "deepseek-chat"),
    "review":       StageModel("deepseek", "deepseek-chat"),
}

_BASE_URLS = {"deepseek": "https://api.deepseek.com", "openai": None}
_API_KEY_ENV = {"deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def for_stage(stage: str) -> StageModel:
    if stage not in _STAGE_DEFAULTS:
        raise KeyError(f"unknown stage {stage!r}")
    cfg = _STAGE_DEFAULTS[stage]
    override = os.environ.get(f"SPEC_GENERATOR_MODEL_{stage.upper()}")
    if override:
        provider, _, model = override.partition(":")
        return StageModel(provider, model, cfg.temperature, cfg.max_tokens)
    return cfg


def build_chat_model(stage: str):
    cfg = for_stage(stage)
    if cfg.provider in ("deepseek", "openai"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                          base_url=_BASE_URLS[cfg.provider],
                          api_key=os.environ.get(_API_KEY_ENV[cfg.provider], ""))
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                             api_key=os.environ.get(_API_KEY_ENV["anthropic"], ""))
    raise ValueError(cfg.provider)
