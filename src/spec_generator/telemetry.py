"""Same discipline as interview-engine: every call logged against the
versioned price table. Kept as a tiny local copy of the contract types rather
than a shared non-contract module - the only shared artifact is
vmodel-contracts (ADR-0002). If this grows, it graduates INTO the contracts
package as telemetry helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from vmodel_contracts import StageTokenUsage

PRICE_TABLE_VERSION = "2026-09-10"
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "deepseek-flash": (0.30, 1.20),       # V4.1 Flash peak; off-peak $0.15/$0.60
    "deepseek-v4-pro": (1.32, 3.96),      # peak; off-peak $0.66/$1.98; retires to Flash 2026-09-14
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4.5": (1.00, 5.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
}

Sink = Callable[[StageTokenUsage], None]


def jsonl_sink(path: str | Path) -> Sink:
    def _write(u: StageTokenUsage) -> None:
        with open(path, "a") as f:
            f.write(u.model_dump_json() + "\n")
    return _write


class UsageLog:
    def __init__(self, session_id: str, sink: Sink | None = None):
        self.session_id = session_id
        self.records: list[StageTokenUsage] = []
        self.sink = sink

    def record(self, stage: str, provider: str, model: str, usage: dict) -> StageTokenUsage:
        if provider == "ollama":
            pin, pout = 0.0, 0.0  # local compute: $0 marginal, logged for the record
        else:
            pin, pout = PRICES_USD_PER_MTOK.get(model, (0.0, 0.0))
        rec = StageTokenUsage(
            session_id=self.session_id, stage=stage, provider=provider, model=model,
            prompt_tokens=int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
            completion_tokens=int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
            cached_tokens=int((usage.get("input_token_details") or {}).get("cache_read", 0) or 0),
            price_per_mtok_prompt_usd=pin, price_per_mtok_completion_usd=pout,
        )
        self.records.append(rec)
        if self.sink:
            self.sink(rec)
        return rec

    def total_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

