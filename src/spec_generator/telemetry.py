"""Every call logged against the versioned price table - THE shared one in
vmodel-contracts (one table for every repo, 2026-09-12; the local copy had
silently zeroed unpriced remote models where interview-engine crashed).
Names re-exported so existing imports keep working."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from vmodel_contracts import StageTokenUsage
from vmodel_contracts import (PRICE_TABLE_VERSION, PRICES_USD_PER_MTOK,
                              price_for)

Sink = Callable[[StageTokenUsage], None]


def jsonl_sink(path: str | Path) -> Sink:
    def _write(u: StageTokenUsage) -> None:
        from .document_runtime import configured,append
        if configured():return append(path,(u.model_dump_json()+"\n").encode())
        with open(path, "a") as f:
            f.write(u.model_dump_json() + "\n")
    return _write


class UsageLog:
    def __init__(self, session_id: str, sink: Sink | None = None):
        self.session_id = session_id
        self.records: list[StageTokenUsage] = []
        self.sink = sink

    def record(self, stage: str, provider: str, model: str, usage: dict) -> StageTokenUsage:
        pin, pout = price_for(model, provider)  # local $0; unpriced remote fails loudly
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

