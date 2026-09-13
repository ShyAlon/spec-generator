"""experiment_report: the MEDIUM-tier quality signal (his 2026-09-13
three-speed design). Reads a bundle dir's telemetry - llm-calls.jsonl
(VMODEL_LLM_LOG), usage.jsonl, and the stage artifacts - and prints the
per-item quality metrics that make two experiment runs comparable: call
counts, first-pass rate, contract retries, provider errors, latency, tokens,
cost, and the entity counts the run produced. `python -m
spec_generator.experiment_report <bundle_dir> [--json]`."""
import json
import sys
from collections import defaultdict
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def build_report(bundle_dir: Path) -> dict:
    calls = _jsonl(bundle_dir / "llm-calls.jsonl")
    usage = _jsonl(bundle_dir / "usage.jsonl")

    items: dict[tuple, dict] = {}
    stage_calls = defaultdict(int)
    stage_latency = defaultdict(float)
    for e in calls:
        if e.get("event") != "llm_call_end":
            continue
        stage = e.get("stage", "?")
        stage_calls[stage] += 1
        stage_latency[stage] += float(e.get("latency_s") or 0)
        key = (stage, e.get("item") or "(call)")
        it = items.setdefault(key, {"stage": stage, "item": key[1], "attempts": 0,
                                    "contract_retries": 0, "latency_s": 0.0})
        it["attempts"] = max(it["attempts"], int(e.get("attempt") or 1))
        if e.get("retry_reason") == "contract":
            it["contract_retries"] += 1
        it["latency_s"] += float(e.get("latency_s") or 0)
    provider_errors = [e for e in calls if e.get("event") == "llm_call_error"]

    item_list = sorted(items.values(), key=lambda i: (i["stage"], i["item"]))
    first_pass = sum(1 for i in item_list if i["attempts"] <= 1)
    retried = [i for i in item_list if i["attempts"] > 1 or i["contract_retries"]]

    counts = {}
    for stage in ("needs", "srs", "architecture", "items", "tests"):
        f = bundle_dir / f"{stage}.json"
        if not f.exists():
            continue
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if stage == "needs":
            counts["needs"] = len(d.get("needs") or [])
        elif stage == "srs":
            counts["requirements"] = sum(len(d.get(sec) or [])
                                         for sec in ("functional", "non_functional"))
            counts["acceptance_criteria"] = sum(
                len(r.get("acceptance") or r.get("acceptance_criteria") or [])
                for sec in ("functional", "non_functional") for r in (d.get(sec) or []))
        elif stage == "architecture":
            counts["stack_entries"] = len(d.get("stack_entries") or [])
            counts["components"] = len(d.get("components") or [])
            counts["decisions"] = len(d.get("decisions") or [])
        elif stage == "items":
            counts["specs"] = len(d if isinstance(d, list) else d.get("items") or [])
        elif stage == "tests":
            counts["tests"] = len(d if isinstance(d, list) else d.get("tests") or [])

    return {
        "bundle_dir": str(bundle_dir),
        "stages": {s: {"calls": stage_calls[s], "llm_seconds": round(stage_latency[s], 1)}
                   for s in sorted(stage_calls)},
        "items": {"total": len(item_list), "first_pass": first_pass,
                  "first_pass_rate": round(first_pass / len(item_list), 3) if item_list else None,
                  "retried": retried},
        "provider_errors": [{"stage": e.get("stage"), "item": e.get("item"),
                             "error": (e.get("error") or {}).get("kind",
                                   (e.get("error") or {}).get("type", "?"))}
                            for e in provider_errors],
        "tokens": {"prompt": sum(int(u.get("prompt_tokens") or 0) for u in usage),
                   "completion": sum(int(u.get("completion_tokens") or 0) for u in usage)},
        "cost_usd": round(sum(float(u.get("prompt_tokens") or 0) * float(u.get("price_per_mtok_prompt_usd") or 0)
                              + float(u.get("completion_tokens") or 0) * float(u.get("price_per_mtok_completion_usd") or 0)
                              for u in usage) / 1_000_000, 4),
        "counts": counts,
    }


def render(rep: dict) -> str:
    lines = ["== experiment report ==", f"bundle: {rep['bundle_dir']}"]
    if rep["counts"]:
        lines.append("entities: " + ", ".join(f"{k}={v}" for k, v in rep["counts"].items()))
    for stage, s in rep["stages"].items():
        lines.append(f"  {stage}: {s['calls']} calls, {s['llm_seconds']}s in model")
    it = rep["items"]
    rate = f"{it['first_pass_rate'] * 100:.0f}%" if it["first_pass_rate"] is not None else "n/a"
    lines.append(f"items: {it['total']} | first-pass: {it['first_pass']} ({rate}) | "
                 f"retried: {len(it['retried'])} | provider errors: {len(rep['provider_errors'])}")
    for i in it["retried"]:
        lines.append(f"  RETRY {i['stage']}:{i['item']} - {i['attempts']} attempts "
                     f"({i['contract_retries']} contract), {i['latency_s']:.1f}s")
    for e in rep["provider_errors"]:
        lines.append(f"  ERROR {e['stage']}:{e.get('item') or '(call)'} - {e['error']}")
    t = rep["tokens"]
    lines.append(f"tokens: {t['prompt']} in / {t['completion']} out | cost: ${rep['cost_usd']:.4f}")
    lines.append("compare runs on: first-pass rate, retries per stage, llm seconds per stage, cost.")
    return "\n".join(lines)


def main() -> None:
    args = [a for a in sys.argv[1:]]
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    if not args:
        print("usage: python -m spec_generator.experiment_report <bundle_dir> [--json]",
              file=sys.stderr)
        sys.exit(1)
    bundle = Path(args[0])
    if not bundle.is_dir():
        print(f"error: no bundle directory at {bundle}", file=sys.stderr)
        sys.exit(1)
    rep = build_report(bundle)
    print(json.dumps(rep, indent=2) if as_json else render(rep))


if __name__ == "__main__":
    main()
