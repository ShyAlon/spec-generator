"""A/B probe: schema-constrained vs free generation on YOUR Ollama.

Runs a stage twice (with and without the contracts-derived schema) and
reports validity, wall time, and both outputs for a quality eyeball. Grammar
constraints can dull small models - look at the need statements, not just the
validity line.

The probe mirrors production exactly: same prompt assembly, same temperature,
same per-stage token budget (model_config), same tolerant JSON extraction
(graph._extract_json) - so a fenced or prose-wrapped reply counts the same
way it counts in a real run, and a truncation here means a truncation there.

Usage:
    python scripts/probe_structured.py <path-to-intent_model.json> [stage]
Stages: needs (default) | tests   (needs is fast; tests is the big payload)
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:11434"

sys.path.insert(0, "src")
from spec_generator import model_config                      # noqa: E402
from spec_generator.graph import _extract_json               # noqa: E402
from spec_generator.prompts import METHODOLOGY, STAGE_PROMPTS  # noqa: E402
from spec_generator.schemas import for_stage                 # noqa: E402


def check_version():
    try:
        with urllib.request.urlopen(BASE + "/api/version", timeout=5) as r:
            version = json.loads(r.read())["version"]
    except Exception as e:
        sys.exit(f"cannot reach Ollama at {BASE}: {e} - is `ollama serve` running?")
    print(f"Ollama version: {version}")
    parts = version.split("-")[0].split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        major, minor = 0, 0
    if (major, minor) < (0, 5):
        print(f"!! Ollama {version} predates structured outputs (need >= 0.5, Dec 2024).")
        print("   Schema mode will 400 with 'format' unmarshal errors. Upgrade:")
        print("   brew upgrade ollama   (or https://ollama.com/download)")
    return version


def call(messages, schema, cfg):
    body = {"model": cfg.model, "messages": messages, "stream": False,
            "options": {"temperature": cfg.temperature, "num_predict": cfg.max_tokens}}
    if schema is not None:
        body["format"] = schema
    req = urllib.request.Request(BASE + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")  # Ollama puts the reason HERE
        raise RuntimeError(f"HTTP {e.code} from Ollama. Full response body:\n{detail}") from e
    return resp["message"]["content"], time.time() - t0


def verdict(raw):
    """Validity exactly the way production judges it: tolerant extraction,
    then parse. Returns (label, parsed-or-None)."""
    try:
        return "VALID JSON", json.loads(_extract_json(raw))
    except Exception as e:
        tail = raw.strip()[-160:].replace("\n", " ")
        return f"INVALID JSON ({type(e).__name__}); output tail: ...{tail}", None


def main():
    path, stage = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "needs")
    intent = json.loads(open(path).read())
    version = check_version()

    cfg = model_config.for_stage(stage)
    print(f"stage '{stage}' on {cfg.model} | temperature {cfg.temperature} | "
          f"num_predict {cfg.max_tokens} (the production budget for this stage)")

    if stage == "needs":
        payload = STAGE_PROMPTS["needs"] + "\n\nINTENT:\n" + json.dumps(intent)
    elif stage == "tests":
        # tests stage consumes the assembled SRS+specs: pass the file explicitly
        if len(sys.argv) < 4:
            sys.exit("tests stage: python scripts/probe_structured.py <intent_model.json> tests <srs_and_specs.json>")
        srs_specs = json.load(open(sys.argv[3]))
        payload = STAGE_PROMPTS["tests"] + "\n\nSRS + SPECS:\n" + json.dumps(srs_specs)
    else:
        sys.exit(f"probe supports stages: needs, tests (got {stage!r})")
    messages = [{"role": "system", "content": METHODOLOGY}, {"role": "user", "content": payload}]

    schema = for_stage(stage)
    results = {}
    for mode in ("free", "schema"):
        print(f"\n--- {mode} generation running (this is the 14b, give it minutes) ---")
        try:
            raw, secs = call(messages, schema if mode == "schema" else None, cfg)
        except Exception as e:
            print(f"{mode}: CALL FAILED: {e}")
            results[mode] = None
            continue
        label, data = verdict(raw)
        results[mode] = data
        out_file = f"probe_{stage}_{mode}.json"
        with open(out_file, "w") as f:
            f.write(raw)
        print(f"{mode}: {secs:.1f}s, {label}")
        print(f"{mode}: full raw output saved to {out_file}")

    print(f"\nOllama {version} | stage '{stage}' | model {cfg.model}")
    if all(results.values()):
        free_items = results["free"].get("needs") or results["free"].get("acceptance_tests") or []
        sch_items = results["schema"].get("needs") or results["schema"].get("acceptance_tests") or []
        key = "statement" if "statement" in (free_items[0] if free_items else {}) else "definition"
        print("\nFree-mode statements:")
        for it in free_items:
            print(f"  - {it.get(key, it)}")
        print("\nSchema-mode statements:")
        for it in sch_items:
            print(f"  - {it.get(key, it)}")
        print("\nCompare the statements above: schema mode is only worth it if the text "
              "quality holds. Report the timings + your quality read back.")
    print("\nProbe done. The probe_*.json files are yours; nothing else was written.")


if __name__ == "__main__":
    main()
