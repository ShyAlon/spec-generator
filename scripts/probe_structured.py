"""A/B probe: schema-constrained vs free generation on YOUR Ollama.

Runs the needs stage twice (with and without the contracts-derived schema) and
reports validity, wall time, and both outputs for a quality eyeball. Grammar
constraints can dull small models - look at the need statements, not just the
validity line.

Usage:
    python scripts/probe_structured.py <path-to-intent_model.json> [stage]
Stages: needs (default) | review   (needs is the representative authoring case)
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = "http://localhost:11434"
MODEL = "qwen2.5-coder:14b"


def call(messages, schema=None):
    body = {"model": MODEL, "messages": messages, "stream": False,
            "options": {"temperature": 0.2, "num_predict": 8192}}
    if schema is not None:
        body["format"] = schema
    req = urllib.request.Request(BASE + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        resp = json.loads(r.read())
    return resp["message"]["content"], time.time() - t0


def main():
    path, stage = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "needs")
    intent = json.loads(open(path).read())

    sys.path.insert(0, "src")
    from spec_generator.prompts import METHODOLOGY, STAGE_PROMPTS
    from spec_generator.schemas import for_stage

    version = json.loads(urllib.request.urlopen(BASE + "/api/version", timeout=5).read())["version"]
    print(f"Ollama {version} (structured outputs need >= 0.5)")

    if stage == "needs":
        payload = STAGE_PROMPTS["needs"] + "\n\nINTENT:\n" + json.dumps(intent)
    else:
        payload = STAGE_PROMPTS["review"].format(n=1) + "\n\nSRS:\n" + json.dumps(intent.get("_srs", intent))
    messages = [{"role": "system", "content": METHODOLOGY}, {"role": "user", "content": payload}]

    schema = for_stage(stage)
    results = {}
    for mode in ("free", "schema"):
        print(f"\n--- {mode} generation running (this is the 14b, give it minutes) ---")
        try:
            text, secs = call(messages, schema if mode == "schema" else None)
        except Exception as e:
            print(f"{mode}: CALL FAILED: {type(e).__name__}: {e}")
            results[mode] = None
            continue
        try:
            data = json.loads(text)
            valid = f"VALID JSON ({len(data.get('needs', data)) if isinstance(data, (dict, list)) else '?'} items)"
        except json.JSONDecodeError:
            valid = "INVALID JSON"
            data = text
        results[mode] = data
        print(f"{mode}: {secs:.1f}s, {valid}")
        print(json.dumps(data, indent=2)[:1200])

    if all(results.values()):
        print("\nCompare the NEED STATEMENTS above: schema mode is only worth it if the "
              "text quality holds. Report the timings + your quality read back.")
    print("\nProbe done. Nothing was written; this hits your local Ollama only.")


if __name__ == "__main__":
    main()
