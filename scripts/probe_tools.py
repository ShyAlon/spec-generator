"""Probe: can YOUR Ollama model drive the typed structural tools (ADR-0020)?

The tool path is opt-in (SPEC_GENERATOR_TOOLS=1) until this probe says the
model emits well-formed, schema-conformant tool calls. The probe mirrors
production exactly: same prompts (prompts.TOOL_PROMPTS + METHODOLOGY), same
tool schemas and the SAME validating handlers (tools.SrsToolbox/TestToolbox)
that run in the pipeline - so a malformed call here fails the same way it
would fail in a real run, and an applied call here would apply there.

Usage:
    python scripts/probe_tools.py [model]      (default: your srs stage model)

Scenarios:
  1. split_expected   - a genuinely compound need: the model SHOULD split it
  2. no_split_expected - atomic needs: the model SHOULD call no tools
  3. add_test         - a skeleton with a real gap: a valid add_test or none
  4. link_trace       - one requirement realizing two needs: a valid link
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:11434"
MAX_ROUNDS = 3

sys.path.insert(0, "src")
from spec_generator import model_config                     # noqa: E402
from spec_generator.prompts import METHODOLOGY, TOOL_PROMPTS  # noqa: E402
from spec_generator.skeleton import (ModelStructureError,     # noqa: E402
                                     build_srs_skeleton, build_test_skeleton)
from spec_generator.tools import SrsToolbox, TestToolbox     # noqa: E402


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
    if (major, minor) < (0, 3):
        print(f"!! Ollama {version} predates tool calling (need >= 0.3, Jul 2024).")
        print("   Upgrade: brew upgrade ollama   (or https://ollama.com/download)")
    return version


def chat(model, cfg, messages, tools):
    body = {"model": model, "messages": messages, "stream": False,
            "tools": tools,
            "options": {"temperature": cfg.temperature, "num_predict": cfg.max_tokens}}
    req = urllib.request.Request(BASE + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["message"], time.time() - t0
    except urllib.error.HTTPError as e:
        print(f"!! HTTP {e.code}: {e.read().decode(errors='replace')[:400]}")
        raise


def run_scenario(name, model, cfg, toolbox, human, expect):
    """A production-shaped propose/dispose loop with live handler verdicts."""
    print(f"\n--- {name} " + "-" * max(1, 60 - len(name)))
    messages = [{"role": "system", "content": METHODOLOGY},
                {"role": "user", "content": human}]
    conformant = malformed = 0
    for rnd in range(1, MAX_ROUNDS + 1):
        msg, secs = chat(model, cfg, messages, toolbox.tools)
        calls = msg.get("tool_calls") or []
        print(f"round {rnd}: {len(calls)} tool call(s), {secs:.1f}s")
        if not calls:
            content = (msg.get("content") or "").strip()
            if content:
                print(f"  model said: {content[:120]}")
            break
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": calls})
        for c in calls:
            fn = c.get("function") or {}
            cname, args = fn.get("name"), fn.get("arguments")
            print(f"  {cname}({json.dumps(args)[:160]})")
            try:
                result = toolbox.call(cname, args if isinstance(args, dict) else {})
                print(f"    APPLIED: {json.dumps(result)[:140]}")
                conformant += 1
                reply = json.dumps(result)
            except (ModelStructureError, TypeError) as e:
                print(f"    REJECTED by handler: {e}")
                malformed += 1
                reply = f"ERROR: {e} - correct the arguments or stop proposing."
            messages.append({"role": "tool", "name": cname, "content": reply})
    for line in toolbox.applied:
        print(f"  structure: {line}")
    return {"calls": conformant + malformed, "conformant": conformant,
            "malformed": malformed, "applied": toolbox.applied}


def main():
    version = check_version()
    cfg = model_config.for_stage("srs")
    model = sys.argv[1] if len(sys.argv) > 1 else cfg.model
    print(f"probing tool calling on '{model}' (stage temperature {cfg.temperature})\n")

    compound = {"needs": [{"need_id": "SN-1", "stakeholder": "founder", "priority": "high",
                           "statement": "Ingest signup CSV uploads and give me a live "
                                        "conversion dashboard per acquisition source"}]}
    atomic = {"needs": [{"need_id": "SN-1", "stakeholder": "founder", "priority": "high",
                         "statement": "Export the report as CSV"},
                        {"need_id": "SN-2", "stakeholder": "founder", "priority": "medium",
                         "statement": "Load the dashboard in under 2 seconds"}]}
    shared = {"needs": [{"need_id": "SN-1", "stakeholder": "founder", "priority": "high",
                         "statement": "Upload a CSV of signups"},
                        {"need_id": "SN-2", "stakeholder": "founder", "priority": "high",
                         "statement": "See conversion per source"}]}
    intent = {"problem_statement": "founders can't see where signups come from"}

    results = {}
    slots = build_srs_skeleton(compound)
    results["split_expected"] = run_scenario(
        "split_expected (a compound need SHOULD be split)", model, cfg,
        SrsToolbox(slots, {"SN-1"}),
        TOOL_PROMPTS["srs"].format(skeleton=json.dumps(slots),
                                   needs=json.dumps(compound),
                                   intent=json.dumps(intent)), "split")
    slots = build_srs_skeleton(atomic)
    results["no_split_expected"] = run_scenario(
        "no_split_expected (atomic needs, expect ZERO tool calls)", model, cfg,
        SrsToolbox(slots, {"SN-1", "SN-2"}),
        TOOL_PROMPTS["srs"].format(skeleton=json.dumps(slots),
                                   needs=json.dumps(atomic),
                                   intent=json.dumps(intent)), "quiet")
    slots = build_srs_skeleton(shared)
    results["link_trace"] = run_scenario(
        "link_trace (prompt nudges one shared requirement)", model, cfg,
        SrsToolbox(slots, {"SN-1", "SN-2"}),
        TOOL_PROMPTS["srs"].format(skeleton=json.dumps(slots),
                                   needs=json.dumps(shared),
                                   intent=json.dumps(intent))
        + "\n\nNote: uploads feed the conversion view - if R-1 realizes both needs, "
          "link_requirement_to_need it to SN-2.", "link")

    # add_test: one acceptance criterion, one spec, and NO unit slot yet -> real gap
    srs = {"functional": [{"requirement_id": "R-1", "kind": "functional", "title": "Upload",
                           "statement": "The system shall accept CSV uploads of signup rows.",
                           "priority": "must", "need_ids": ["SN-1"],
                           "acceptance": [{"criterion_id": "AC-R-1-1",
                                           "statement": "a valid CSV uploads", "testable": True}]}],
           "non_functional": []}
    items = [{"spec_id": "SPEC-1", "component": "uploader", "item_id": "I-1", "title": "Upload API",
              "description": "accepts CSV", "unit_test": "t", "depends_on": [],
              "exposes": ["POST /api/uploads"], "data": [], "risks": [],
              "requirement_ids": ["R-1"], "estimated_effort": "S"}]
    tslots = build_test_skeleton(srs, items)
    results["add_test"] = run_scenario(
        "add_test (a module/system test spanning the flow would be a real addition)",
        model, cfg, TestToolbox(tslots, {"AC-R-1-1"}, {"SPEC-1"}),
        TOOL_PROMPTS["tests"].format(skeleton=json.dumps(tslots),
                                     srs=json.dumps(srs), items=json.dumps(items)), "add")

    print("\n" + "=" * 64)
    print(f"Ollama {version} | model '{model}'")
    total = sum(r["calls"] for r in results.values())
    malformed = sum(r["malformed"] for r in results.values())
    for name, r in results.items():
        print(f"  {name:24s} calls={r['calls']} conformant={r['conformant']} "
              f"malformed={r['malformed']} applied={len(r['applied'])}")
    print()
    split_ok = bool(results["split_expected"]["applied"])
    quiet_ok = results["no_split_expected"]["calls"] == 0
    print(f"  splits a compound need:      {'yes' if split_ok else 'NO'}")
    print(f"  stays quiet on atomic needs: {'yes' if quiet_ok else 'NO'}")
    print(f"  schema-conformant calls:     {total - malformed}/{total}")
    if malformed == 0 and split_ok and quiet_ok:
        print("\nVERDICT: RELIABLE - tool path looks safe on this model.")
        print("  Enable it:  SPEC_GENERATOR_TOOLS=1 python -m spec_generator ...")
    elif total == 0:
        print("\nVERDICT: SILENT - the model emitted no tool calls at all (did it "
              "see the tools?). Keep the default (SPEC_GENERATOR_TOOLS=0) and "
              "share this output.")
    else:
        print("\nVERDICT: DEGRADED - keep the default (SPEC_GENERATOR_TOOLS=0); "
              "the pipeline falls back to the plain 1:1 skeleton automatically. "
              "Share this output.")


if __name__ == "__main__":
    main()
