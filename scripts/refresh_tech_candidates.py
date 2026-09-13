"""Regenerate the dated tech-candidate MATRIX (founder decision 2026-09-13:
not automation - a script HE runs periodically; the artifact stays committed
to git so builds are deterministic and offline, and this script is the ONLY
writer of src/spec_generator/tech_candidates.json).

One row per technology: name, kinds, platforms, runtime, ecosystem, aliases,
note - the same artifact feeds slot candidates, kind fitness, PLATFORM
fitness (React tagged web cannot silently answer an iOS frontend slot), and
the run-log freshness line. Getting the platforms column right IS the job.

Default is a DRY RUN: regenerate every kind with the local model, merge
multi-kind rows, validate, and print the per-kind added/removed/platform-
changed diff against the committed artifact. --write replaces the artifact
atomically with today's as-of date. The script refuses to write an invalid
matrix (schema, size, duplicate aliases, missing self-alias, sparse kinds).

Honest limit: the local model's knowledge has a cutoff too - regeneration
re-drafts and re-ranks candidates, and the as-of date marks when YOU last
reviewed the matrix. Read the diff before committing; fix wrong platform
tags by hand and commit (the platform column is what the gates enforce).

Usage:
    python scripts/refresh_tech_candidates.py [--write] [--model ollama:phi4]
Model: --model > SPEC_GENERATOR_MODEL_TECH_CANDIDATES > SPEC_GENERATOR_MODEL_ARCHITECTURE
> the architecture stage default.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import urllib.request

BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").removesuffix("/v1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spec_generator import model_config                      # noqa: E402
from spec_generator.errors import UserFacingError            # noqa: E402
from spec_generator.graph import _extract_json               # noqa: E402
from spec_generator.tech_candidates import (                 # noqa: E402
    PLATFORMS, TECH_CANDIDATES_AS_OF, TECH_CANDIDATES_PATH, TECH_KINDS, TECH_ROWS,
    diff_candidates, validate_artifact)

_ROW = {"type": "object",
        "properties": {"name": {"type": "string"},
                       "platforms": {"type": "array", "items": {"enum": list(PLATFORMS)}},
                       "runtime": {"type": ["string", "null"]},
                       "ecosystem": {"type": ["string", "null"]},
                       "aliases": {"type": "array", "items": {"type": "string"}},
                       "note": {"type": "string"}},
        "required": ["name", "platforms", "runtime", "ecosystem", "aliases", "note"],
        "additionalProperties": False}
_SCHEMA = {"type": "object",
           "properties": {"technologies": {"type": "array", "items": _ROW}},
           "required": ["technologies"], "additionalProperties": False}

_PROMPT = """List the current, real, production-used technologies for the software architecture
slot kind '{kind}' as of {today}. 5-12 rows: the established default(s), the strong current
alternatives, and any rising option a fresh review should know.
For EACH row:
- name: the canonical display name ('React Native', 'PostgreSQL').
- platforms: where software built with it RUNS, from {platforms} - this column is enforced
  by a deterministic gate, so get it right: 'React' is web ONLY; 'React Native' and
  'Flutter' are ios+android; a managed cloud service is cloud; a language is any.
- runtime: its language/runtime ('python', 'javascript', 'dart') or null.
- ecosystem: its package/vendor ecosystem ('npm', 'pip', 'firebase', 'apple') or null.
- aliases: lowercase lookup names INCLUDING the lowercase of name itself
  ('PostgreSQL' -> ['postgresql', 'postgres', 'pg']).
- note: one short line on WHEN it fits - orientation, not a verdict.
Existing committed rows for this kind (keep what is still true, drop what aged out, add
what is missing):
{existing}
FORMAT CONTRACT (final instruction - overrides anything above): return one STRICT JSON
object {{"technologies": [...]}} matching the schema - nothing else."""


def _resolve_model() -> str:
    for env in ("SPEC_GENERATOR_MODEL_TECH_CANDIDATES", "SPEC_GENERATOR_MODEL_ARCHITECTURE"):
        if os.environ.get(env):
            return os.environ[env]
    cfg = model_config.for_stage("architecture")
    return f"{cfg.provider}:{cfg.model}"


def _regenerate_kind(kind: str, existing: list[dict], model: str, today: str) -> list[dict]:
    provider, _, name = model.partition(":")
    if provider != "ollama":
        raise UserFacingError(
            f"refresh only knows the local ollama provider; got {model!r}.",
            "the script speaks to the same local model the pipeline uses.",
            "run with --model ollama:<name> (e.g. ollama:phi4).")
    body = {"model": name, "stream": False, "format": _SCHEMA,
            "messages": [{"role": "user", "content": _PROMPT.format(
                kind=kind, today=today, platforms=list(PLATFORMS),
                existing=json.dumps([{"name": r["name"], "platforms": r["platforms"]}
                                     for r in existing]))}],
            "options": {"temperature": 0.0}}
    req = urllib.request.Request(BASE + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.loads(r.read())
    except Exception as e:
        raise UserFacingError(
            f"cannot reach the model at {BASE} ({e}).",
            "ollama is not serving, or the model is not pulled.",
            "start it (ollama serve) and check the model (ollama list).")
    data = _extract_json(resp.get("message", {}).get("content", ""))
    rows = data.get("technologies", []) if isinstance(data, dict) else []
    for r in rows:
        if isinstance(r, dict):
            r["kinds"] = [kind]
    return rows


def _merge_rows(per_kind: dict[str, list[dict]]) -> list[dict]:
    """One row per technology: a row regenerated under several kinds merges
    (union of kinds; first non-empty note/runtime/ecosystem wins)."""
    merged: dict[str, dict] = {}
    for kind, rows in per_kind.items():
        for r in rows:
            key = str(r.get("name", "")).strip().lower()
            if not key:
                continue
            if key in merged:
                m = merged[key]
                m["kinds"] = sorted(set(m["kinds"]) | {kind})
                m["aliases"] = sorted(set(m["aliases"]) | set(r.get("aliases", [])))
                for f in ("note", "runtime", "ecosystem"):
                    if not m.get(f) and r.get(f):
                        m[f] = r[f]
            else:
                merged[key] = {**r, "kinds": [kind]}
    return list(merged.values())


def main() -> int:
    write = "--write" in sys.argv
    model = _resolve_model()
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    today = datetime.date.today().isoformat()
    print(f"Regenerating the tech candidate matrix with {model} "
          f"(committed matrix as of {TECH_CANDIDATES_AS_OF})")
    existing_by_kind = {k: [r for r in TECH_ROWS if k in r["kinds"]] for k in TECH_KINDS}
    per_kind = {}
    for kind in TECH_KINDS:
        print(f"  - {kind} ({len(existing_by_kind[kind])} committed) ...", flush=True)
        per_kind[kind] = _regenerate_kind(kind, existing_by_kind[kind], model, today)
    artifact = {"_generated_by": "scripts/refresh_tech_candidates.py - the ONLY writer of "
                                 "this file; hand-edits get overwritten",
                "as_of": today, "kinds": list(TECH_KINDS),
                "technologies": _merge_rows(per_kind)}
    problems = validate_artifact(artifact)
    if problems:
        raise UserFacingError(
            "the regenerated matrix is invalid - refusing to write:\n  " + "\n  ".join(problems[:10]),
            "the model returned malformed rows (missing fields, duplicate aliases, a sparse kind).",
            "re-run the script; if a kind keeps failing, hand-fix that kind's rows and commit "
            "the artifact - the platform column is what the gates enforce.")
    committed = {"technologies": TECH_ROWS}
    diff = diff_candidates(committed, artifact)
    print("\nDiff against the committed matrix:")
    print("\n".join("  " + ln for ln in diff) if diff else "  (no changes)")
    if not write:
        print("\nDRY RUN - nothing written. Re-run with --write, review the artifact, commit it.")
        return 0
    tmp = TECH_CANDIDATES_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(artifact, indent=2) + "\n")
    os.replace(tmp, TECH_CANDIDATES_PATH)
    print(f"\nWrote {TECH_CANDIDATES_PATH} (as_of {today}). Review the diff and commit it:")
    print("  git add src/spec_generator/tech_candidates.json && git commit")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except UserFacingError as e:
        print(e.render(), file=sys.stderr)
        sys.exit(1)
