"""The technology candidate MATRIX (founder decisions 2026-09-13, extended
same morning: 'it's probably more of a matrix'). One row per technology with
attributes: kinds, platforms, runtime, ecosystem, aliases, note, plus the
artifact's as-of freshness date. ONE artifact feeds four gates:

  1. slot candidates     - per-kind suggestions injected into stack prompts
  2. kind fitness        - the slot-fitness gate's lexicons are DERIVED here
                           (absorbed, not duplicated: name + aliases per row)
  3. attribute fitness   - slot context vs candidate platform (React tagged
                           web cannot silently answer an iOS frontend slot;
                           the 2026-09-13 habit-app React-for-mobile defect)
  4. freshness           - the as-of date prints in every run log

FRESHNESS (his call): the DATA lives in tech_candidates.json - a generated
artifact committed to git so builds stay deterministic and offline.
scripts/refresh_tech_candidates.py is its ONLY writer: he runs it
periodically, reviews the printed per-kind diff, and commits. This module is
the hand-owned loader and contract; the artifact is the data."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .errors import UserFacingError

TECH_CANDIDATES_PATH = Path(__file__).with_name("tech_candidates.json")

PLATFORMS = ("web", "ios", "android", "cloud", "desktop", "any")

# Kinds where the platform attribute is a real constraint: a wrong-platform
# technology CANNOT answer these slots without explicit justification.
PLATFORM_CHECKED_KINDS = ("framework", "mobile_framework", "visualization",
                          "push_notifications", "health_data")

# Slot-key -> candidate kind, beyond the fitness gate's exact map (slot keys
# are model-proposed, so substring rules catch the mobile/app keys the exact
# map cannot foresee). The exact map wins; first substring rule in order wins.
SLOT_KEY_SUBSTR: tuple[tuple[str, str], ...] = (
    ("mobile", "mobile_framework"),
    ("ios", "mobile_framework"),
    ("android", "mobile_framework"),
    ("push", "push_notifications"),
    ("notification", "push_notifications"),
    ("health", "health_data"),
    ("fitness", "health_data"),
    ("step", "health_data"),
    ("backend", "backend"),
    ("api", "backend"),
    ("server", "backend"),
    ("frontend", "framework"),
    ("web", "framework"),
    ("ui", "framework"),
    ("database", "database"),
    ("storage", "database"),
    ("persist", "database"),
    ("auth", "auth"),
    ("chart", "visualization"),
    ("visual", "visualization"),
    ("queue", "queue"),
    ("language", "language"),
    ("runtime", "language"),
)


def validate_artifact(data) -> list[str]:
    """Structural contract for the generated matrix; [] means valid. The
    refresh script refuses to WRITE anything failing this, and the loader
    refuses to READ it - a corrupt matrix must fail loudly, never silently
    narrow the model's options."""
    if not isinstance(data, dict):
        return ["artifact is not a JSON object"]
    problems = []
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(data.get("as_of", ""))):
        problems.append("as_of must be an ISO date string (YYYY-MM-DD)")
    kinds = data.get("kinds")
    if not isinstance(kinds, list) or not kinds or not all(isinstance(k, str) for k in kinds):
        problems.append("kinds must be a non-empty list of kind names")
        kinds = []
    rows = data.get("technologies")
    if not isinstance(rows, list) or not rows:
        return problems + ["technologies must be a non-empty list of rows"]
    seen_aliases: dict[str, str] = {}
    per_kind: dict[str, int] = {k: 0 for k in kinds}
    for r in rows:
        name = str(r.get("name", "")).strip() if isinstance(r, dict) else ""
        if not name:
            problems.append("every row needs a non-empty name")
            continue
        rk = r.get("kinds")
        if (not isinstance(rk, list) or not rk
                or any(k not in kinds for k in rk)):
            problems.append(f"{name!r}: kinds must be non-empty and declared in the artifact's kinds list")
        else:
            for k in rk:
                per_kind[k] += 1
        rp = r.get("platforms")
        if (not isinstance(rp, list) or not rp
                or any(p not in PLATFORMS for p in rp)):
            problems.append(f"{name!r}: platforms must be non-empty, from {list(PLATFORMS)}")
        if not str(r.get("note", "")).strip():
            problems.append(f"{name!r}: note must be non-empty")
        for a in r.get("aliases", []) if isinstance(r.get("aliases"), list) else []:
            if not isinstance(a, str) or a != a.lower().strip() or not a:
                problems.append(f"{name!r}: aliases must be lowercase non-empty strings")
                continue
            if a in seen_aliases:
                problems.append(f"alias {a!r} claimed by both {seen_aliases[a]!r} and {name!r} "
                                "- the derived lexicons would be ambiguous")
            seen_aliases[a] = name
        if name.lower() not in [a for a in r.get("aliases", [])]:
            problems.append(f"{name!r}: aliases must contain the row's own lowercase name")
    for k, n in sorted(per_kind.items()):
        if not (3 <= n <= 40):
            problems.append(f"kind {k!r}: needs 3-40 rows (got {n})")
    return problems


def diff_candidates(old: dict, new: dict) -> list[str]:
    """The refresh script's review surface: per-kind added/removed rows, then
    attribute changes on kept rows (a platform flip IS the signal he reviews)."""
    def rows_of(d):
        return {str(r.get("name", "")).strip().lower(): r
                for r in d.get("technologies", []) if isinstance(r, dict)}
    o, n = rows_of(old), rows_of(new)
    lines = []
    kinds = sorted({k for r in list(o.values()) + list(n.values()) for k in r.get("kinds", [])})
    for kind in kinds:
        ok = {nm for nm, r in o.items() if kind in r.get("kinds", [])}
        nk = {nm for nm, r in n.items() if kind in r.get("kinds", [])}
        parts = [f"+{a}" for a in sorted(nk - ok)] + [f"-{r}" for r in sorted(ok - nk)]
        if parts:
            lines.append(f"{kind}: " + " ".join(parts))
    for nm in sorted(o.keys() & n.keys()):
        op, np_ = o[nm].get("platforms", []), n[nm].get("platforms", [])
        if sorted(op) != sorted(np_):
            lines.append(f"~{nm}: platforms {sorted(op)} -> {sorted(np_)}")
    return lines


def _load() -> tuple[str, list[str], list[dict]]:
    if not TECH_CANDIDATES_PATH.exists():
        raise UserFacingError(
            "the tech candidate matrix is missing: src/spec_generator/tech_candidates.json.",
            "the artifact is committed to git - a partial checkout, or it was deleted.",
            "restore it (git checkout -- src/spec_generator/tech_candidates.json) or regenerate: "
            "python scripts/refresh_tech_candidates.py --write")
    data = json.loads(TECH_CANDIDATES_PATH.read_text())
    problems = validate_artifact(data)
    if problems:
        raise UserFacingError(
            "the tech candidate matrix is invalid: " + "; ".join(problems[:5]),
            "the file's only writer is scripts/refresh_tech_candidates.py - a hand edit or a "
            "bad regeneration left it in a shape the pipeline refuses to read.",
            "restore it (git checkout -- src/spec_generator/tech_candidates.json) or regenerate: "
            "python scripts/refresh_tech_candidates.py --write")
    return data["as_of"], data["kinds"], data["technologies"]


TECH_CANDIDATES_AS_OF, TECH_KINDS, TECH_ROWS = _load()

# kind -> ((display name, note), ...) - the slot-candidate prompt view. Kind
# order follows the artifact's DECLARED kind list, never set iteration:
# _wrong_kind returns the first matching kind, and hash-seed-dependent
# verdicts would break ADR-0022 replayability.
TECH_CANDIDATES: dict[str, tuple[tuple[str, str], ...]] = {
    k: tuple((r["name"], r["note"]) for r in TECH_ROWS if k in r["kinds"])
    for k in TECH_KINDS}

# kind -> (lowercase name + aliases, ...) - the slot-fitness gate's lexicons,
# DERIVED from the matrix (the 2026-09-13 matrix decision: one artifact, no
# duplicated tech knowledge).
KIND_LEXICONS: dict[str, tuple[str, ...]] = {
    k: tuple(a for r in TECH_ROWS if k in r["kinds"] for a in r["aliases"])
    for k in TECH_KINDS}

_BY_ALIAS = {a: r for r in TECH_ROWS for a in r["aliases"]}


def tech_row(value: str) -> dict | None:
    """The matrix row for a technology name or alias (case-insensitive), or
    None for novel tech - a matrix miss is never a verdict by itself."""
    return _BY_ALIAS.get(value.strip().lower())


def candidate_kind(key: str, exact_map: dict[str, str]) -> str | None:
    """Slot kind for candidate-list injection: the fitness gate's exact map
    first, then substring rules for the model-proposed keys no map can
    foresee (mobile_framework, push_notifications, health_data...)."""
    if key in exact_map:
        return exact_map[key]
    k = key.lower()
    for sub, kind in SLOT_KEY_SUBSTR:
        if sub in k:
            return kind
    return None
