"""lineage.json: the local authority record binding settled artifacts to their provenance.

A disk artifact is settled state only when a lineage entry binds it:
task + session + generation + phase + artifact name + content sha256 + source,
and the record's launch-input hashes (intent/conops/baseline) match the current
run's input files. IDs alone are not lineage; hashes alone are not identity.

Writers: delivery-api restore (source="restore", carries the published revision),
the generator at each stage commit (source="stage-commit"), and the founder-decision
path (source="founder-decision"). The file never crosses hosts and is never
published. A `restore`-bound (phase, name) is never silently re-bound by a
stage commit; only an explicit --from rewind (invalidate_from) or a
founder-decision advance may replace it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .document_runtime import read_bytes,read_text,exists,atomic_text

RECORD_NAME = "lineage.json"
_VERSION = 1

# phase -> artifact file names the generator stage commits own (declared set only).
STAGE_ARTIFACTS = {
    "needs": ("needs.json",),
    "srs": ("srs.json",),
    "architecture": ("architecture.json",),
    "items": ("items.json",),
    "tests": ("tests.json",),
    "validation": ("validation.json",),
}


class LineageRefusal(Exception):
    """A settled-state claim failed verification. Always carries a recovery."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(read_bytes(path)).hexdigest()


# Launch-input identity contract (scheme-tagged):
# - "<name>.json" inputs are JSON documents: identity is the sha256 of their
#   canonical serialization (UTF-8; object keys sorted; compact separators;
#   array order preserved). Whitespace and object key order are serialization,
#   not content. Ambiguous documents (duplicate object keys) and non-finite
#   numbers refuse - they have no single meaning to bind.
# - Any other input is opaque bytes: identity is the sha256 of the exact bytes.
# Records carry the scheme explicitly ("jsonv1:" / "sha256:") so a record is
# never silently reinterpreted under another scheme; a pre-scheme bare-hex
# record is refused with the explicit re-restore recovery, never re-read.
JSON_INPUT_SCHEME = "jsonv1"
BYTES_INPUT_SCHEME = "sha256"


def _canonical_json(data: bytes, *, name: str) -> bytes:
    def _object(pairs):
        seen = set()
        out = {}
        for key, value in pairs:
            if key in seen:
                raise LineageRefusal(
                    f"launch input {name} is ambiguous JSON (duplicate key {key!r}); restore "
                    "the session again from the published manifest (no model calls).")
            seen.add(key)
            out[key] = value
        return out

    def _constant(token):
        raise LineageRefusal(
            f"launch input {name} is non-finite JSON ({token}); restore the session "
            "again from the published manifest (no model calls).")

    try:
        parsed = json.loads(data.decode("utf-8"), object_pairs_hook=_object,
                            parse_constant=_constant)
    except LineageRefusal:
        raise
    except (ValueError, UnicodeDecodeError) as exc:
        raise LineageRefusal(
            f"launch input {name} is not well-formed JSON ({exc}); restore the session "
            "again from the published manifest (no model calls).") from exc
    try:
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except ValueError as exc:  # a non-finite literal such as 1e999
        raise LineageRefusal(
            f"launch input {name} is non-finite JSON ({exc}); restore the session "
            "again from the published manifest (no model calls).") from exc


def hash_launch_input(path: Path, *, name: str) -> str:
    """Scheme-tagged launch-input identity under the contract above."""
    data = read_bytes(path)
    if name.endswith(".json"):
        return JSON_INPUT_SCHEME + ":" + hashlib.sha256(_canonical_json(data, name=name)).hexdigest()
    return BYTES_INPUT_SCHEME + ":" + hashlib.sha256(data).hexdigest()


def _record_path(root: Path) -> Path:
    return Path(root) / RECORD_NAME


def load(root: Path) -> dict | None:
    """None when no record exists (standalone runs before any restore)."""
    path = _record_path(root)
    if not exists(path):
        return None
    try:
        record = json.loads(read_text(path))
    except (ValueError, OSError) as exc:
        raise LineageRefusal(
            f"lineage record at {path} is unreadable ({exc}); restore the session again "
            "from the published manifest to rebuild it (no model calls needed)."
        ) from exc
    if not isinstance(record, dict) or record.get("version") != _VERSION:
        raise LineageRefusal(
            f"lineage record at {path} has an unsupported shape; restore the session again "
            "from the published manifest to rebuild it (no model calls needed)."
        )
    record.setdefault("inputs", {})
    record.setdefault("entries", [])
    return record


def _write(root: Path, record: dict) -> None:
    path = _record_path(root)
    atomic_text(path,json.dumps(record, indent=2, sort_keys=True))


def _identity(task_id, session_id, generation) -> dict:
    return {"task_id": task_id or "", "session_id": session_id or "",
            "generation": str(generation or "")}


def record_inputs(root: Path, *, task_id, session_id, generation, input_hashes: dict) -> dict:
    """Bind the launch inputs (intent/conops/baseline hashes) to the identity."""
    record = load(root) or {"version": _VERSION, "inputs": {}, "entries": []}
    record["identity"] = _identity(task_id, session_id, generation)
    record["inputs"] = dict(input_hashes)
    _write(root, record)
    return record


def verify_inputs(record: dict, input_files: dict) -> None:
    """The current run's launch inputs must match the record under their tagged scheme."""
    bound = (record or {}).get("inputs") or {}
    for name, path in input_files.items():
        if name in bound and exists(path):
            stored = str(bound[name])
            if ":" not in stored:
                raise LineageRefusal(
                    f"launch input {name} was recorded before scheme-tagged input identity; "
                    "the record is not reinterpreted under the current scheme. Restore the "
                    "session again from the published manifest (no model calls) to re-record it."
                )
            actual = hash_launch_input(path, name=name)
            if actual != stored:
                raise LineageRefusal(
                    f"launch input {name} differs from the settled record "
                    f"(recorded {stored[:19]}..., found {actual[:19]}...); "
                    "this session's approved work was produced from other inputs. "
                    "Restore the session again from the published manifest (no model calls), "
                    "or start a new output directory for the new inputs."
                )


def bind(root: Path, *, task_id, session_id, generation, phase: str, name: str,
         sha256: str, source: str, revision: str | None = None, settled: bool = True) -> dict:
    """Record one artifact binding. A SETTLED entry (approved/completed phase) is
    never silently replaced by a stage commit: that re-bind IS the settled-work
    substitution. A provisional entry (awaiting_approval at restore: the gate may
    legitimately regenerate the phase on founder feedback) may be re-bound by the
    stage commit that produces it. Explicit --from rewinds invalidate bindings
    first, so the refusal below only fires on an unsanctioned overwrite."""
    if source not in ("restore", "stage-commit", "founder-decision", "assembly"):
        raise ValueError(f"unknown lineage source {source!r}")
    record = load(root) or {"version": _VERSION, "inputs": {}, "entries": []}
    record["identity"] = _identity(task_id, session_id, generation)
    entries = [e for e in record["entries"] if not (e["phase"] == phase and e["name"] == name)]
    prior = [e for e in record["entries"] if e["phase"] == phase and e["name"] == name]
    # "assembly" is the authorized terminal rewrite (final bundle render re-serializes
    # architecture/tests/validation from state); it carries the prior entry's
    # settledness forward explicitly and is allowed to replace the entry.
    if prior and source == "stage-commit":
        hard = [e for e in prior if e.get("settled", True)]
        if hard and all(e["sha256"] != sha256 for e in hard):
            raise LineageRefusal(
                f"stage commit for {phase}/{name} would replace settled bytes "
                f"(settled sha {hard[-1]['sha256'][:12]}... from {hard[-1]['source']}, "
                f"new {sha256[:12]}...) without an explicit rewind. "
                "Re-running a settled stage is only allowed via --from <stage>."
            )
        if prior and any(e["sha256"] == sha256 for e in prior):
            return record  # identical re-commit: no-op
    entry = {"phase": phase, "name": name, "sha256": sha256, "source": source,
             "settled": bool(settled), **_identity(task_id, session_id, generation)}
    if revision is not None:
        entry["revision"] = revision
    entries.append(entry)
    record["entries"] = entries
    _write(root, record)
    return record


def lookup(record: dict, phase: str, name: str) -> dict | None:
    for entry in (record or {}).get("entries") or []:
        if entry["phase"] == phase and entry["name"] == name:
            return entry
    return None


def verify_file(record: dict, *, phase: str, name: str, path: Path,
                task_id=None, session_id=None, generation=None) -> dict:
    """Admit a disk artifact as settled only when the binding matches exactly."""
    entry = lookup(record, phase, name)
    if entry is None:
        raise LineageRefusal(
            f"no settled binding for {phase}/{name}; the file on disk is not authoritative. "
            "Restore the session again from the published manifest (no model calls)."
        )
    if not exists(path):
        raise LineageRefusal(
            f"settled artifact {phase}/{name} is missing on disk; "
            "restore the session again from the published manifest (no model calls)."
        )
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise LineageRefusal(
            f"settled artifact {phase}/{name} changed on disk "
            f"(settled {entry['sha256'][:12]}..., found {actual[:12]}...); "
            "restore the session again from the published manifest (no model calls)."
        )
    for key, expected in (("task_id", task_id), ("session_id", session_id), ("generation", generation)):
        if expected is not None and entry.get(key) not in (None, "", str(expected)):
            raise LineageRefusal(
                f"settled artifact {phase}/{name} belongs to {key} {entry.get(key)}, "
                f"not {expected}; refusing to treat another run's output as settled."
            )
    return entry


def mark_settled(root: Path, phase: str) -> dict | None:
    """Gate approval flips every binding for the phase to settled: from here on a
    stage commit may not replace those bytes without an explicit rewind (--from or
    a gate redo, both of which invalidate first)."""
    record = load(root)
    if record is None:
        return None
    changed = False
    for entry in record["entries"]:
        if entry["phase"] == phase and not entry.get("settled", True):
            entry["settled"] = True
            changed = True
    if changed:
        _write(root, record)
    return record


def invalidate_from(root: Path, phases: set[str]) -> dict | None:
    """Explicit --from rewind: drop bindings for the rewound stage and everything
    downstream so stale downstream bytes can never be hydrated afterward."""
    record = load(root)
    if record is None:
        return None
    record["entries"] = [e for e in record["entries"] if e["phase"] not in phases]
    _write(root, record)
    return record
