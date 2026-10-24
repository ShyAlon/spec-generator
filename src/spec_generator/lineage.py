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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record_path(root: Path) -> Path:
    return Path(root) / RECORD_NAME


def load(root: Path) -> dict | None:
    """None when no record exists (standalone runs before any restore)."""
    path = _record_path(root)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
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
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True))
    os.replace(tmp, path)


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
    """The current run's launch input files must hash-match the record."""
    bound = (record or {}).get("inputs") or {}
    for name, path in input_files.items():
        if name in bound and Path(path).is_file():
            actual = sha256_file(path)
            if actual != bound[name]:
                raise LineageRefusal(
                    f"launch input {name} differs from the settled record "
                    f"(recorded {bound[name][:12]}..., found {actual[:12]}...); "
                    "this session's approved work was produced from other inputs. "
                    "Restore the session again from the published manifest (no model calls), "
                    "or start a new output directory for the new inputs."
                )


def bind(root: Path, *, task_id, session_id, generation, phase: str, name: str,
         sha256: str, source: str, revision: str | None = None) -> dict:
    """Record one settled artifact. A restore-bound entry is never silently
    replaced by a stage commit: that re-bind IS the settled-work substitution."""
    if source not in ("restore", "stage-commit", "founder-decision"):
        raise ValueError(f"unknown lineage source {source!r}")
    record = load(root) or {"version": _VERSION, "inputs": {}, "entries": []}
    record["identity"] = _identity(task_id, session_id, generation)
    entries = [e for e in record["entries"] if not (e["phase"] == phase and e["name"] == name)]
    prior = [e for e in record["entries"] if e["phase"] == phase and e["name"] == name]
    if prior and source == "stage-commit":
        settled = [e for e in prior if e["source"] in ("restore", "stage-commit", "founder-decision")]
        if settled and all(e["sha256"] != sha256 for e in settled):
            raise LineageRefusal(
                f"stage commit for {phase}/{name} would replace settled bytes "
                f"(settled sha {settled[-1]['sha256'][:12]}... from {settled[-1]['source']}, "
                f"new {sha256[:12]}...) without an explicit rewind. "
                "Re-running a settled stage is only allowed via --from <stage>."
            )
        if settled and any(e["sha256"] == sha256 for e in settled):
            return record  # identical re-commit: no-op
    entry = {"phase": phase, "name": name, "sha256": sha256, "source": source,
             **_identity(task_id, session_id, generation)}
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
    if not Path(path).is_file():
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


def invalidate_from(root: Path, phases: set[str]) -> dict | None:
    """Explicit --from rewind: drop bindings for the rewound stage and everything
    downstream so stale downstream bytes can never be hydrated afterward."""
    record = load(root)
    if record is None:
        return None
    record["entries"] = [e for e in record["entries"] if e["phase"] not in phases]
    _write(root, record)
    return record
