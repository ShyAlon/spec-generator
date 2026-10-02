"""Readable reasons for SRS lint findings: what was checked, what was expected, what was found."""
from __future__ import annotations

import re

_SOURCES = "the interview (intent model), the stakeholder needs, or the concept of operations"
_RULE_TEXT = {
    "named_technology": ("a named technology", "behavior described without naming a product or library"),
    "http_verb": ("an HTTP request", "observable behavior, not an API call"),
    "http_status": ("an HTTP status code", "observable behavior, not a protocol status"),
    "url_path": ("a URL path", "observable behavior, not a route"),
    "json_shape": ("a JSON structure", "observable behavior, not a data format"),
    "database": ("a database term", "observable behavior, not a storage design"),
    "ui_widget": ("a UI widget", "the outcome the user needs, not the control"),
}


def _field(location: str) -> str:
    return location.split(" field ", 1)[1] if " field " in location else location


def _kind(span: str) -> str:
    if re.fullmatch(r"\d+(?:\.\d+)?%?", span):
        return "number"
    if span.isupper():
        return "acronym"
    return "name"


def describe(rid: str, group: list[dict]) -> str:
    """One sentence per rule fired on the same span of the same requirement field."""
    first = group[0]; span = first["span"]; field = _field(first["location"])
    parts = []
    for f in group:
        rule = f["rule"]
        if rule == "absent_from_interview":
            kind = _kind(span)
            note = "a figure the interview never gave" if kind == "number" else f"a {kind} the interview never used"
            parts.append(f"Checked: every number, acronym and capitalized name in {rid} {field} against {_SOURCES}. "
                         f"Expected: each one appears there. Found: '{span}', {note}. It may be an invented detail.")
        elif rule == "ears_pattern":
            parts.append(f"Checked: the statement's sentence pattern. Expected: {f.get('expected', 'an EARS pattern')} "
                         f"Found: '{span[:160]}'.")
        elif rule in _RULE_TEXT:
            what, expected = _RULE_TEXT[rule]
            parts.append(f"Checked: {rid} {field} for implementation language. Expected: {expected}. "
                         f"Found: {what} ('{span}').")
        else:
            parts.append(f"Checked: {rid} {field} with rule {rule}. Found: '{span}'.")
    return " ".join(parts)[:1900]


def group_findings(lint: list[dict], rid: str) -> list[list[dict]]:
    """Merge findings on the same requirement field and span (case-insensitive) into one group."""
    groups: dict[tuple, list[dict]] = {}
    for f in lint:
        if not f["location"].startswith(f"{rid} field "):
            continue
        groups.setdefault((f["location"], f["span"].lower()), []).append(f)
    return list(groups.values())
