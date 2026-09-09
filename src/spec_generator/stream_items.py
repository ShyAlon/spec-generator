"""Per-item progress during a stage - observability only, never control flow.

Stages ask the model for ONE JSON response (an array, or an object holding
arrays). This module streams that response and reports each array element the
moment its JSON closes, so the founder watches needs/requirements/specs land
one line at a time instead of staring at a silent stage line for minutes.

The output contract is UNTOUCHED: same prompt, same model, same parameters;
.streaming just delivers the same completion incrementally. The accumulated
text is exactly what _invoke would have returned, and downstream parsing,
guards, and validation see the identical string.
"""
from __future__ import annotations

import json
import time

from .progress import ProgressBus, ProgressEvent

_ID_KEYS = ("need_id", "requirement_id", "spec_id", "test_id", "criterion_id",
            "decision_id", "milestone_id", "item_id")
_TEXT_KEYS = ("title", "name", "statement", "description")


def extract_array_items(text: str) -> list[str]:
    """Return the raw JSON strings of every COMPLETE element of any top-level
    array (a bare top-level array, or an array that is a value in the root
    object). Nested arrays inside elements are not reported. String-aware:
    braces inside string literals do not count. A trailing partial element is
    simply not returned yet."""
    items: list[str] = []
    stack: list[str] = []
    in_str = False
    esc = False
    elem: list[int] | None = None  # [start_index, stack depth it must close back to]

    def reportable() -> bool:
        return stack == ["["] or stack == ["{", "["]

    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
                if elem is not None and elem[1] == len(stack) and text[elem[0]] == '"':
                    items.append(text[elem[0]:i + 1])
                    elem = None
            i += 1
            continue
        if c == '"':
            in_str = True
            if elem is None and reportable():
                elem = [i, len(stack)]
        elif c in "{[":
            if elem is None and reportable():
                elem = [i, len(stack)]
            stack.append(c)
        elif c in "}]":
            if stack:
                stack.pop()
            if elem is not None and len(stack) == elem[1]:
                items.append(text[elem[0]:i + 1])
                elem = None
        i += 1
    return items


def item_label(raw: str, index: int) -> str:
    """Human one-liner for a completed element: its id + short title, or a
    truncated string for plain-string elements (e.g. review fixes)."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return f"item {index}"
    if isinstance(obj, str):
        text = obj.strip()
        return (text[:70] + "...") if len(text) > 70 else (text or f"item {index}")
    if not isinstance(obj, dict):
        return f"item {index}"
    ident = next((str(obj[k]) for k in _ID_KEYS if obj.get(k)), "")
    desc = next((str(obj[k]).strip() for k in _TEXT_KEYS if obj.get(k)), "")
    if len(desc) > 60:
        desc = desc[:60] + "..."
    if ident and desc:
        return f"{ident} {desc}"
    return ident or desc or f"item {index}"


class ItemReporter:
    """Feeds streamed text through the scanner and emits one progress event
    per newly completed array element. Cheap enough to run per chunk."""

    def __init__(self, bus: ProgressBus | None, stage: str) -> None:
        self._bus = bus
        self._stage = stage
        self._reported = 0

    def feed(self, accumulated: str) -> None:
        if self._bus is None:
            return
        items = extract_array_items(accumulated)
        for raw in items[self._reported:]:
            self._reported += 1
            self._bus.emit(ProgressEvent(self._stage, "", "item", time.time(),
                                         None, item_label(raw, self._reported)))
