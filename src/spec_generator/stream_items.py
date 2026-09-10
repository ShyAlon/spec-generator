"""Per-item progress during a stage - observability only, never control flow.

Stages ask the model for ONE JSON response (an array, or an object holding
arrays, or an object holding an object of keyed fills). This module streams
that response and reports each element the moment its JSON closes, so the
founder watches needs/requirements/specs land one line at a time instead of
staring at a silent stage line for minutes.

Reportable element shapes:
- elements of a bare top-level array (review fixes, bulk items)
- elements of an array that is a value in the root object (needs, legacy fills)
- VALUES of an object that is a value in the root object, wrapped with their
  key as {"<key>": <value>} (the per-skeleton fills shape: {"fills": {"R-1":
  {...}}}) - the key becomes the line's id. Scalar values inside nested
  objects are never reported (a "stack": {"language": ...} field is not an
  item).

The output contract is UNTOUCHED: same prompt, same model, same parameters;
streaming just delivers the same completion incrementally. The accumulated
text is exactly what _invoke would have returned, and downstream parsing,
guards, and validation see the identical string.
"""
from __future__ import annotations

import json
import time

from .progress import ProgressBus, ProgressEvent

_ID_KEYS = ("need_id", "requirement_id", "spec_id", "slug", "test_id", "criterion_id",
            "decision_id", "milestone_id", "item_id")
_TEXT_KEYS = ("title", "name", "statement", "description")


def extract_array_items(text: str) -> list[str]:
    """Return the raw JSON strings of every COMPLETE reportable element (see
    module docstring). Nested containers inside elements are not reported.
    String-aware: braces inside string literals do not count. A trailing
    partial element is simply not returned yet (the caller re-feeds the whole
    accumulation, so an undecided tail resolves on a later pass)."""
    items: list[str] = []
    stack: list[str] = []
    in_str = False
    esc = False
    elem: list | None = None  # [start_index, close_depth, key_span | None]
    pending: list | None = None  # [start, end, depth] - a key string awaiting its value

    def reportable() -> bool:
        return stack == ["["] or stack == ["{", "["] or stack == ["{", "{"]

    def pending_key() -> list | None:
        return pending if (pending is not None and pending[2] == len(stack)) else None

    def emit(start: int, end: int, key: list | None) -> None:
        if key is not None:
            items.append('{"' + text[key[0] + 1:key[1] - 1] + '": ' + text[start:end] + "}")
        else:
            items.append(text[start:end])

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
                    if stack and stack[-1] == "{":
                        # object context: a string here is a KEY or a scalar
                        # value - scalar values are not items; a key is held
                        # until its (object/array) value closes.
                        j = i + 1
                        while j < n and text[j] in " \t\r\n":
                            j += 1
                        if j < n and text[j] == ":":
                            pending = [elem[0], i + 1, len(stack)]
                        # buffer exhausted before the ':' arrives: decide on a
                        # later feed (the whole text is re-scanned each time).
                        elem = None
                    else:
                        emit(elem[0], i + 1, elem[2])
                        elem = None
            i += 1
            continue
        if c == '"':
            in_str = True
            if elem is None and reportable():
                elem = [i, len(stack), pending_key()]
                if elem[2] is not None:
                    pending = None
        elif c in "{[":
            if elem is None and reportable():
                elem = [i, len(stack), pending_key()]
                if elem[2] is not None:
                    pending = None
            stack.append(c)
        elif c in "}]":
            if stack:
                stack.pop()
            if pending is not None and len(stack) < pending[2]:
                pending = None
            if elem is not None and len(stack) == elem[1]:
                emit(elem[0], i + 1, elem[2])
                elem = None
        i += 1
    return items


def item_label(raw: str, index: int) -> str:
    """Human one-liner for a completed element: its id + short title, or a
    truncated string for plain-string elements (e.g. review fixes). A
    single-key wrapper ({"T-1": {...}} from the keyed fills shape) labels by
    its key."""
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
    if not ident and not desc and len(obj) == 1:
        key, val = next(iter(obj.items()))
        if isinstance(val, dict):
            ident = str(key)
            desc = next((str(val[k]).strip() for k in _TEXT_KEYS if val.get(k)), "")
    if len(desc) > 60:
        desc = desc[:60] + "..."
    if ident and desc:
        return f"{ident} {desc}"
    return ident or desc or f"item {index}"


class ItemReporter:
    """Feeds streamed text through the scanner and emits one progress event
    per newly completed element. Cheap enough to run per chunk."""

    def __init__(self, bus: ProgressBus | None, stage: str, on_item=None) -> None:
        self._bus = bus
        self._stage = stage
        self._on_item = on_item  # callback(raw_json, label) per closed item - per-item resume hook
        self._reported = 0
        self._last = time.time()  # an item's start is the previous item's end

    def feed(self, accumulated: str) -> None:
        if self._bus is None and self._on_item is None:
            return
        items = extract_array_items(accumulated)
        for raw in items[self._reported:]:
            self._reported += 1
            now = time.time()
            label = item_label(raw, self._reported)
            if self._bus is not None:
                self._bus.emit(ProgressEvent(self._stage, "", "item", now,
                                             now - self._last, label))
            if self._on_item is not None:
                self._on_item(raw, label)
            self._last = now
