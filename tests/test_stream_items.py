"""Per-item progress: scanner correctness + the streamed invoke path keeps the
output contract byte-identical while emitting one event per completed item."""
import json

from spec_generator.progress import ProgressBus, stderr_printer
from spec_generator.stream_items import ItemReporter, extract_array_items, item_label


def test_flat_array_objects_with_nested_arrays():
    text = '[{"requirement_id": "R-1", "title": "Alpha", "acceptance": [{"criterion_id": "AC-1"}]}, {"requirement_id": "R-2", "title": "Beta"}]'
    items = extract_array_items(text)
    assert len(items) == 2
    assert json.loads(items[0])["requirement_id"] == "R-1"
    assert json.loads(items[1])["title"] == "Beta"


def test_root_object_with_two_arrays_tests_shape():
    text = '{"acceptance_tests": [{"test_id": "T-1"}], "unit_module_tests": [{"test_id": "T-2"}, {"test_id": "T-3"}]}'
    items = extract_array_items(text)
    assert [json.loads(i)["test_id"] for i in items] == ["T-1", "T-2", "T-3"]


def test_string_elements_review_fixes():
    items = extract_array_items('["tighten R-2 wording", "add a performance criterion"]')
    assert [json.loads(i) for i in items] == ["tighten R-2 wording", "add a performance criterion"]


def test_markdown_fence_and_braces_inside_strings_are_ignored():
    text = '```json\n[{"need_id": "N-1", "statement": "the {system} shall \\"survive\\" [brackets]"}]\n```'
    items = extract_array_items(text)
    assert len(items) == 1
    assert json.loads(items[0])["statement"] == 'the {system} shall "survive" [brackets]'


def test_trailing_partial_element_not_reported():
    full = '[{"need_id": "N-1"}, {"need_id": "N-2"}]'
    partial = full[:-12]  # N-2 object cut mid-way
    assert len(extract_array_items(partial)) == 1
    assert len(extract_array_items(full)) == 2


def test_incremental_feed_emits_each_item_once():
    bus = ProgressBus()
    seen = []
    bus.subscribe(seen.append)
    rep = ItemReporter(bus, "needs")
    rep.feed('[{"need_id": "N-1", "statement": "alp')
    assert seen == []  # element not closed yet
    rep.feed('[{"need_id": "N-1", "statement": "alpha"}, {"need_id": "N-2", "statement": "beta"}]')
    assert [e.detail for e in seen] == ["N-1 alpha", "N-2 beta"]
    rep.feed('[{"need_id": "N-1", "statement": "alpha"}, {"need_id": "N-2", "statement": "beta"}]')
    assert len(seen) == 2  # no duplicates on re-feed


def test_item_label_fallbacks():
    assert item_label('{"requirement_id": "R-4", "title": "Audit trail"}', 1) == "R-4 Audit trail"
    assert item_label('"tighten the wording of R-2"', 2).startswith("tighten the wording")
    assert item_label('{"foo": 1}', 3) == "item 3"
    assert item_label('not json', 4) == "item 4"
    long_str = '"' + "x" * 100 + '"'
    assert item_label(long_str, 5).endswith("...")


def test_printer_renders_item_lines(capsys):
    bus = ProgressBus()
    bus.subscribe(stderr_printer())
    ItemReporter(bus, "needs").feed('[{"need_id": "N-1", "statement": "alpha"}]')
    out = capsys.readouterr().err
    assert "     - N-1 alpha" in out


def test_invoke_streams_and_preserves_output_contract(monkeypatch):
    """The streamed path returns EXACTLY the concatenated completion (what a
    one-shot invoke would have produced) and emits item events mid-flight."""
    from spec_generator import graph, model_config
    from spec_generator.telemetry import UsageLog

    payload = '[{"need_id": "N-1", "statement": "alpha"}, {"need_id": "N-2", "statement": "beta"}]'
    cuts = [payload[:17], payload[17:40], payload[40:]]  # split mid-element

    class FakeChunk:
        def __init__(self, text):
            self.content = text
            self.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    class FakeModel:
        def stream(self, messages):
            for c in cuts:
                yield FakeChunk(c)

    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: FakeModel())
    bus = ProgressBus()
    seen = []
    bus.subscribe(seen.append)
    log = UsageLog("s-test")
    result = graph._invoke("needs", "anything", log, bus)
    assert result == payload  # byte-identical to the one-shot contract
    assert [e.detail for e in seen] == ["N-1 alpha", "N-2 beta"]
    assert log.records[0].prompt_tokens == 10



def test_keyed_fills_stream_one_item_per_value():
    """2026-09-09 live regression: strict per-skeleton fills changed arrays to
    a keyed object and silently removed test-stage item lines."""
    text = '{"fills": {"T-1": {"description": "first", "runner": "http", "definition": {"url": "/"}}, "T-22": {"description": "last", "runner": "cli", "definition": {"command": "x"}}}}'
    items = extract_array_items(text)
    assert len(items) == 2
    assert json.loads(items[0]) == {"T-1": {"description": "first", "runner": "http", "definition": {"url": "/"}}}
    assert [item_label(x, i + 1) for i, x in enumerate(items)] == ["T-1 first", "T-22 last"]


def test_keyed_fills_incremental_stream_emits_each_once():
    bus = ProgressBus()
    seen = []
    bus.subscribe(seen.append)
    rep = ItemReporter(bus, "tests")
    rep.feed('{"fills": {"T-1": {"description": "fir')
    assert seen == []
    full = '{"fills": {"T-1": {"description": "first"}, "T-2": {"description": "second"}}}'
    rep.feed(full)
    rep.feed(full)
    assert [e.detail for e in seen] == ["T-1 first", "T-2 second"]


def test_nested_ordinary_object_is_not_misreported_as_items():
    assert extract_array_items('{"stack": {"language": "python", "framework": "fastapi"}}') == []


def test_invoke_reports_keyed_test_fills(monkeypatch):
    from spec_generator import graph, model_config
    from spec_generator.telemetry import UsageLog

    payload = '{"fills": {"T-1": {"description": "one"}, "T-2": {"description": "two"}}}'

    class Chunk:
        usage_metadata = {}
        def __init__(self, c): self.content = c
    class Model:
        def stream(self, messages):
            for i in range(0, len(payload), 13):
                yield Chunk(payload[i:i + 13])
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: Model())
    bus = ProgressBus()
    seen = []
    bus.subscribe(seen.append)
    assert graph._invoke("tests", "anything", UsageLog("s"), bus) == payload
    assert [e.detail for e in seen] == ["T-1 one", "T-2 two"]
