import json

from spec_generator.telemetry import UsageLog, jsonl_sink


def test_written_usage_row_carries_cost_usd(tmp_path):
    path = tmp_path / "usage.jsonl"
    log = UsageLog("s1", sink=jsonl_sink(path))
    log.record("srs", "deepseek", "deepseek-flash", {"prompt_tokens": 1000, "completion_tokens": 500})
    row = json.loads(path.read_text().splitlines()[0])
    assert "cost_usd" in row and row["cost_usd"] > 0
