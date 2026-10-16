"""Technical-choice prose fit: word boundary only, bounded, original preserved, never touches ids/values/titles."""
import json

from spec_generator import model_config
from spec_generator.graph import _fit_prose_overage, _fit_text_within

BUDGET = {"title": 100, "purpose": 300, "default_value": 200, "reason": 400,
          "options[].implication": 400}
SENT = "Self-hosting on a single VPS keeps the product self-contained and cheap to maintain, which matches the founder's stated preference. "


def _data(reason):
    return {"title": "Deployment", "purpose": "p", "default_value": "VPS", "reason": reason,
            "requirement_ids": ["FR-1"], "options": [{"value": "VPS", "label": "VPS", "implication": "i"}]}


def test_small_overage_cut_at_word_boundary_keeps_original():
    text = SENT * 3 + "Last sentence that pushes it over."
    assert 400 < len(text) <= 500
    out, trims = _fit_prose_overage(_data(text), BUDGET)
    fitted = out["reason"]
    assert len(fitted) <= 400 and text.startswith(fitted)
    assert text[len(fitted):][0] in " ,;:-" or text[len(fitted)].isspace()
    assert len(trims) == 1 and trims[0]["original"] == text and trims[0]["fitted"] == fitted
    assert trims[0]["field"] == "reason" and trims[0]["before"] == len(text)


def test_cut_is_mid_sentence_not_sentence_end_and_adds_nothing():
    text = ("word " * 90).strip()
    assert 400 < len(text) <= 500
    out, _ = _fit_prose_overage(_data(text), BUDGET)
    assert out["reason"] == ("word " * 80).strip()[:len(out["reason"])] and out["reason"].endswith("word")
    assert len(out["reason"]) <= 400


def test_large_overage_is_left_for_retry():
    text = SENT * 6
    assert len(text) > 500
    out, trims = _fit_prose_overage(_data(text), BUDGET)
    assert out["reason"] == text and trims == []


def test_in_limit_text_and_protected_fields_untouched():
    data = _data("short reason")
    data["default_value"] = "x" * 230
    out, trims = _fit_prose_overage(data, BUDGET)
    assert out["reason"] == "short reason" and out["default_value"] == "x" * 230 and trims == []


def test_input_not_mutated_and_no_boundary_left_alone():
    data = _data("x" * 450)
    out, trims = _fit_prose_overage(data, BUDGET)
    assert data["reason"] == "x" * 450 and out["reason"] == "x" * 450 and trims == []
    assert _fit_text_within("x" * 450, 400) is None


def test_original_text_is_preserved_byte_exact_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(model_config, "_llm_log_sink", lambda: str(tmp_path / "llm.log"))
    text = "alpha beta, " * 38 + "tail \u00e9 text"
    out, trims = _fit_prose_overage(_data(text), BUDGET)
    assert trims
    model_config._capture_fitted_output("architecture", "technical-choice:deployment", trims)
    manifest = (tmp_path / "fitted-llm-outputs" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(manifest[0])
    assert row["chars_before"] == len(text) and row["chars_after"] == len(out["reason"])
    assert open(row["path"], encoding="utf-8").read() == text
