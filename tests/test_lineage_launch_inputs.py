"""Launch-input identity contract: JSON inputs bind canonical content
(serialization-only rewrites admit, substantive changes refuse), opaque
inputs stay byte-exact, ambiguous/non-finite JSON refuses, and pre-scheme
records get the explicit re-restore recovery instead of reinterpretation."""
import json

import pytest

from spec_generator import lineage
from spec_generator.lineage import LineageRefusal


def _record(tmp_path, name, path):
    lineage.record_inputs(tmp_path, task_id="t", session_id="s", generation="1",
                          input_hashes={name: lineage.hash_launch_input(path, name=name)})


def _verify(tmp_path, name, path):
    lineage.verify_inputs(lineage.load(tmp_path), {name: path})


def test_record_then_verify_roundtrip(tmp_path):
    path = tmp_path / "system-baseline.json"
    path.write_text(json.dumps({"mode": "greenfield", "repositories": []}, indent=2) + "\n")
    _record(tmp_path, "system-baseline.json", path)
    _verify(tmp_path, "system-baseline.json", path)


def test_serialization_only_rewrites_admit(tmp_path):
    path = tmp_path / "system-baseline.json"
    body = {"contract_version": "0.28.0", "baseline_id": "s-greenfield-v1",
            "system_name": "Surf school", "mode": "greenfield",
            "repositories": [], "immutable_decisions": []}
    path.write_text(json.dumps(body, indent=2) + "\n")
    _record(tmp_path, "system-baseline.json", path)
    # the generator's own startup rewrite: same content, no trailing newline
    path.write_text(json.dumps(body, indent=2))
    _verify(tmp_path, "system-baseline.json", path)
    # shuffled object keys + different whitespace are still the same content
    shuffled = dict(reversed(list(body.items())))
    path.write_text(json.dumps(shuffled, separators=(", ", " : ")))
    _verify(tmp_path, "system-baseline.json", path)


def test_array_order_is_content(tmp_path):
    path = tmp_path / "intent_model.json"
    path.write_text(json.dumps({"items": ["a", "b", "c"]}))
    _record(tmp_path, "intent_model.json", path)
    path.write_text(json.dumps({"items": ["c", "b", "a"]}))
    with pytest.raises(LineageRefusal, match="differs from the settled record"):
        _verify(tmp_path, "intent_model.json", path)


def test_substantive_change_refuses(tmp_path):
    path = tmp_path / "system-baseline.json"
    path.write_text(json.dumps({"system_name": "Surf school", "mode": "greenfield"}))
    _record(tmp_path, "system-baseline.json", path)
    path.write_text(json.dumps({"system_name": "Other product", "mode": "greenfield"}))
    with pytest.raises(LineageRefusal, match="differs from the settled record"):
        _verify(tmp_path, "system-baseline.json", path)


def test_malformed_json_refuses_at_record_and_verify(tmp_path):
    path = tmp_path / "conops.json"
    path.write_bytes(b'{"a": ')
    with pytest.raises(LineageRefusal, match="not well-formed JSON"):
        lineage.hash_launch_input(path, name="conops.json")
    # recorded while valid, corrupted before verify: explicit refusal, never a silent fallback
    path.write_text(json.dumps({"a": 1}))
    _record(tmp_path, "conops.json", path)
    path.write_bytes(b'{"a": ')
    with pytest.raises(LineageRefusal, match="not well-formed JSON"):
        _verify(tmp_path, "conops.json", path)


def test_duplicate_keys_refuse(tmp_path):
    path = tmp_path / "conops.json"
    path.write_bytes(b'{"a": 1, "a": 2}')
    with pytest.raises(LineageRefusal, match="ambiguous JSON"):
        lineage.hash_launch_input(path, name="conops.json")


@pytest.mark.parametrize("payload", [b'{"a": NaN}', b'{"a": Infinity}', b'{"a": -Infinity}', b'{"a": 1e999}'])
def test_non_finite_refuses(tmp_path, payload):
    path = tmp_path / "conops.json"
    path.write_bytes(payload)
    with pytest.raises(LineageRefusal, match="non-finite JSON"):
        lineage.hash_launch_input(path, name="conops.json")


def test_opaque_bytes_stay_byte_exact(tmp_path):
    path = tmp_path / "notes.bin"
    path.write_bytes(b"payload v1")
    _record(tmp_path, "notes.bin", path)
    _verify(tmp_path, "notes.bin", path)
    path.write_bytes(b"payload v1\n")  # serialization-tolerance never applies to opaque bytes
    with pytest.raises(LineageRefusal, match="differs from the settled record"):
        _verify(tmp_path, "notes.bin", path)


def test_pre_scheme_record_gets_explicit_rerestore_refusal(tmp_path):
    path = tmp_path / "intent_model.json"
    path.write_text(json.dumps({"session_id": "s1"}))
    lineage.record_inputs(tmp_path, task_id="t", session_id="s", generation="1",
                          input_hashes={"intent_model.json": lineage.sha256_file(path)})  # pre-scheme shape
    with pytest.raises(LineageRefusal, match="not reinterpreted"):
        _verify(tmp_path, "intent_model.json", path)


def test_scheme_tag_distinguishes_json_from_opaque(tmp_path):
    path = tmp_path / "intent_model.json"
    path.write_text(json.dumps({"a": 1}))
    tagged = lineage.hash_launch_input(path, name="intent_model.json")
    assert tagged.startswith("jsonv1:")
    opaque = lineage.hash_launch_input(path, name="payload.bin")
    assert opaque.startswith("sha256:") and opaque != tagged
