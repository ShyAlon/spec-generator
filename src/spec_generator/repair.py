"""Shape-drift repair for model-produced JSON before contract validation.

Small/local models drift shape under pressure: a list field comes back as {},
a dict-of-lists comes back with bare scalars inside. These are mechanical
drifts with an unambiguous repair - coercion is safe because it only fires
when the value cannot match the schema as-is. Genuinely missing or wrong
CONTENT (a field invented, a wrong statement) is not repairable here and
goes to the validation-error retry loop in the distillation node.
"""
from __future__ import annotations


def _wants(spec: dict) -> set[str]:
    types = set()
    if isinstance(spec.get("type"), str):
        types.add(spec["type"])
    for s in spec.get("anyOf", []) + spec.get("oneOf", []):
        if isinstance(s, dict) and isinstance(s.get("type"), str):
            types.add(s["type"])
    return types


def coerce_for_model(data: dict, model_cls) -> dict:
    """Return a copy of data with mechanical shape drift repaired against the
    model's JSON schema. Only empty/scalar mismatches are touched."""
    props = model_cls.model_json_schema().get("properties", {})
    out = dict(data)
    for name, spec in props.items():
        if name not in out:
            continue
        v = out[name]
        wants = _wants(spec)
        if "array" in wants:
            if v is None or v == {}:
                out[name] = []
            elif isinstance(v, dict):
                out[name] = list(v.values())
            elif isinstance(v, str):
                out[name] = [v]
        elif "object" in wants:
            if v is None or v == []:
                out[name] = {}
            elif isinstance(v, dict):
                ap = spec.get("additionalProperties")
                if isinstance(ap, dict) and ap.get("type") == "array":
                    out[name] = {
                        k: (val if isinstance(val, list)
                            else [val] if isinstance(val, (int, str)) else [])
                        for k, val in v.items()
                    }
    return out
