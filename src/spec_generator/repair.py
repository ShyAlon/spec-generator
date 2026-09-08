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


def _canon(v: str) -> str:
    return v.strip().lower().replace("-", "_").replace(" ", "_")


def normalize_enums(data: dict, model_cls) -> dict:
    """Small models spell enum values loosely ('non-functional', 'Functional').
    For array-of-object fields whose item schema has enum-typed properties,
    snap each string to the closest declared enum value (case/separator
    insensitive). Only exact canonical matches are changed; unknown values
    stay put and fail validation loudly."""
    props = model_cls.model_json_schema().get("properties", {})
    defs = model_cls.model_json_schema().get("$defs", {})
    out = dict(data)
    for name, spec in props.items():
        items = spec.get("items")
        if not (isinstance(items, dict) and isinstance(out.get(name), list)):
            continue
        ref = items.get("$ref", "")
        item_schema = defs.get(ref.split("/")[-1], {}) if ref else items
        item_props = item_schema.get("properties", {})
        enum_fields = {}
        for pn, ps in item_props.items():
            if not isinstance(ps, dict):
                continue
            target = ps
            if isinstance(ps.get("$ref"), str):  # enum via $defs reference
                target = defs.get(ps["$ref"].split("/")[-1], {})
            if isinstance(target.get("enum"), list):
                enum_fields[pn] = [str(v) for v in target["enum"]]
        if not enum_fields:
            continue
        fixed_items = []
        for item in out[name]:
            if isinstance(item, dict):
                item = dict(item)
                for pn, allowed in enum_fields.items():
                    v = item.get(pn)
                    if isinstance(v, str):
                        for a in allowed:
                            if _canon(v) == _canon(a):
                                item[pn] = a
                                break
            fixed_items.append(item)
        out[name] = fixed_items
    return out
