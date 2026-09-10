"""Generation pipeline as a LangGraph state machine.

LangGraph earns its place here (ADR-0003): staged drafts with independent
review passes (3 drafts + 2 review rounds is table stakes), resumable
background execution (Vercel's 300s cap means this runs as a job, not a
request), and a deterministic assembly tail that no LLM touches.
"""
from __future__ import annotations

import json
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver

from .progress import ProgressBus, track
from langgraph.graph import END, StateGraph

from vmodel_contracts import (
    AcceptanceTest,
    ArchitectureOverview,
    IntentModel,
    ItemSpec,
    PunchListItem,
    RequirementSet,
    SpecBundle,
    SpecManifest,
    SRSDocument,
    StakeholderNeedsSet,
    text_qa,
)

from pydantic import ValidationError

from . import model_config
from .tools import tools_enabled
from .skeleton import (ModelStructureError, assign_architecture_ids,
                                   build_srs_skeleton, build_test_skeleton,
                                   merge_srs_fills, merge_test_fills)
from .assembly import build_traceability, order_milestones, validate_traceability
from .prompts import METHODOLOGY, STAGE_PROMPTS
from .telemetry import UsageLog

REVIEW_PASSES = 2  # independent review rounds; fixes applied by re-running the flagged stage's prompt with the fix list


class GenState(TypedDict, total=False):
    intent_model: dict
    needs: dict
    srs: dict
    requirements: dict
    guard_flags: list[str]
    architecture: dict
    items: list[dict]
    tests: list[dict]
    review_round: int
    review_fixes: list[str]
    bundle: dict
    violations: list[str]
    gate_feedback: dict[str, str]  # stage -> founder feedback from a review gate (ADR-0019)


def _invoke(stage: str, payload: str, log: UsageLog, progress: ProgressBus | None = None,
            feedback: str | None = None, schema: dict | None = None,
            stream_items: bool = True, item_label: str | None = None,
            on_item=None) -> str:
    """One model call per stage, STREAMED so the console can show each item as
    its JSON closes (same prompt, same parameters, same completion - progress
    is observability only, never a change to what is generated).
    item_label: the call IS one item (per-slot micro-call) - emit a start line
    when it begins and a completion line (with token count when the provider
    reports it) when it ends, so a hung item is visible in the log.
    on_item: callback(raw_json, label) fired as each streamed item closes -
    the per-item resume hook (partial persistence mid-call)."""
    import time as _t

    from langchain_core.messages import HumanMessage, SystemMessage

    from .progress import ProgressEvent
    from .stream_items import ItemReporter

    model = model_config.build_chat_model(stage, schema=schema)
    # ADR-0022: per-slot micro-calls skip the per-token item scanner (their
    # output is one small object; the item_label lines cover them instead).
    reporter = ItemReporter(progress if stream_items else None, stage, on_item=on_item)
    t0 = _t.time()
    if item_label and progress is not None:
        progress.emit(ProgressEvent(stage, "", "item_started", t0, None, item_label))
    messages = [SystemMessage(content=METHODOLOGY), HumanMessage(content=payload)]
    if feedback:
        messages.append(HumanMessage(content=feedback))
    chunks: list[str] = []
    usage: dict = {}
    for chunk in model_config.stream_checked(model, messages, stage):
        text = chunk.content if isinstance(chunk.content, str) else "".join(
            b.get("text", "") for b in chunk.content if isinstance(b, dict))
        chunks.append(text)
        reporter.feed("".join(chunks))
        usage = getattr(chunk, "usage_metadata", None) or usage
    cfg = model_config.for_stage(stage)
    log.record(stage, cfg.provider, cfg.model, usage)
    if item_label and progress is not None:
        tok = (usage or {}).get("total_tokens")
        progress.emit(ProgressEvent(stage, "", "item", _t.time(), _t.time() - t0,
                                    item_label + (f" · {tok} tok" if tok else "")))
    return "".join(chunks)


def _extract_json(text: str) -> str:
    """Tolerant extraction (ported from interview-engine's run-4 fix): strip
    fences, else cut the first balanced {...} or [...] out of prose. Returns
    the candidate string; raises ValueError when no balanced value exists."""
    candidate = text.strip().strip("`").removeprefix("json").strip()
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass
    starts = [(text.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if text.find(o) >= 0]
    for start, open_c, close_c in sorted(starts):
        depth = 0
        in_str = esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == open_c:
                depth += 1
            elif c == close_c:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    raise ValueError("no balanced JSON value found in the model output")


def _json(text: str, stage: str = "unknown"):
    try:
        return json.loads(_extract_json(text))
    except (json.JSONDecodeError, ValueError) as e:
        from .errors import translate_json_error

        raise translate_json_error(e, stage) from e


def _contract_errors(exc) -> str:
    """Compact one-line-per-error rendering for retry feedback and the
    human-readable failure: 'functional.0.title: Field required'."""
    if isinstance(exc, ModelStructureError):
        return str(exc)
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in exc.errors()[:5])


def _call_json(stage: str, payload: str, log: UsageLog, progress: ProgressBus | None = None,
               validate=None, schema: dict | None = None, stream_items: bool = True,
               item_label: str | None = None, on_item=None):
    """Every authoring stage goes through here: invoke -> tolerant parse ->
    CONTRACT VALIDATION -> ONE automatic retry with the failure as feedback
    (the interview-engine pattern, ADR-0015 doctrine: transient model flakes
    auto-recover; the founder never manually retries a known class). The
    validator is the stage's contract model: grammar (structured outputs)
    guarantees shape only, so pattern/enum/min-length drift still lands here -
    catching it at the stage means a flake costs one retry, never a crashed
    gate. On the second failure the same human-readable error fires, now with
    a raw-output snippet."""
    import time as _t

    from .progress import ProgressEvent

    fb = None
    last_raw = ""
    for attempt in (1, 2):
        raw = _invoke(stage, payload, log, progress, feedback=fb, schema=schema,
                      stream_items=stream_items, item_label=item_label, on_item=on_item)
        last_raw = raw
        contract_err = None
        try:
            data = json.loads(_extract_json(raw))
        except (json.JSONDecodeError, ValueError) as e:
            data = None
            parse_err = e
        else:
            if validate is None:
                return data
            try:
                checked = validate(data)
                return data if checked is None else checked
            except (ValidationError, ModelStructureError) as e:
                contract_err = e
        from .errors import UserFacingError

        snippet = raw.strip()[:200].replace("\n", " ")
        if attempt == 2:
            if contract_err is not None:
                raise UserFacingError(
                    f"stage '{stage}': the model's output does not satisfy the contract (an automatic retry did not fix it).",
                    f"contract errors: {_contract_errors(contract_err)}. Raw output began: {snippet!r}",
                    "rerun the same command - the checkpoint resumes at this stage; if it repeats, route the stage "
                    f"to a stronger model (SPEC_GENERATOR_MODEL_{stage.upper()} env override) - and report it: "
                    "schema-constrained output should not drift structurally.",
                ) from contract_err
            raise UserFacingError(
                f"stage '{stage}': the model returned output that is not valid JSON (an automatic retry did not fix it).",
                f"smaller/local models often wrap JSON in prose, drop structure, or truncate. Raw output began: {snippet!r}",
                "rerun the same command - the checkpoint resumes at this stage; if it repeats, route the stage "
                f"to a stronger model (SPEC_GENERATOR_MODEL_{stage.upper()} env override).",
            ) from parse_err
        if progress is not None:
            progress.emit(ProgressEvent(stage, "", "item", _t.time(), None,
                                        ("model output failed the contract" if contract_err is not None
                                         else "model output was not valid JSON") + " - retrying once with feedback"))
        if contract_err is not None:
            fb = ("Your previous reply failed the output contract: " + _contract_errors(contract_err) + ". "
                  "Return ONLY the complete corrected JSON payload - every required field present at every "
                  "level, ids and enums exactly as specified.")
        else:
            fb = ("Your previous reply was not valid JSON (prose-wrapped, unbalanced, or truncated). "
                  "Return ONLY the complete JSON payload, no prose, no commentary, balanced to the end.")


def srs_document_data(intent_model: dict, srs_body: dict) -> dict:
    """Full SRSDocument data from the LLM-authored body: structure (definitions,
    constraints, assumptions, open questions) is carried DETERMINISTICALLY from
    the intent model - the LLM never re-copies it. Used at assembly AND at the
    review gate (validating hand edits)."""
    intent = IntentModel.model_validate(intent_model)
    data = dict(srs_body)
    data["definitions"] = intent.glossary
    data["constraints"] = intent.constraints
    data["assumptions"] = [a.model_dump(mode="json") for a in intent.assumption_items]
    data["open_questions"] = intent.open_questions
    return data


def _check(fn):
    """A validator that only RAISES (returns no normalized form): wrap it so
    its pydantic return value is not mistaken for normalized data."""
    def go(d):
        fn(d)
        return None
    return go


def _strip_pipeline_owned(data):
    """The LLM body must not carry pipeline-owned fields; a model that copies
    them anyway is tolerated (the pipeline overwrites), anything else extra
    still fails the contract's extra=forbid."""
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if k not in ("session_id", "contract_version")}
    return data


_PROMPT_HIDDEN = ("session_id", "intent_model_ref", "contract_version")


def _prompt_view(data):
    """State as the MODEL may see it: pipeline-owned bookkeeping stripped
    recursively. session ids are plumbing; a local model that sees one reuses
    it as content (the 2026-09-09 live failure: item specs named
    '20260909-094216-e900 Performance')."""
    if isinstance(data, dict):
        return {k: _prompt_view(v) for k, v in data.items() if k not in _PROMPT_HIDDEN}
    if isinstance(data, list):
        return [_prompt_view(v) for v in data]
    return data


def checkpoint_serde():
    """Checkpoint (de)serializer with the vmodel_contracts enums registered.
    LangGraph warns on READ - 'Deserializing unregistered type
    vmodel_contracts.spec.SpecItemKind from checkpoint ... will be blocked in
    a future version' - because pydantic model_dump (python mode) leaves enum
    instances in state. New state is dumped mode='json' (plain strings, no
    enums); this registration silences the warning and keeps PRE-FIX
    checkpoints resumable under future strict msgpack mode."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=[
        ("vmodel_contracts.spec", "SpecItemKind"),
        ("vmodel_contracts.requirements", "RequirementKind"),
        ("vmodel_contracts.requirements", "Priority"),
        ("vmodel_contracts.verification", "VModelLevel"),
        ("vmodel_contracts.verification", "DivergenceKind"),
        ("vmodel_contracts.verification", "VerificationVerdict"),
        ("vmodel_contracts.architecture", "DecisionDriver"),
    ])




def _tech_constraint_block(intent_model: dict) -> str:
    """The founder's ecosystem preferences confine the architecture stage
    (ADR-0019). required binds; preferred guides; ADRs name their driver."""
    prefs = intent_model.get("tech_preferences") or []
    if not prefs:
        return ""
    lines = ["FOUNDER TECHNOLOGY CONSTRAINTS - confine your stack, library, and "
             "infrastructure choices to these (SRS stays solution-agnostic; here "
             "naming technology IS the job):"]
    for p in prefs:
        if p.get("strength") == "required":
            lines.append(f"- REQUIRED {p.get('area')}: {p.get('preference')} (founder-mandated; do not deviate)")
    for p in prefs:
        if p.get("strength") != "required":
            lines.append(f"- preferred {p.get('area')}: {p.get('preference')} (affinity - follow it, or argue otherwise in an ADR)")
    lines.append("Every ADR names the constraint that drove it; where the founder is "
                 "agnostic, state that the choice is yours and why.")
    return "\n\n" + "\n".join(lines)


def _srs_fill_call(state: GenState, extra: str = ""):
    """The ADR-0020 SRS fill call, shared by node_srs and node_apply_fixes:
    same skeleton, same prompt, same merge validator - the apply-fixes path
    re-fires the FILL stage with the review fixes, never the retired
    full-document shape (the 2026-09-09 live failure: grammar said fills,
    validator said document). The grammar is per-skeleton (every slot id a
    required property) and the FORMAT CONTRACT closes the prompt - the 14b
    follows the last frame it read, and a long REVIEW FIXES / PREVIOUS SRS
    tail otherwise buries it (same-day live failure: purpose echoed the
    V-model methodology instead of the founder's product)."""
    from .schemas import srs_fill_schema

    slots = build_srs_skeleton(state["needs"])
    payload = (STAGE_PROMPTS["srs"] + "\n\nREQUIREMENT SLOTS (fill prose for each, keyed by requirement_id):\n"
               + json.dumps(slots) + "\n\nNEEDS:\n" + json.dumps(_prompt_view(state["needs"]))
               + "\n\nINTENT:\n" + json.dumps(_prompt_view(state["intent_model"])) + extra)
    ids = [sl["requirement_id"] for sl in slots]
    payload += ("\n\nFORMAT CONTRACT (final instruction - overrides anything above that conflicts): "
                "return one STRICT JSON object {purpose, scope_summary, fills: {...}} where fills has "
                "EXACTLY these keys, one per slot: " + json.dumps(ids) + ". Each fill is "
                "{title, statement, priority, acceptance: [statements]}. purpose and scope_summary "
                "describe THE FOUNDER'S PRODUCT in stakeholder language - never the V model, this "
                "pipeline, or these instructions.")

    def _validate(d):
        merged = merge_srs_fills(slots, d)
        sid = state["intent_model"]["session_id"]
        body = {**merged, "session_id": sid, "intent_model_ref": sid}
        SRSDocument.model_validate(srs_document_data(state["intent_model"], body))
        return merged

    return payload, _validate, srs_fill_schema(slots)


def _legacy_arch_fill(payload: dict) -> dict:
    """Pre-0.8.0 checkpoints hold decisions without the ADR-0021 fields;
    synthesize the honest legacy form (same rule as vmodel_contracts.migrate)
    so resuming an old session past the architecture stage cannot crash
    assembly. New production never takes this path: the stage validator
    requires a real driver before state is written."""
    out = dict(payload)
    decisions = []
    for d in out.get("decisions") or []:
        dd = dict(d)
        dd.setdefault("driver", None)
        dd.setdefault("options_considered", [])
        decisions.append(dd)
    out["decisions"] = decisions
    return out


def _check_decision_drivers(arch: ArchitectureOverview, intent_model: dict) -> None:
    """ADR-0021: no arbitrary decisions. The contract requires the FIELDS
    present (driver nullable only for legacy migration); the stage requires a
    REAL driver on every new decision, and a founder_constraint driver must
    cite a TechPreference that actually exists in the intent model - the
    mechanical form of ADR-0019's 'ADRs name the constraint that drove it'.
    Options have no floor (founder call 2026-09-09): under a mandate, empty
    is honest."""
    prefs = [p.get("preference", "") for p in (intent_model.get("tech_preferences") or [])]
    for d in arch.decisions:
        if d.driver is None:
            raise ModelStructureError(
                f"{d.decision_id} '{d.title}': no driver - classify why the chosen option won "
                f"(founder_constraint | ecosystem_maturity | community_adoption | team_fit | simplicity).")
        if d.driver.value == "founder_constraint" and prefs:
            text = (d.title + " " + d.decision + " " + d.rationale).lower()
            if not any(p.lower() in text for p in prefs if p):
                raise ModelStructureError(
                    f"{d.decision_id} '{d.title}': driver founder_constraint but the decision cites none "
                    f"of the founder's stated preferences ({', '.join(prefs)}) - name the one that drove it.")


def _check_item_refs(entries: list, req_ids: list[str]) -> None:
    """Shared reference/coverage checks for the items stage (bulk specs and
    ADR-0022 plans): unique ids, references resolve, deps point at emitted
    specs, every requirement covered. Failures are NAMED so the automatic
    retry's feedback says exactly what to fix."""
    spec_ids = [e.get("spec_id") for e in entries]
    dupes = sorted({s for s in spec_ids if spec_ids.count(s) > 1})
    if dupes:
        raise ModelStructureError(f"duplicate spec ids: {dupes} - every spec needs a unique spec_id")
    known = set(req_ids)
    emitted = set(spec_ids)
    for e in entries:
        unknown = sorted(set(e.get("requirement_ids") or []) - known)
        if unknown:
            raise ModelStructureError(
                f"{e.get('spec_id')} claims unknown requirements {unknown} - valid ids: {req_ids}")
        bad_deps = sorted(set(e.get("depends_on") or []) - emitted)
        if bad_deps:
            raise ModelStructureError(
                f"{e.get('spec_id')} depends on unknown specs {bad_deps} - depends_on may only "
                f"reference spec ids from this same answer")
    covered = {rid for e in entries for rid in (e.get("requirement_ids") or [])}
    uncovered = [r for r in req_ids if r not in covered]
    if uncovered:
        raise ModelStructureError(
            f"requirements covered by no item spec: {uncovered} - cover every requirement: "
            f"add a spec or extend an existing spec's requirement_ids")


def _srs_overview_payload(state: GenState, feedback: str | None = None) -> str:
    """ADR-0022: the purpose/scope micro-call. Its own small prompt - no slot
    JSON, no fix tail - so the domain frame cannot get buried."""
    previous = state.get("srs") or {}
    payload = (STAGE_PROMPTS["srs_overview"] + "\n\nNEEDS:\n" + json.dumps(_prompt_view(state["needs"]))
               + "\n\nINTENT:\n" + json.dumps(_prompt_view(state["intent_model"])))
    if feedback:
        payload += "\n\n" + feedback
    if previous:
        payload += ("\n\nPREVIOUS purpose/scope (revise them):\n"
                    + json.dumps({"purpose": previous.get("purpose"),
                                  "scope_summary": previous.get("scope_summary")}))
    payload += ("\n\nFORMAT CONTRACT (final instruction - overrides anything above that conflicts): "
                "return one STRICT JSON object {purpose, scope_summary} describing THE FOUNDER'S "
                "PRODUCT in stakeholder language, from the INTENT above - never the V model, this "
                "pipeline, or these instructions.")
    return payload


def _srs_slot_payload(state: GenState, slot: dict, feedback: str | None = None) -> str:
    """ADR-0022: one requirement fill per call - slot, its need, slim intent
    context, and (on revision) only THIS slot's previous fill. Small prompt,
    one responsibility, contract closes."""
    rid = slot["requirement_id"]
    need = next((n for n in state["needs"].get("needs") or []
                 if n.get("need_id") in (slot.get("need_ids") or [])), None)
    intent = state["intent_model"]
    slim = {k: intent.get(k) for k in
            ("problem_statement", "core_value_proposition", "in_scope", "out_of_scope", "constraints")}
    payload = (STAGE_PROMPTS["srs_slot"] + "\n\nTHE SLOT (fill exactly this one):\n" + json.dumps(slot)
               + "\n\nTHE NEED BEHIND IT:\n" + json.dumps(_prompt_view(need))
               + "\n\nINTENT (context):\n" + json.dumps(_prompt_view(slim)))
    if feedback:
        payload += "\n\n" + feedback
    prev = None
    for sec in ("functional", "non_functional"):
        for r in ((state.get("srs") or {}).get(sec) or []):
            if r.get("requirement_id") == rid:
                prev = {"title": r.get("title"), "statement": r.get("statement"),
                        "priority": r.get("priority"),
                        "acceptance": [c.get("statement") for c in (r.get("acceptance") or [])]}
    if prev:
        payload += "\n\nPREVIOUS FILL (revise it):\n" + json.dumps(prev)
    payload += ("\n\nFORMAT CONTRACT (final instruction - overrides anything above that conflicts): "
                f"return one STRICT JSON object {{title, statement, priority, acceptance}} - the fill "
                f"for slot {rid} ONLY. priority is must|should|could; acceptance is a non-empty list "
                f"of plain observable outcome statements. The text describes THE FOUNDER'S PRODUCT - "
                "never the V model, this pipeline, or these instructions.")
    return payload


def _overview_validator():
    def _v(d):
        if not isinstance(d, dict) or not str(d.get("purpose") or "").strip() \
                or not str(d.get("scope_summary") or "").strip():
            raise ModelStructureError("purpose and scope_summary are required non-empty strings")
        return None
    return _v


def _slot_fill_validator(rid: str):
    def _v(d):
        f = d if isinstance(d, dict) else {}
        if set(f) == {rid} and isinstance(f.get(rid), dict):
            f = f[rid]  # tolerate a keyed answer {rid: fill} despite the flat contract
        missing = [k for k in ("title", "statement", "priority", "acceptance") if k not in f]
        if missing:
            raise ModelStructureError(
                f"{rid}: fill missing fields {missing} - return exactly "
                "{title, statement, priority, acceptance}")
        if f.get("priority") not in ("must", "should", "could"):
            raise ModelStructureError(f"{rid}: priority must be must|should|could, got {f.get('priority')!r}")
        if not isinstance(f.get("acceptance"), list) or not f["acceptance"]:
            raise ModelStructureError(f"{rid}: acceptance must be a non-empty list of statements")
        if not str(f.get("title") or "").strip() or not str(f.get("statement") or "").strip():
            raise ModelStructureError(f"{rid}: title and statement must be non-empty")
        return f  # normalized (unwrapped) fill
    return _v


def _items_plan_payload(state: GenState, req_ids: list[str]) -> str:
    return (STAGE_PROMPTS["items_plan"] + "\n\nSRS:\n" + json.dumps(_prompt_view(state["srs"]))
            + "\n\nARCHITECTURE:\n" + json.dumps(_prompt_view(state["architecture"]))
            + "\n\nFORMAT CONTRACT (final instruction - overrides anything above that conflicts): "
              "return one STRICT JSON array of spec plans {spec_id, kind, title, requirement_ids, "
              "depends_on} - NO body_markdown, bodies are written one spec per call next. "
              "requirement_ids may contain ONLY these requirement ids: "
            + json.dumps(req_ids) + " - and every one of them must appear in at least one plan's "
              "requirement_ids. spec_id values are short lowercase slugs (e.g. SPEC-api, SPEC-orders); "
              "ids and titles NEVER contain session ids, timestamps, or run identifiers. The plan is "
              "for THE FOUNDER'S PRODUCT - never this pipeline.")


def _items_plan_validator(req_ids: list[str]):
    import re as _re

    from vmodel_contracts import SpecItemKind

    def _v(d):
        if not isinstance(d, list) or not d:
            raise ModelStructureError("return a non-empty JSON array of spec plans")
        kinds = {k.value for k in SpecItemKind}
        for p in d:
            sid = p.get("spec_id") if isinstance(p, dict) else None
            if not sid or not _re.fullmatch(r"SPEC-[a-z0-9-]+", str(sid)):
                raise ModelStructureError(
                    f"invalid spec_id {sid!r} - a short lowercase slug like SPEC-api "
                    "(never a session id or timestamp)")
            if p.get("kind") not in kinds:
                raise ModelStructureError(f"{sid}: kind must be one of {sorted(kinds)}")
            if not str(p.get("title") or "").strip():
                raise ModelStructureError(f"{sid}: title required")
        _check_item_refs(d, req_ids)
        return None
    return _v


def _items_body_payload(state: GenState, spec: dict) -> str:
    reqs = [r for sec in ("functional", "non_functional") for r in (state["srs"].get(sec) or [])
            if r.get("requirement_id") in (spec.get("requirement_ids") or [])]
    return (STAGE_PROMPTS["items_body"] + "\n\nTHE SPEC (write its body):\n" + json.dumps(spec)
            + "\n\nTHE REQUIREMENTS IT COVERS:\n" + json.dumps(_prompt_view(reqs))
            + "\n\nARCHITECTURE:\n" + json.dumps(_prompt_view(state["architecture"]))
            + "\n\nFORMAT CONTRACT (final instruction - overrides anything above that conflicts): "
              f"return one STRICT JSON object {{body_markdown}} - the complete markdown spec for "
              f"{spec.get('spec_id')} ONLY: interface, data model, behaviors, error cases, "
              "non-functionals, written for a coding agent building THE FOUNDER'S PRODUCT - never "
              "this pipeline.")


def _items_body_validator(spec: dict):
    def _v(d):
        md = (d or {}).get("body_markdown") if isinstance(d, dict) else None
        if not str(md or "").strip():
            raise ModelStructureError(f"{spec.get('spec_id')}: body_markdown must be a non-empty markdown spec")
        ItemSpec.model_validate({**spec, "body_markdown": md})
        return {"body_markdown": md}
    return _v


def _items_payload(state: GenState, req_ids: list[str]) -> str:
    """The items prompt. The FORMAT CONTRACT closes it (the 14b follows the
    last frame it read): the reference enum, the coverage rule, and the
    id-discipline frame are the final words."""
    return (STAGE_PROMPTS["items"] + "\n\nSRS:\n" + json.dumps(_prompt_view(state["srs"]))
            + "\n\nARCHITECTURE:\n" + json.dumps(_prompt_view(state["architecture"]))
            + "\n\nFORMAT CONTRACT (final instruction - overrides anything above that conflicts): "
              "return one STRICT JSON array of spec objects {spec_id, kind, title, body_markdown, "
              "requirement_ids, depends_on}. requirement_ids may contain ONLY these requirement ids: "
            + json.dumps(req_ids) + " - and every one of them must appear in at least one spec's "
              "requirement_ids. spec_id values are short lowercase slugs (e.g. SPEC-api, SPEC-orders); "
              "ids and titles NEVER contain session ids, timestamps, or run identifiers. Specs are "
              "written for a coding agent building THE FOUNDER'S PRODUCT - never this pipeline.")


def _items_validator(req_ids: list[str]):
    """Stage-level items contract: each spec passes ItemSpec; spec ids are
    unique; every reference resolves (the grammar's enum already makes unknown
    requirement ids impossible when structured output is on - this catches the
    free-mode fallback); dependencies point at emitted specs; and EVERY
    requirement is covered by at least one spec. Failures are named, so the
    automatic retry's feedback says exactly what to fix."""
    def _validate(d):
        items = [ItemSpec.model_validate(_strip_pipeline_owned(i)) for i in d]
        _check_item_refs([i.model_dump() for i in items], req_ids)
        return None
    return _validate


def _gate_feedback(state: GenState, stage: str) -> str | None:
    return (state.get("gate_feedback") or {}).get(stage)


def _clear_feedback(state: GenState, stage: str) -> dict:
    return {k: v for k, v in (state.get("gate_feedback") or {}).items() if k != stage}


def build_graph(log: UsageLog, checkpointer=None, progress: ProgressBus | None = None,
                review: bool = False, persist=None, partials=None):
    """persist(stage, data, state): optional side-effect hook fired after each
    authoring stage completes - the CLI uses it to write the bundle
    incrementally, so a late failure leaves all completed work on disk.
    Pure side effect: resume semantics are unchanged (a re-run stage
    re-persists).
    partials: optional store with load(key)/save(key, data)/clear(key) for
    PER-ITEM resume inside the looped authoring stages (srs fills, items
    bodies, tests fills) - completed items are preserved on crash/Ctrl+C and
    skipped on rerun, the same way completed stages already are."""

    def _wrap(stage: str, label: str, detail=None):
        """detail: a string, or a callable(state) -> str. A callable resolves at
        node start, so the opening line can carry the stage's INPUT cardinality
        ('from 12 interview needs') - the count a founder watches items count
        up against. Output counts are never claimed: they are not knowable
        before the call."""
        def deco(fn):
            def inner(state: GenState) -> dict:
                d = detail(state) if callable(detail) else detail
                with track(progress, stage, label, d):
                    return fn(state)
            return inner
        return deco

    def _sn_count(state: GenState) -> int:
        return len(state["intent_model"].get("stakeholder_needs") or [])

    def _n_count(state: GenState) -> int:
        return len(state["needs"].get("needs") or [])

    def _req_count(state: GenState) -> int:
        srs = state["srs"]
        return len(srs.get("functional") or []) + len(srs.get("non_functional") or [])

    def _ac_count(state: GenState) -> int:
        srs = state["srs"]
        return sum(len(r.get("acceptance") or [])
                   for sec in ("functional", "non_functional") for r in (srs.get(sec) or []))

    def _run_srs_fill_stage(state: GenState, feedback: str | None = None) -> dict:
        """The ADR-0020 SRS fill stage, two harness shapes (ADR-0022):
        - default: one micro-call for purpose/scope + one per requirement slot
          (a bad fill fails one cheap call, never a long composite payload);
        - SPEC_GENERATOR_BULK=1: the original single composite call.
        Both end at the same merged, contract-validated body."""
        import time as _t

        from .progress import ProgressEvent
        from .schemas import srs_overview_schema, srs_slot_fill_schema

        slots = build_srs_skeleton(state["needs"])
        if model_config.bulk_fills_enabled():
            extra = ""
            if feedback:
                extra = "\n\n" + feedback
            if state.get("srs"):
                extra += "\n\nPREVIOUS SRS (revise it; slots and ids are unchanged):\n" + \
                    json.dumps(_prompt_view(state["srs"]))
            payload, _validate, schema = _srs_fill_call(state, extra)
            return _call_json("srs", payload, log, progress, validate=_validate, schema=schema)
        overview = _call_json("srs", _srs_overview_payload(state, feedback), log, progress,
                              validate=_overview_validator(), schema=srs_overview_schema(),
                              stream_items=False)
        done = (partials.load("srs_fills") or {}) if partials else {}
        need_hint = {n.get("need_id"): (n.get("statement") or "")[:50]
                     for n in state["needs"].get("needs") or []}
        fills = {}
        for slot in slots:
            rid = slot["requirement_id"]
            if rid in done:  # per-item resume: preserved from the crashed run
                fills[rid] = done[rid]
                continue
            hint = need_hint.get((slot.get("need_ids") or [""])[0], "")
            fills[rid] = _call_json("srs", _srs_slot_payload(state, slot, feedback), log, progress,
                                    validate=_slot_fill_validator(rid), schema=srs_slot_fill_schema(),
                                    stream_items=False,
                                    item_label=f"{rid} {hint}".strip())
            if partials:
                done[rid] = fills[rid]
                partials.save("srs_fills", done)
        body = {"purpose": overview["purpose"], "scope_summary": overview["scope_summary"],
                "fills": fills}
        merged = merge_srs_fills(slots, body)
        sid = state["intent_model"]["session_id"]
        SRSDocument.model_validate(srs_document_data(
            state["intent_model"], {**merged, "session_id": sid, "intent_model_ref": sid}))
        if partials:
            partials.clear("srs_fills")
        return merged

    def node_needs(state: GenState) -> dict:
        intent = IntentModel.model_validate(state["intent_model"])
        data = _call_json("needs", STAGE_PROMPTS["needs"] + "\n\nINTENT:\n" + json.dumps(_prompt_view(state["intent_model"])), log, progress,
                          validate=_check(lambda d: StakeholderNeedsSet.model_validate(
                              {**_strip_pipeline_owned(d), "session_id": intent.session_id})))
        data.setdefault("session_id", intent.session_id)
        if persist:
            persist("needs", data, state)
        return {"needs": data}

    def node_srs(state: GenState) -> dict:
        """The SRS stage: authors the standalone requirements document body.
        Structure (definitions, constraints, assumptions, open questions) is
        carried DETERMINISTICALLY from the intent model in assembly - the LLM
        authors purpose, scope_summary, and the sectioned requirements only.
        Mechanical guards (text_qa) flag solution language and invented detail."""
        # ADR-0020: code owns the skeleton (R-ids, kinds, trace links, section
        # assignment); the model fills prose per slot. Structure carried from
        # the intent model stays deterministic (ADR-0015) via srs_document_data.
        slots = build_srs_skeleton(state["needs"])  # the toolbox needs them; the fill call rebuilds identically
        if tools_enabled():
            from .tools import SrsToolbox, run_tool_loop
            from .prompts import TOOL_PROMPTS

            needs_ids = {n.get("need_id") for n in state["needs"].get("needs") or []}
            toolbox = SrsToolbox(slots, needs_ids)
            run_tool_loop("srs", toolbox,
                          TOOL_PROMPTS["srs"].format(skeleton=json.dumps(slots),
                                                     needs=json.dumps(_prompt_view(state["needs"])),
                                                     intent=json.dumps(_prompt_view(state["intent_model"]))),
                          log, progress)
        fb = _gate_feedback(state, "srs")
        feedback = None
        if fb:
            feedback = "FOUNDER FEEDBACK ON YOUR PREVIOUS DRAFT (apply it):\n" + fb
        data = _run_srs_fill_stage(state, feedback)
        data.setdefault("session_id", state["intent_model"]["session_id"])
        data.setdefault("intent_model_ref", state["intent_model"]["session_id"])
        # mechanical guards: solution-agnostic + grounded (ADR-0015 doctrine)
        founder_text = json.dumps({k: state["intent_model"].get(k) for k in
                                   ("problem_statement", "core_value_proposition", "in_scope")})
        needs_text = json.dumps(state["needs"])
        flags = []
        for section in ("functional", "non_functional"):
            for req in data.get(section) or []:
                for field in ("title", "statement"):
                    if text_qa.is_solution_language(str(req.get(field, ""))):
                        flags.append(f"{req.get('requirement_id', '?')}.{field}: solution language")
                    if not text_qa.grounded_in(str(req.get(field, "")), [founder_text, needs_text]):
                        flags.append(f"{req.get('requirement_id', '?')}.{field}: ungrounded (no founder vocabulary)")
                kept = []
                for ac in req.get("acceptance") or []:
                    if text_qa.is_solution_language(str(ac.get("statement", ""))):
                        flags.append(f"{ac.get('criterion_id', '?')}: solution-language criterion dropped")
                        continue
                    kept.append(ac)
                if kept or not req.get("acceptance"):
                    req["acceptance"] = kept
        if persist:
            persist("srs", data, state)
        return {"srs": data, "guard_flags": state.get("guard_flags", []) + flags,
                "gate_feedback": _clear_feedback(state, "srs")}

    def node_architecture(state: GenState) -> dict:
        payload = (STAGE_PROMPTS["architecture"] + "\n\nSRS:\n" + json.dumps(_prompt_view(state["srs"]))
                   + _tech_constraint_block(state["intent_model"]))
        fb = _gate_feedback(state, "architecture")
        if fb:
            payload += ("\n\nFOUNDER FEEDBACK ON YOUR PREVIOUS DRAFT (apply it):\n" + fb
                        + "\n\nPREVIOUS DRAFT:\n" + json.dumps(_prompt_view(state["architecture"])))
        def _validate_arch(d):
            merged = assign_architecture_ids(_strip_pipeline_owned(d))  # ADR-0020: code assigns ADR-n
            arch = ArchitectureOverview.model_validate(
                {**merged, "session_id": state["intent_model"]["session_id"]})
            _check_decision_drivers(arch, state["intent_model"])  # ADR-0021
            return merged

        data = _call_json("architecture", payload, log, progress, validate=_validate_arch)
        data.setdefault("session_id", state["intent_model"]["session_id"])
        if persist:
            persist("architecture", data, state)
        return {"architecture": data, "gate_feedback": _clear_feedback(state, "architecture")}

    def node_items(state: GenState) -> dict:
        # Bulk proposal, code disposes (same doctrine as architecture): the
        # model chooses the module grouping - that IS design analysis, not
        # skeleton structure to derive - but every reference is constrained
        # to the REAL requirement ids (per-run grammar enum) and coverage is
        # validated HERE with a retry, not as a punch-list surprise after
        # assembly (the 2026-09-09 live failure: invented R-001..R-005 left
        # all ten real requirements uncovered).
        req_ids = [r.get("requirement_id") for sec in ("functional", "non_functional")
                   for r in (state["srs"].get(sec) or [])]
        import time as _t

        from .progress import ProgressEvent
        from .schemas import items_body_schema, items_plan_schema, items_schema

        if model_config.bulk_fills_enabled():
            data = _call_json("items", _items_payload(state, req_ids), log, progress,
                              validate=_items_validator(req_ids), schema=items_schema(req_ids))
        else:
            # ADR-0022: plan the module grouping (one small call, references
            # enum-constrained, coverage validated), then author each spec
            # body in its own micro-call - a weak body fails one cheap call.
            plan = _call_json("items", _items_plan_payload(state, req_ids), log, progress,
                              validate=_items_plan_validator(req_ids),
                              schema=items_plan_schema(req_ids), stream_items=False)
            if progress is not None:
                progress.emit(ProgressEvent("items", "", "item", _t.time(), None,
                                            f"plan: {len(plan)} specs to write"))
            done = {d["spec_id"]: d for d in ((partials.load("items") or []) if partials else [])}
            data = []
            for spec in plan:
                if spec["spec_id"] in done:  # per-item resume: preserved body
                    data.append(done[spec["spec_id"]])
                    continue
                body = _call_json("items", _items_body_payload(state, spec), log, progress,
                                  validate=_items_body_validator(spec), schema=items_body_schema(),
                                  stream_items=False,
                                  item_label=f"{spec['spec_id']} {spec.get('title', '')}".strip())
                entry = {**spec, "body_markdown": body["body_markdown"]}
                data.append(entry)
                if partials:
                    done[entry["spec_id"]] = entry
                    partials.save("items", list(done.values()))
        if persist:
            persist("items", data, state)
        if partials:
            partials.clear("items")
        return {"items": data}

    def node_tests(state: GenState) -> dict:
        """Two V pairings in one pass: acceptance tests pair with SRS criteria,
        unit/module tests pair with item specs."""
        # ADR-0020: one slot per SRS acceptance criterion (acceptance level)
        # + one per item spec (unit level); T-ids and links are code-owned.
        slots = build_test_skeleton(state["srs"], state["items"])
        if tools_enabled():
            from .tools import VerificationToolbox, run_tool_loop
            from .prompts import TOOL_PROMPTS

            toolbox = VerificationToolbox(
                slots,
                criterion_ids={c.get("criterion_id") for sec in ("functional", "non_functional")
                               for r in (state["srs"].get(sec) or []) for c in (r.get("acceptance") or [])},
                spec_ids={i.get("spec_id") for i in state["items"]})
            run_tool_loop("tests", toolbox,
                          TOOL_PROMPTS["tests"].format(skeleton=json.dumps(slots),
                                                      srs=json.dumps(_prompt_view(state["srs"])),
                                                      items=json.dumps(_prompt_view(state["items"]))),
                          log, progress)

        def _validate_tests(d):
            merged = merge_test_fills(slots, d)
            for t in merged:
                AcceptanceTest.model_validate(t)
            return merged

        from .schemas import test_fill_schema

        # Per-item resume: each fill is preserved the moment its JSON closes
        # (streamed), so a Ctrl+C at T-34/36 reruns only T-35 and T-36.
        done = (partials.load("tests_fills") or {}) if partials else {}
        missing = [sl for sl in slots if sl["test_id"] not in done]
        if missing:
            def _keep(raw: str, label: str = "") -> None:
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    return
                if isinstance(obj, dict) and len(obj) == 1:
                    tid, fill = next(iter(obj.items()))
                    if isinstance(fill, dict) and fill.get("definition") is not None:
                        done[tid] = fill
                        partials.save("tests_fills", done)

            def _validate_new(d):
                for t in merge_test_fills(missing, d):
                    AcceptanceTest.model_validate(t)
                return d

            missing_ids = [sl["test_id"] for sl in missing]
            out = _call_json("tests",
                      STAGE_PROMPTS["tests"] + "\n\nTEST SLOTS (fill prose for each, keyed by test_id):\n"
                      + json.dumps(missing) + "\n\nSRS:\n" + json.dumps(_prompt_view(state["srs"]))
                      + "\n\nITEMS:\n" + json.dumps(_prompt_view(state["items"]))
                      + "\n\nFORMAT CONTRACT (final instruction - overrides anything above that conflicts): "
                        "return one STRICT JSON object {fills: {...}} with EXACTLY these keys, one per slot: "
                      + json.dumps(missing_ids) + ". Each fill is {description, runner, definition}. runner and "
                        "definition describe how to test THE FOUNDER'S PRODUCT - never this pipeline.", log, progress,
                          validate=_validate_new, schema=test_fill_schema(missing),
                          on_item=(_keep if partials else None))
            fills = {**done, **(out.get("fills") or {})}
        else:
            fills = done
        data = _validate_tests({"fills": fills})
        if persist:
            persist("tests", data, state)
        if partials:
            partials.clear("tests_fills")
        return {"tests": data}

    def node_review(state: GenState) -> dict:
        n = state.get("review_round", 0) + 1
        fixes = _call_json("review", STAGE_PROMPTS["review"].format(n=n)
                           + "\n\nSRS:\n" + json.dumps(_prompt_view(state["srs"]))
                           + "\n\nITEMS:\n" + json.dumps(_prompt_view(state["items"]))
                           + "\n\nTESTS:\n" + json.dumps(_prompt_view(state["tests"])), log, progress)
        return {"review_round": n, "review_fixes": fixes}

    def route_review(state: GenState) -> str:
        if state.get("review_round", 0) >= REVIEW_PASSES:
            return "assemble"
        if not state.get("review_fixes"):
            return "assemble"  # clean review: stop early, save the pass
        return "apply_fixes"

    def node_apply_fixes(state: GenState) -> dict:
        # ADR-0020 shape: re-fire the FILL stage with the review fixes - the
        # skeleton slots/ids are unchanged, the model revises prose per slot.
        data = _run_srs_fill_stage(
            state,
            "REVIEW FIXES - apply EVERY one to the affected fills:\n"
            + json.dumps(state["review_fixes"]))
        data.setdefault("session_id", state["intent_model"]["session_id"])
        data.setdefault("intent_model_ref", state["intent_model"]["session_id"])
        if persist:
            persist("srs", data, state)
        return {"srs": data}

    def node_assemble(state: GenState) -> dict:
        """Deterministic tail. Validate hard; ship nothing with orphan traces."""
        sid = state["intent_model"]["session_id"]
        intent = IntentModel.model_validate(state["intent_model"])
        needs = StakeholderNeedsSet.model_validate(state["needs"])
        # structure is carried from the intent model deterministically - never
        # trust the LLM to re-copy glossary/assumptions/open questions.
        srs = SRSDocument.model_validate(srs_document_data(state["intent_model"], state["srs"]))
        # the flat view feeds the existing traceability math unchanged
        reqs = RequirementSet(session_id=sid, requirements=srs.requirements())
        arch = ArchitectureOverview.model_validate(_legacy_arch_fill(state["architecture"]))
        items = [ItemSpec.model_validate(i) for i in state["items"]]
        tests = [AcceptanceTest.model_validate(t) for t in state["tests"]]
        violations = validate_traceability(needs, reqs, items, tests) + state.get("guard_flags", [])
        trace = build_traceability(reqs, items, tests)
        milestones = order_milestones(items, tests)
        manifest = SpecManifest(
            bundle_version="0.2.0",
            files={
                "needs": "needs.md", "srs": "srs.md",
                "architecture": "architecture.md", "agents": "AGENTS.md",
                "traceability": "traceability.json", "manifest": "manifest.json",
                **{i.spec_id: f"specs/{i.spec_id}.md" for i in items},
            },
            milestones=milestones,
        )
        punch = [PunchListItem(item_id=f"PL-{n+1}", description=v, blocks=[], severity="blocking")
                 for n, v in enumerate(violations)]
        bundle = SpecBundle(
            session_id=sid, intent_model_ref=sid, items=items,
            agents_md="",  # rendered at delivery from manifest+arch (render.py)
            punch_list=punch, traceability=trace,
        )
        out = bundle.model_dump(mode="json")
        out["_manifest"] = manifest.model_dump(mode="json")
        out["_needs"] = needs.model_dump(mode="json")
        out["_srs"] = srs.model_dump(mode="json")
        out["_requirements"] = reqs.model_dump(mode="json")
        out["_architecture"] = arch.model_dump(mode="json")
        out["_tests"] = [t.model_dump(mode="json") for t in tests]
        return {"bundle": out, "violations": violations}

    g = StateGraph(GenState)
    g.add_node("needs", _wrap("needs", "Formalizing stakeholder needs",
            lambda st: f"stage 1/6, from {_sn_count(st)} interview needs (+ any implied)")(node_needs))
    g.add_node("srs", _wrap("srs", "Authoring the SRS",
            lambda st: f"stage 2/6, from {_n_count(st)} stakeholder needs")(node_srs))
    g.add_node("architecture", _wrap("architecture", "Drafting architecture overview",
            lambda st: f"stage 3/6, from {_req_count(st)} requirements")(node_architecture))
    g.add_node("items", _wrap("items", "Writing module specs",
            lambda st: f"stage 4/6, covering {_req_count(st)} requirements")(node_items))
    g.add_node("tests", _wrap("tests", "Deriving test specifications",
            lambda st: f"stage 5/6, from {_ac_count(st)} acceptance criteria + {len(st['items'])} module specs")(node_tests))
    g.add_node("review", _wrap("review", "Cross-checking consistency",
            lambda st: f"stage 6/6, over {_req_count(st)} requirements, {len(st['items'])} specs, {len(st['tests'])} tests")(node_review))
    g.add_node("apply_fixes", _wrap("apply_fixes", "Applying review fixes")(node_apply_fixes))
    g.add_node("assemble", _wrap("assemble", "Assembling + validating bundle", "deterministic")(node_assemble))
    g.set_entry_point("needs")
    g.add_edge("needs", "srs")
    if review:
        from langgraph.types import interrupt

        def _make_gate(stage_key: str, artifact_key: str):
            def node_gate(state: GenState) -> dict:
                verdict = interrupt({"gate": stage_key, "artifact": state.get(artifact_key)})
                if verdict.get("action") == "regenerate":
                    return {"gate_feedback": {stage_key: verdict["feedback"]}}
                return {}  # accept (edits landed via update_state before resume)
            return node_gate

        # NOT wrapped in track(): interrupt() raises through the progress
        # contextmanager and would print a spurious FAILED line per pause.
        g.add_node("gate_srs", _make_gate("srs", "srs"))
        g.add_node("gate_architecture", _make_gate("architecture", "architecture"))
        g.add_edge("srs", "gate_srs")
        g.add_conditional_edges("gate_srs",
                                lambda st: "srs" if _gate_feedback(st, "srs") else "architecture",
                                {"srs": "srs", "architecture": "architecture"})
        g.add_edge("architecture", "gate_architecture")
        g.add_conditional_edges("gate_architecture",
                                lambda st: "architecture" if _gate_feedback(st, "architecture") else "items",
                                {"architecture": "architecture", "items": "items"})
    else:
        g.add_edge("srs", "architecture")
        g.add_edge("architecture", "items")
    g.add_edge("items", "tests")
    g.add_edge("tests", "review")
    g.add_conditional_edges("review", route_review, {"apply_fixes": "apply_fixes", "assemble": "assemble"})
    g.add_edge("apply_fixes", "items")
    g.add_edge("assemble", END)
    return g.compile(checkpointer=checkpointer or MemorySaver(serde=checkpoint_serde()))

