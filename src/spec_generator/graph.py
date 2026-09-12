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
    SystemBaseline,
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
    system_baseline: dict
    needs: dict
    srs: dict
    requirements: dict
    guard_flags: list[str]
    architecture: dict
    items: list[dict]
    tests: list[dict]
    review_round: int
    review_fixes: list[dict]
    bundle: dict
    violations: list[str]
    gate_feedback: dict[str, str]  # stage -> founder feedback from a review gate (ADR-0019)


def _invoke(stage: str, payload: str, log: UsageLog, progress: ProgressBus | None = None,
            feedback: str | None = None, schema: dict | None = None,
            stream_items: bool = True, item_label: str | None = None,
            on_item=None, attempt: int = 1, retry_reason: str | None = None) -> str:
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
    for chunk in model_config.stream_checked(model, messages, stage, item=item_label,
                                             attempt=attempt, retry_reason=retry_reason):
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
    raw = "".join(chunks)
    model_config.log_raw_completion(stage, item_label, attempt, raw)
    return raw


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
                      stream_items=stream_items, item_label=item_label, on_item=on_item,
                      attempt=attempt, retry_reason=("contract" if fb else None))
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
                    f"{model_config._rerun_guidance(stage)}; if it repeats, route the stage "
                    f"to a stronger model (SPEC_GENERATOR_MODEL_{stage.upper()} env override) - and report it: "
                    "schema-constrained output should not drift structurally.",
                ) from contract_err
            raise UserFacingError(
                f"stage '{stage}': the model returned output that is not valid JSON (an automatic retry did not fix it).",
                f"smaller/local models often wrap JSON in prose, drop structure, or truncate. Raw output began: {snippet!r}",
                f"{model_config._rerun_guidance(stage)}; if it repeats, route the stage "
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
    requires a real driver before state is written.

    Pre-0.13.0 checkpoints hold the parallel maps (stack + stack_baseline_refs);
    heal them into typed stack_entries. A MISSING refs map heals to empty refs
    ("not recorded" - pre-grounding artifacts). A MISMATCHED refs map is the
    2026-09-11 phi4 failure shape: its per-key refs cannot be trusted, so every
    entry heals with refs [] - empty refs mark a migration choice and can never
    fabricate baseline grounding; the reconciliation gate judges the result.
    (The explicit `vmodel_contracts.migrate` path refuses this shape instead -
    there, "re-run the stage" is an actionable answer.)"""
    out = dict(payload)
    if "stack_entries" not in out and "stack" in out:
        stack = out.pop("stack") or {}
        refs = out.pop("stack_baseline_refs", None)
        if isinstance(refs, dict) and set(refs) == set(stack):
            healed = [(k, v, refs[k] if isinstance(refs[k], list) else []) for k, v in stack.items()]
        else:
            healed = [(k, v, []) for k, v in stack.items()]
        out["stack_entries"] = [
            {"key": k, "value": v, "baseline_refs": [str(r) for r in rs], "migration_adr_id": None}
            for k, v, rs in healed]
    else:
        out.pop("stack", None)
        out.pop("stack_baseline_refs", None)
    decisions = []
    for d in out.get("decisions") or []:
        dd = dict(d)
        dd.setdefault("driver", None)
        dd.setdefault("options_considered", [])
        decisions.append(dd)
    out["decisions"] = decisions
    return out


def baseline_reference_index(baseline: SystemBaseline) -> set[str]:
    """Canonical, mechanically checkable references available to the model."""
    refs = {f"repo:{r.name}" for r in baseline.repositories}
    refs.update(f"decision:system:{i}" for i, _ in enumerate(baseline.immutable_decisions, 1))
    fields = (("entry", "package_entry_points"), ("module", "module_entry_points"),
              ("artifact", "existing_artifacts"), ("contract", "existing_contracts"),
              ("runtime", "runtime"), ("persistence", "persistence"),
              ("integration", "integrations"), ("decision", "immutable_decisions"))
    for r in baseline.repositories:
        for prefix, field in fields:
            refs.update(f"{prefix}:{r.name}:{i}" for i, _ in enumerate(getattr(r, field), 1))
    return refs


def _check_baseline_reconciliation(arch: ArchitectureOverview, baseline: SystemBaseline) -> None:
    if arch.baseline_id != baseline.baseline_id:
        raise ModelStructureError(f"architecture baseline_id must be {baseline.baseline_id!r}, got {arch.baseline_id!r}")
    allowed = baseline_reference_index(baseline)
    repo_names = {r.name for r in baseline.repositories}
    def check_refs(label: str, refs: list[str]) -> None:
        unknown = sorted(set(refs) - allowed)
        if unknown:
            raise ModelStructureError(f"{label}: unknown baseline refs: {', '.join(unknown)}")
    migration_list = [d for d in arch.decisions if d.migration]
    migration_adrs = {d.title: d for d in migration_list}
    if len(migration_adrs) != len(migration_list):
        raise ModelStructureError("migration ADR titles must be unique")
    migration_adr_ids = {d.decision_id for d in migration_list}
    entry_keys = [e.key for e in arch.stack_entries]
    if len(entry_keys) != len(set(entry_keys)):
        raise ModelStructureError("stack entry keys must be unique")
    if baseline.mode.value == "existing":
        baseline_vocab = " ".join(
            v for r in baseline.repositories for field in (
                "languages", "runtime", "persistence", "integrations", "immutable_decisions")
            for v in getattr(r, field)).lower()
        for e in arch.stack_entries:
            check_refs(f"stack.{e.key}", e.baseline_refs)
            if not e.baseline_refs:
                # a departure: authorized by pipeline-wired id (0.13.0) or, for
                # healed pre-0.13.0 artifacts, by the legacy title/decision text.
                if e.migration_adr_id is not None:
                    if e.migration_adr_id not in migration_adr_ids:
                        raise ModelStructureError(
                            f"stack.{e.key}: migration_adr_id {e.migration_adr_id} is not an ADR tagged migration")
                else:
                    matching = [d for d in migration_list
                                if e.key.lower() in (d.title + " " + d.decision).lower()
                                or str(e.value).lower() in (d.title + " " + d.decision).lower()]
                    if not matching:
                        raise ModelStructureError(f"stack.{e.key}: departure requires an ADR tagged migration that names the stack key or value")
            else:
                if e.migration_adr_id is not None:
                    raise ModelStructureError(
                        f"stack.{e.key}: migration_adr_id requires empty baseline_refs (a departure)")
                if str(e.value).lower() not in baseline_vocab:
                    raise ModelStructureError(f"stack.{e.key}: value {e.value!r} is not present in the cited baseline facts")
        for c in arch.components:
            check_refs(f"component {c.name}", c.baseline_refs)
            if c.migration:
                if c.migration_adr_title not in migration_adrs:
                    raise ModelStructureError(f"component {c.name}: migration must cite an ADR tagged migration")
            else:
                if c.migration_adr_title is not None:
                    raise ModelStructureError(f"component {c.name}: migration_adr_title requires migration=true")
                if not c.baseline_refs:
                    raise ModelStructureError(f"component {c.name}: existing component requires baseline refs")
    else:
        if (any(e.baseline_refs or e.migration_adr_id for e in arch.stack_entries)
                or any(c.baseline_refs for c in arch.components)):
            raise ModelStructureError("greenfield architecture cannot cite nonexistent baseline refs or migration ADRs")
        if any(c.migration or c.migration_adr_title for c in arch.components) or migration_adrs:
            raise ModelStructureError("greenfield architecture is new design, not a migration")
    for d in arch.decisions:
        check_refs(f"{d.decision_id} {d.title}", d.baseline_refs)
        if d.migration:
            if not d.evidence:
                raise ModelStructureError(f"{d.decision_id} {d.title}: migration requires evidence")
            if not d.affected_repositories:
                raise ModelStructureError(f"{d.decision_id} {d.title}: migration requires affected_repositories")
            unknown_repos = sorted(set(d.affected_repositories) - repo_names)
            if unknown_repos:
                raise ModelStructureError(f"{d.decision_id} {d.title}: unknown affected repositories: {', '.join(unknown_repos)}")
        elif d.evidence or d.affected_repositories:
            raise ModelStructureError(f"{d.decision_id} {d.title}: evidence/affected_repositories only belong on migration ADRs")


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


def _validate_stack_plan(d: dict) -> dict:
    """Plan call A result: a non-empty list of unique, non-empty slot keys."""
    keys = d.get("stack_keys") if isinstance(d, dict) else None
    if not isinstance(keys, list) or not keys or not all(isinstance(k, str) and k.strip() for k in keys):
        raise ModelStructureError("stack plan must be a non-empty list of non-empty string keys")
    keys = [k.strip() for k in keys]
    if len(set(keys)) != len(keys):
        raise ModelStructureError(f"duplicate stack keys: {sorted({k for k in keys if keys.count(k) > 1})}")
    if len(keys) > 24:
        raise ModelStructureError(f"stack plan has {len(keys)} slots - merge near-duplicate slots (24 max)")
    return {"stack_keys": keys}


def _validate_component_plan(d, req_ids: list[str]) -> list[dict]:
    """Plan call B result: component slots. The grammar enum-constrains
    requirement refs; here: unique names, real coverage, sane shape."""
    if not isinstance(d, list) or not d:
        raise ModelStructureError("component plan must be a non-empty array of slots")
    slots = []
    names = []
    for s in d:
        if not isinstance(s, dict) or not str(s.get("name", "")).strip():
            raise ModelStructureError("every component slot needs a non-empty name")
        name = str(s["name"]).strip()
        names.append(name)
        slots.append({"name": name,
                      "requirement_ids": [str(r) for r in s.get("requirement_ids") or []],
                      "baseline_refs": [str(r) for r in s.get("baseline_refs") or []],
                      "migration": bool(s.get("migration"))})
    if len(names) != len(set(names)):
        raise ModelStructureError(f"duplicate component names: {sorted({n for n in names if names.count(n) > 1})}")
    covered = {r for s in slots for r in s["requirement_ids"]}
    missing = [r for r in req_ids if r not in covered]
    if missing:
        raise ModelStructureError(f"requirements covered by no component: {missing}")
    return slots


def _required_adr_slots(authorizable: list[str]) -> list[dict]:
    """Required migration ADR slots are pipeline-GENERATED, one per declared
    departure (the 2026-09-12 live phi4 failure: the model planned a
    ci_tool-absence slot with migration:false and left stack:ci_tool
    unauthorized - a mechanical obligation the model forgot). Stable
    human-readable slugs derive from the departure target; the model only
    adds optional design ADRs and authors the prose."""
    import re as _re

    slots, used = [], set()
    for target in authorizable:
        base = "migrate-" + _re.sub(r"[^a-z0-9]+", "-", target.lower()).strip("-")
        slug, n = base, 2
        while slug in used:
            slug, n = f"{base}-{n}", n + 1
        used.add(slug)
        slots.append({"slug": slug, "migration": True, "authorizes": [target]})
    return slots


def _validate_adr_plan(d, taken_slugs: list[str]) -> list[dict]:
    """Plan call C result: OPTIONAL design-decision slots. Migration ADRs are
    pipeline-generated (_required_adr_slots), so the model can no longer miss
    or mis-set one. Here: non-empty unique slugs that do not collide with the
    pipeline slots, and no model-authored migration/authorizes keys - those
    are pipeline-owned."""
    if not isinstance(d, list):
        raise ModelStructureError("ADR plan must be an array of decision slots")
    slots = []
    slugs = []
    for s in d:
        if not isinstance(s, dict) or not str(s.get("slug", "")).strip():
            raise ModelStructureError("every ADR slot needs a non-empty slug")
        slug = str(s["slug"]).strip()
        if "migration" in s or "authorizes" in s:
            raise ModelStructureError(
                f"ADR slot {slug!r}: migration/authorizes are pipeline-owned - "
                "propose only optional design decisions, by slug alone")
        slugs.append(slug)
        slots.append({"slug": slug, "migration": False, "authorizes": []})
    if len(slugs) != len(set(slugs)):
        raise ModelStructureError(f"duplicate ADR slugs: {sorted({n for n in slugs if slugs.count(n) > 1})}")
    clash = sorted(set(taken_slugs) & set(slugs))
    if clash:
        raise ModelStructureError(
            f"ADR slugs collide with pipeline-required migration slots: {clash}")
    return slots


def _merge_adr_slots(required: list[dict], optional: list[dict]) -> list[dict]:
    """Required migration slots first, then the model's optional slots; code
    assigns ADR ids in order (ADR-0020)."""
    slots = [dict(s) for s in required] + [dict(s) for s in optional]
    for i, s in enumerate(slots, start=1):
        s["decision_id"] = f"ADR-{i}"
    return slots


def _assemble_architecture(*, session_id: str, baseline_id: str, entries: dict,
                           comp_slots: list, comp_bodies: dict, adr_slots: list,
                           adr_bodies: dict, nfr: dict) -> dict:
    """Deterministic assembly (0.13.0): the model authored per-slot substance;
    code wires every cross-reference - ADR ids, departure -> authorizing ADR,
    component -> migration ADR title - from the validated slots."""
    decisions = [{"decision_id": slot["decision_id"], **adr_bodies[slot["decision_id"]]}
                 for slot in adr_slots]

    def covering(target: str) -> str | None:
        for slot in adr_slots:
            if slot["migration"] and target in slot["authorizes"]:
                return slot["decision_id"]
        return None

    stack_entries = []
    for key, e in entries.items():
        adr_id = None if e["baseline_refs"] else covering(f"stack:{key}")
        stack_entries.append({"key": key, "value": e["value"],
                              "baseline_refs": e["baseline_refs"], "migration_adr_id": adr_id})
    components = []
    for c in comp_slots:
        adr_id = covering(f"component:{c['name']}") if c.get("migration") else None
        components.append({"name": c["name"],
                           "responsibility": comp_bodies[c["name"]]["responsibility"],
                           "requirement_ids": c["requirement_ids"],
                           "baseline_refs": c["baseline_refs"],
                           "migration": bool(c.get("migration")),
                           "migration_adr_title": (adr_bodies[adr_id]["title"] if adr_id else None)})
    return {"session_id": session_id, "baseline_id": baseline_id,
            "stack_entries": stack_entries, "components": components,
            "decisions": decisions, "non_functionals": nfr}



def _heal_slugs(entries: list) -> list:
    """ADR-0022: punctuation drift in a model-proposed slug is not a semantic
    failure - normalize before validation instead of burning a retry:
    lowercase, whitespace/underscores/invalid runs -> single hyphens
    ('user_capabilities' -> 'user-capabilities'). Every heal is printed;
    depends_on references are remapped with the same table. Heals that would
    come out empty are left untouched so validation still fails loudly on
    REAL drift; collisions get deterministic -2/-3 suffixes in plan order."""
    import re as _re

    def _heal(s: str) -> str:
        h = _re.sub(r"[^a-z0-9]+", "-", s.strip().lower()).strip("-")
        return _re.sub(r"-{2,}", "-", h)

    mapping: dict[str, str] = {}
    used: set[str] = set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        raw = e.get("slug")
        if not isinstance(raw, str) or not raw.strip():
            continue
        h = _heal(raw)
        if h and _re.fullmatch(r"[a-z0-9][a-z0-9-]*", h):
            base, n = h, 2
            while h in used:
                h, n = f"{base}-{n}", n + 1
            used.add(h)
            mapping[raw] = h
    out = []
    for e in entries:
        if not isinstance(e, dict):
            out.append(e)
            continue
        e = dict(e)
        raw = e.get("slug")
        healed = mapping.get(raw) if isinstance(raw, str) else None
        if healed and healed != raw:
            e["slug"] = healed
            model_config._console_line(f"     - slug healed: {raw!r} -> {healed!r}")
        deps = e.get("depends_on")
        if isinstance(deps, list):
            e["depends_on"] = [mapping.get(str(d), d) for d in deps]
        out.append(e)
    return out


def _check_item_refs(entries: list, req_ids: list[str], key: str = "spec_id") -> None:
    """Shared reference/coverage checks for the items stage (bulk specs and
    ADR-0022 plans): unique ids, references resolve, deps point at emitted
    specs, every requirement covered. Failures are NAMED so the automatic
    retry's feedback says exactly what to fix. key="slug" checks the
    model-proposed slugs BEFORE code numbers the specs (see _assign_spec_ids)."""
    spec_ids = [e.get(key) for e in entries]
    dupes = sorted({s for s in spec_ids if spec_ids.count(s) > 1})
    if dupes:
        raise ModelStructureError(f"duplicate spec ids: {dupes} - every spec needs a unique spec_id")
    known = set(req_ids)
    emitted = set(spec_ids)
    for e in entries:
        unknown = sorted(set(e.get("requirement_ids") or []) - known)
        if unknown:
            raise ModelStructureError(
                f"{e.get(key)} claims unknown requirements {unknown} - valid ids: {req_ids}")
        bad_deps = sorted(set(e.get("depends_on") or []) - emitted)
        if bad_deps:
            raise ModelStructureError(
                f"{e.get(key)} depends on unknown specs {bad_deps} - depends_on may only "
                f"reference {key}s from this same answer")
    covered = {rid for e in entries for rid in (e.get("requirement_ids") or [])}
    uncovered = [r for r in req_ids if r not in covered]
    if uncovered:
        raise ModelStructureError(
            f"requirements covered by no item spec: {uncovered} - cover every requirement: "
            f"add a spec or extend an existing spec's requirement_ids")


def _assign_spec_ids(entries: list) -> list:
    """Code owns ids (ADR-0022 doctrine): the model proposes short slugs, the
    pipeline numbers the specs SPEC-1..N in plan order and translates
    depends_on slugs to ids. Numbered ids keep every output line consistent
    (R-1, SPEC-1, T-1); the slug stays on the spec as the human-readable name."""
    slug_to_id = {e["slug"]: f"SPEC-{i}" for i, e in enumerate(entries, 1)}
    return [{**e, "spec_id": slug_to_id[e["slug"]],
             "depends_on": [slug_to_id[d] for d in e.get("depends_on") or []]}
            for e in entries]


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
    payload = (STAGE_PROMPTS["srs_slot"]
               + "\n\nINTENT (shared context):\n" + json.dumps(_prompt_view(slim))
               + "\n\nTHE SLOT (fill exactly this one):\n" + json.dumps(slot)
               + "\n\nTHE NEED BEHIND IT:\n" + json.dumps(_prompt_view(need)))
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
              "return one STRICT JSON array of spec plans {slug, kind, title, requirement_ids, "
              "depends_on} - NO body_markdown, bodies are written one spec per call next. "
              "requirement_ids may contain ONLY these requirement ids: "
            + json.dumps(req_ids) + " - and every one of them must appear in at least one plan's "
              "requirement_ids. slug values are short lowercase names (e.g. api, orders); depends_on "
              "references slugs from this same answer. The pipeline numbers the specs SPEC-1, SPEC-2, "
              "... in your order - never invent SPEC-ids. slugs and titles NEVER contain session ids, "
              "timestamps, or run identifiers. The plan is for THE FOUNDER'S PRODUCT - never this pipeline.")


def _items_plan_validator(req_ids: list[str]):
    import re as _re

    from vmodel_contracts import SpecItemKind

    def _v(d):
        if not isinstance(d, list) or not d:
            raise ModelStructureError("return a non-empty JSON array of spec plans")
        d = _heal_slugs(d)
        kinds = {k.value for k in SpecItemKind}
        for p in d:
            slug = p.get("slug") if isinstance(p, dict) else None
            if not slug or not _re.fullmatch(r"[a-z0-9][a-z0-9-]*", str(slug)):
                raise ModelStructureError(
                    f"invalid slug {slug!r} - a short lowercase name like api or orders "
                    "(never a session id or timestamp)")
            if p.get("kind") not in kinds:
                raise ModelStructureError(f"{slug}: kind must be one of {sorted(kinds)}")
            if not str(p.get("title") or "").strip():
                raise ModelStructureError(f"{slug}: title required")
        _check_item_refs(d, req_ids, key="slug")
        return _assign_spec_ids(d)
    return _v


def _items_body_payload(state: GenState, spec: dict) -> str:
    reqs = [r for sec in ("functional", "non_functional") for r in (state["srs"].get(sec) or [])
            if r.get("requirement_id") in (spec.get("requirement_ids") or [])]
    # Shared prefix first (prompt + architecture are identical across spec
    # calls - prefix-cache friendly), the varying spec + requirements last.
    return (STAGE_PROMPTS["items_body"]
            + "\n\nARCHITECTURE:\n" + json.dumps(_prompt_view(state["architecture"]))
            + "\n\nTHE SPEC (write its body):\n" + json.dumps(spec)
            + "\n\nTHE REQUIREMENTS IT COVERS:\n" + json.dumps(_prompt_view(reqs))
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
              "return one STRICT JSON array of spec objects {slug, kind, title, body_markdown, "
              "requirement_ids, depends_on}. requirement_ids may contain ONLY these requirement ids: "
            + json.dumps(req_ids) + " - and every one of them must appear in at least one spec's "
              "requirement_ids. slug values are short lowercase names (e.g. api, orders); depends_on "
              "references slugs from this same answer. The pipeline numbers the specs SPEC-1, SPEC-2, "
              "... in your order - never invent SPEC-ids. slugs and titles NEVER contain session ids, "
              "timestamps, or run identifiers. Specs are written for a coding agent building THE "
              "FOUNDER'S PRODUCT - never this pipeline.")


def _items_validator(req_ids: list[str]):
    """Stage-level items contract: each spec passes ItemSpec; spec ids are
    unique; every reference resolves (the grammar's enum already makes unknown
    requirement ids impossible when structured output is on - this catches the
    free-mode fallback); dependencies point at emitted specs; and EVERY
    requirement is covered by at least one spec. Failures are named, so the
    automatic retry's feedback says exactly what to fix."""
    def _validate(d):
        import re as _re

        if not isinstance(d, list) or not d:
            raise ModelStructureError("return a non-empty JSON array of spec objects")
        d = _heal_slugs(d)
        for i in d:
            slug = i.get("slug") if isinstance(i, dict) else None
            if not slug or not _re.fullmatch(r"[a-z0-9][a-z0-9-]*", str(slug)):
                raise ModelStructureError(
                    f"invalid slug {slug!r} - a short lowercase name like api or orders "
                    "(never a session id or timestamp)")
        _check_item_refs(d, req_ids, key="slug")
        items = [ItemSpec.model_validate(_strip_pipeline_owned(i)) for i in _assign_spec_ids(d)]
        return [i.model_dump() for i in items]
    return _validate


def _gate_feedback(state: GenState, stage: str) -> str | None:
    return (state.get("gate_feedback") or {}).get(stage)


def _clear_feedback(state: GenState, stage: str) -> dict:
    return {k: v for k, v in (state.get("gate_feedback") or {}).items() if k != stage}


def _normalized_review_fixes(value) -> list[dict]:
    out = []
    for fix in value or []:
        if isinstance(fix, str):
            out.append({"scope": "systemic", "target_ids": [], "fix": fix})
        elif isinstance(fix, dict) and str(fix.get("fix") or "").strip():
            scope = fix.get("scope") if fix.get("scope") in ("requirement", "spec", "test", "systemic") else "systemic"
            ids = [str(x) for x in (fix.get("target_ids") or []) if str(x)]
            if scope != "systemic" and not ids:
                scope = "systemic"
            out.append({"scope": scope, "target_ids": ids, "fix": str(fix["fix"])})
    return out


def _fix_targets(fixes: list[dict], scope: str) -> set[str] | None:
    fixes = _normalized_review_fixes(fixes)
    if any(f["scope"] == "systemic" for f in fixes):
        return None
    return {x for f in fixes if f["scope"] == scope for x in f["target_ids"]}


def _affected_spec_ids(state: GenState, fixes: list[dict]) -> set[str] | None:
    req_targets = _fix_targets(fixes, "requirement")
    spec_targets = _fix_targets(fixes, "spec")
    if req_targets is None or spec_targets is None:
        return None
    return set(spec_targets) | {spec["spec_id"] for spec in state.get("items") or []
                                if set(spec.get("requirement_ids") or []) & set(req_targets)}


def _compact_test_context(state: GenState, slot: dict) -> dict:
    """Only the linked source plus small architecture/ID context, never full artifacts."""
    linked = {}
    if slot.get("criterion_id"):
        for sec in ("functional", "non_functional"):
            for req in state["srs"].get(sec) or []:
                for criterion in req.get("acceptance") or []:
                    if criterion.get("criterion_id") == slot["criterion_id"]:
                        linked = {"requirement_id": req.get("requirement_id"), "criterion": criterion}
    elif slot.get("spec_id"):
        spec = next((i for i in state["items"] if i.get("spec_id") == slot["spec_id"]), {})
        linked = {"spec": spec}
    arch = state.get("architecture") or {}
    minimal_arch = {k: arch.get(k) for k in ("stack_entries", "components", "non_functionals") if arch.get(k)}
    glossary = {"requirements": [r.get("requirement_id") for sec in ("functional", "non_functional")
                                  for r in state["srs"].get(sec) or []],
                "specs": [i.get("spec_id") for i in state["items"]]}
    return {"linked_source": linked, "architecture": minimal_arch, "id_glossary": glossary}


def build_graph(log: UsageLog, checkpointer=None, progress: ProgressBus | None = None,
                review: bool = False, persist=None, partials=None,
                only_stage: str | None = None):
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

    def _run_srs_fill_stage(state: GenState, feedback: str | None = None,
                            only: set[str] | None = None) -> dict:
        """The ADR-0020 SRS fill stage, two harness shapes (ADR-0022):
        - default: one micro-call for purpose/scope + one per requirement slot
          (a bad fill fails one cheap call, never a long composite payload);
        - SPEC_GENERATOR_BULK=1: the original single composite call.
        Both end at the same merged, contract-validated body."""
        import time as _t

        from .progress import ProgressEvent
        from .schemas import srs_overview_schema, srs_slot_fill_schema

        slots = build_srs_skeleton(state["needs"])
        if model_config.bulk_fills_enabled() and only is None:
            extra = ""
            if feedback:
                extra = "\n\n" + feedback
            if state.get("srs"):
                extra += "\n\nPREVIOUS SRS (revise it; slots and ids are unchanged):\n" + \
                    json.dumps(_prompt_view(state["srs"]))
            payload, _validate, schema = _srs_fill_call(state, extra)
            return _call_json("srs", payload, log, progress, validate=_validate, schema=schema)
        previous = state.get("srs") or {}
        overview = ({"purpose": previous.get("purpose"), "scope_summary": previous.get("scope_summary")}
                    if only is not None else _call_json(
                        "srs", _srs_overview_payload(state, feedback), log, progress,
                        validate=_overview_validator(), schema=srs_overview_schema(), stream_items=False))
        done = {r["requirement_id"]: {k: r.get(k) for k in ("title", "statement", "priority", "acceptance")}
                for sec in ("functional", "non_functional") for r in previous.get(sec) or []}
        if partials:
            done.update(partials.load("srs_fills") or {})
        need_hint = {n.get("need_id"): (n.get("statement") or "")[:50]
                     for n in state["needs"].get("needs") or []}
        fills = {}
        for slot in slots:
            rid = slot["requirement_id"]
            if only is not None and rid not in only:
                fills[rid] = done[rid]
                continue
            if only is None and rid in done:  # per-item resume: preserved from the crashed run
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
        """0.13.0 micro-pipeline (ADR-0022 doctrine applied to the 2026-09-11
        phi4 failure): three tiny planning calls propose SLOTS (stack keys,
        component slots, ADR slots), one shallow call fills each slot, and
        code owns ids, grounding checks, cross-references, and assembly. The
        old single giant call asked a small local model to keep two free-form
        maps key-synchronized - a relation no grammar can enforce. No schema
        here carries two fields that must agree: there is nothing left to
        drift out of sync. Per-slot partials resume a crashed stage item by
        item, the same way the SRS fills already do."""
        from .prompts import (ARCH_ADR_BODY, ARCH_ADR_PLAN, ARCH_COMPONENT_BODY,
                              ARCH_COMPONENT_PLAN, ARCH_CONTEXT, ARCH_NFR,
                              ARCH_STACK_ENTRY, ARCH_STACK_PLAN)
        from .schemas import (arch_adr_body_schema, arch_adr_plan_schema,
                              arch_component_body_schema, arch_component_plan_schema,
                              arch_nfr_schema, arch_stack_entry_schema,
                              arch_stack_plan_schema)

        if "system_baseline" not in state:
            # a KNOWN pain (old checkpoint, flag never merged) - human-readable,
            # never a raw KeyError traceback (the founder's standing rule)
            from .errors import UserFacingError
            raise UserFacingError(
                "stage 'architecture': this run has no system baseline in its state.",
                "the checkpoint predates the system-baseline feature and no --system-baseline was merged in.",
                "rerun the same command with --system-baseline <baseline.json> - the CLI merges it into the resumed state.")
        baseline = SystemBaseline.model_validate(state["system_baseline"])
        sid = state["intent_model"]["session_id"]
        valid_refs = sorted(baseline_reference_index(baseline))
        repo_names = sorted(r.name for r in baseline.repositories)
        existing = baseline.mode.value == "existing"
        req_ids = [r.get("requirement_id") for sec in ("functional", "non_functional")
                   for r in (state["srs"].get(sec) or [])]
        nfr_slots = {r["requirement_id"]: r for r in (state["srs"].get("non_functional") or [])}

        context = (ARCH_CONTEXT
                   + "\n\nSYSTEM BASELINE (source of truth):\n" + json.dumps(_prompt_view(state["system_baseline"]))
                   + "\n\nVALID BASELINE REFS (use exact strings):\n" + json.dumps(valid_refs)
                   + "\n\nSRS:\n" + json.dumps(_prompt_view(state["srs"]))
                   + _tech_constraint_block(state["intent_model"]))
        fb = _gate_feedback(state, "architecture")
        feedback = ""
        if fb:
            feedback = ("\n\nFOUNDER FEEDBACK ON YOUR PREVIOUS DRAFT (apply it):\n" + fb
                        + "\n\nPREVIOUS DRAFT:\n" + json.dumps(_prompt_view(state["architecture"])))

        part = (partials.load("architecture") if partials else None) or {}

        def _save():
            if partials:
                partials.save("architecture", part)

        # 1. Plan: which technology slots to decide.
        if "stack_keys" not in part:
            plan = _call_json("architecture", context + "\n\n" + ARCH_STACK_PLAN + feedback,
                              log, progress, validate=_validate_stack_plan,
                              schema=arch_stack_plan_schema(), stream_items=False,
                              item_label="plan: stack slots")
            part["stack_keys"] = plan["stack_keys"]
            _save()

        # 2. One micro-call per stack entry: value + grounding refs. The
        # value-vs-cited-facts check runs HERE (per-slot, retryable), not only
        # at assembly: an invented technology costs one retried entry call,
        # not the whole stage (the 2026-09-12 Auth0 acceptance failure -
        # 1282s of good calls lost to a gate that fired at the end).
        baseline_vocab = (" ".join(
            v for r in baseline.repositories for field in (
                "languages", "runtime", "persistence", "integrations", "immutable_decisions")
            for v in getattr(r, field)).lower()) if existing else ""

        def _v_entry(d, key):
            value = str(d.get("value", "")).strip()
            refs = [str(r) for r in d.get("baseline_refs") or []]
            if not value:
                # targeted feedback, not a generic contract failure (the
                # 2026-09-12 sso_integration blank: the model saw no legal
                # move when the baseline held no fact for the key)
                if refs:
                    raise ModelStructureError(
                        f"stack.{key}: value must name a concrete technology "
                        "(you cited baseline facts but named no value)")
                raise ModelStructureError(
                    f"stack.{key}: an empty value decides nothing - if no baseline fact fits "
                    "this key, name a concrete technology anyway with baseline_refs []; it "
                    "becomes a proposed migration automatically and the pipeline generates "
                    "the authorizing ADR itself")
            entry = {"key": key, "value": value, "baseline_refs": refs}
            if existing and entry["baseline_refs"] and entry["value"].lower() not in baseline_vocab:
                raise ModelStructureError(
                    f"stack.{key}: value {entry['value']!r} is not present in the cited baseline facts - "
                    "pick a value from the cited facts, or declare a migration (empty baseline_refs)")
            if not existing and entry["baseline_refs"]:
                raise ModelStructureError(
                    f"stack.{key}: greenfield architecture cannot cite baseline refs")
            return entry

        entries = part.setdefault("stack_entries", {})
        for key in list(entries):
            # persisted slots must satisfy the CURRENT contract; a stale entry
            # (e.g. Auth0 saved before this check existed) is dropped and
            # re-asked - never silently trusted, never a rerun from zero.
            try:
                _v_entry(entries[key], key)
            except ModelStructureError as e:
                del entries[key]
                model_config._console_line(
                    f"     - persisted stack entry {key!r} failed the current contract ({e}) - re-asking that slot")
        for key in part["stack_keys"]:
            if key in entries:
                continue  # per-slot resume
            entries[key] = _call_json(
                "architecture",
                context + f"\n\nTHE STACK KEY TO DECIDE: {key}\n\n" + ARCH_STACK_ENTRY + feedback,
                log, progress, validate=lambda d, key=key: _v_entry(d, key),
                schema=arch_stack_entry_schema(valid_refs),
                stream_items=False, item_label=f"stack:{key}")
            _save()

        # 3. Plan: component slots (coverage checked here, not at assembly).
        if "component_slots" not in part:
            part["component_slots"] = _call_json(
                "architecture",
                context + "\n\nREQUIREMENT IDS TO COVER: " + json.dumps(req_ids)
                + "\n\n" + ARCH_COMPONENT_PLAN + feedback,
                log, progress, validate=lambda d: _validate_component_plan(d, req_ids),
                schema=arch_component_plan_schema(req_ids, valid_refs),
                stream_items=False, item_label="plan: component slots")
            _save()

        # 4. One micro-call per component: responsibility prose.
        comp_bodies = part.setdefault("component_bodies", {})
        for slot in part["component_slots"]:
            if slot["name"] in comp_bodies:
                continue
            def _v_comp(d, name=slot["name"]):
                if not str(d.get("responsibility", "")).strip():
                    raise ModelStructureError(f"component {name}: responsibility must not be empty")
                return {"responsibility": str(d["responsibility"]).strip()}
            comp_bodies[slot["name"]] = _call_json(
                "architecture",
                context + "\n\nTHE COMPONENT SLOT (fixed):\n" + json.dumps(slot)
                + "\n\n" + ARCH_COMPONENT_BODY + feedback,
                log, progress, validate=_v_comp, schema=arch_component_body_schema(),
                stream_items=False, item_label=f"component:{slot['name']}")
            _save()

        # 5. Plan: ADR slots. Departures are CODE-COMPUTED from the accepted
        # entries/slots and the required migration slots are CODE-GENERATED
        # from them (the 2026-09-12 live phi4 failure: the model forgot to
        # authorize stack:ci_tool). The model only proposes OPTIONAL design
        # decisions; ids are assigned after the merge, required slots first.
        departures = [k for k, e in entries.items() if not e["baseline_refs"]] if existing else []
        migrating = [c["name"] for c in part["component_slots"] if c["migration"]] if existing else []
        authorizable = [f"stack:{k}" for k in departures] + [f"component:{n}" for n in migrating]
        if "adr_slots" not in part:
            required = _required_adr_slots(authorizable)
            req_block = ("\n\nPIPELINE-REQUIRED MIGRATION ADRS (already planned - do not "
                         "re-propose them; you will author their bodies against these "
                         "fixed slots):\n" + json.dumps(required)) if required else \
                        "\n\nPIPELINE-REQUIRED MIGRATION ADRS: none - no departures, nothing migrates."
            optional = _call_json(
                "architecture",
                context + "\n\nDECIDED STACK (key: value):\n"
                + json.dumps({k: e["value"] for k, e in entries.items()})
                + "\n\nCOMPONENT SLOTS:\n" + json.dumps(part["component_slots"])
                + req_block + "\n\n" + ARCH_ADR_PLAN + feedback,
                log, progress,
                validate=lambda d: _validate_adr_plan(d, [s["slug"] for s in required]),
                schema=arch_adr_plan_schema(),
                stream_items=False, item_label="plan: decision slots")
            part["adr_slots"] = _merge_adr_slots(required, optional)
            _save()

        # 6. One micro-call per ADR: the decision substance.
        adr_bodies = part.setdefault("adr_bodies", {})
        for slot in part["adr_slots"]:
            did = slot["decision_id"]
            if did in adr_bodies:
                continue
            def _v_adr(d, slot=slot):
                body = {"title": str(d.get("title", "")).strip(),
                        "decision": str(d.get("decision", "")).strip(),
                        "rationale": str(d.get("rationale", "")).strip(),
                        "driver": d.get("driver"),
                        "options_considered": d.get("options_considered") or [],
                        "baseline_refs": [str(r) for r in d.get("baseline_refs") or []],
                        "migration": slot["migration"],
                        "evidence": [str(x) for x in d.get("evidence") or []],
                        "affected_repositories": [str(x) for x in d.get("affected_repositories") or []],
                        "rejected": []}
                if not body["title"] or not body["decision"] or not body["rationale"]:
                    raise ModelStructureError(f"{slot['decision_id']}: title, decision and rationale are all required")
                if slot["migration"]:
                    if not body["evidence"]:
                        raise ModelStructureError(f"{slot['decision_id']} {body['title']}: migration requires evidence")
                    if not body["affected_repositories"]:
                        raise ModelStructureError(f"{slot['decision_id']} {body['title']}: migration requires affected_repositories")
                elif body["evidence"] or body["affected_repositories"]:
                    raise ModelStructureError(f"{slot['decision_id']} {body['title']}: evidence/affected_repositories only belong on migration ADRs")
                return body
            adr_bodies[did] = _call_json(
                "architecture",
                context + "\n\nTHE ADR SLOT (fixed - ids, migration flag and authorizations are pipeline-owned):\n"
                + json.dumps({"decision_id": did, "slug": slot["slug"],
                              "migration": slot["migration"], "authorizes": slot["authorizes"]})
                + "\n\nDECIDED STACK (key: value):\n"
                + json.dumps({k: e["value"] for k, e in entries.items()})
                + "\n\n" + ARCH_ADR_BODY + feedback,
                log, progress, validate=_v_adr,
                schema=arch_adr_body_schema(valid_refs, repo_names),
                stream_items=False, item_label=f"{did} {slot['slug']}")
            _save()

        # 7. One micro-call per non-functional requirement: the position.
        nfr = part.setdefault("non_functionals", {})
        for rid, req in nfr_slots.items():
            if rid in nfr:
                continue
            def _v_nfr(d, rid=rid):
                if not str(d.get("position", "")).strip():
                    raise ModelStructureError(f"{rid}: non-functional position must not be empty")
                return str(d["position"]).strip()
            nfr[rid] = _call_json(
                "architecture",
                context + "\n\nTHE NON-FUNCTIONAL REQUIREMENT:\n" + json.dumps(req)
                + "\n\n" + ARCH_NFR + feedback,
                log, progress, validate=_v_nfr, schema=arch_nfr_schema(),
                stream_items=False, item_label=f"non-functional:{rid}")
            _save()

        # 8. Deterministic assembly + the whole-architecture semantic gate.
        merged = _assemble_architecture(
            session_id=sid, baseline_id=baseline.baseline_id, entries=entries,
            comp_slots=part["component_slots"], comp_bodies=comp_bodies,
            adr_slots=part["adr_slots"], adr_bodies=adr_bodies, nfr=nfr)
        try:
            arch = ArchitectureOverview.model_validate(merged)
            _check_decision_drivers(arch, state["intent_model"])  # ADR-0021
            _check_baseline_reconciliation(arch, baseline)
        except (ValidationError, ModelStructureError) as e:
            # the gate failing is a KNOWN pain (a slot the per-slot checks could
            # not see) - it gets the human-readable shape, never a raw traceback
            # (the 2026-09-12 acceptance run dumped one on the founder).
            from .errors import UserFacingError
            raise UserFacingError(
                "stage 'architecture': the assembled architecture failed the whole-architecture semantic gate.",
                _contract_errors(e),
                f"{model_config._rerun_guidance('architecture')} - per-slot work is preserved and any slot "
                "that fails the current contract is re-asked, not the whole stage; if the same gate error "
                "repeats after the re-ask, report it: a persistent gate failure is a pipeline bug, not a model limit.",
            ) from e
        if partials:
            partials.clear("architecture")
        data = merged
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

        fixes = _normalized_review_fixes(state.get("review_fixes"))
        affected = _affected_spec_ids(state, fixes) if fixes else set()
        selective = bool(fixes) and affected is not None
        if selective and state.get("items"):
            plan = [{k: v for k, v in spec.items() if k != "body_markdown"} for spec in state["items"]]
            data = []
            for spec, prior in zip(plan, state["items"]):
                if spec["spec_id"] not in affected:
                    data.append(prior); continue
                body = _call_json("items", _items_body_payload(state, spec), log, progress,
                                  validate=_items_body_validator(spec), schema=items_body_schema(),
                                  stream_items=False, item_label=f"{spec['spec_id']} surgical review fix")
                data.append({**spec, "body_markdown": body["body_markdown"]})
        elif model_config.bulk_fills_enabled():
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
                                  item_label=((f"{spec['spec_id']} ({spec['slug']})"
                                               if spec.get("slug") else spec["spec_id"])
                                              + f" {spec.get('title', '')}").strip())
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
        done = {t["test_id"]: {k: t.get(k) for k in ("description", "runner", "definition")}
                for t in state.get("tests") or []}
        if partials:
            done.update(partials.load("tests_fills") or {})
        fixes = _normalized_review_fixes(state.get("review_fixes"))
        req_targets = _fix_targets(fixes, "requirement") if fixes else set()
        affected_specs = _affected_spec_ids(state, fixes) if fixes else set()
        test_targets = _fix_targets(fixes, "test") if fixes else set()
        selective = bool(fixes) and None not in (req_targets, affected_specs, test_targets)
        if fixes and not selective:  # systemic/legacy review: full safe fallback
            done = {}
        affected_criteria = {c.get("criterion_id") for sec in ("functional", "non_functional")
                             for r in state["srs"].get(sec) or [] if r.get("requirement_id") in set(req_targets or [])
                             for c in r.get("acceptance") or []}
        missing = [sl for sl in slots if (sl["test_id"] not in done or
                   (selective and (sl["test_id"] in set(test_targets or []) or
                                   sl.get("spec_id") in set(affected_specs or []) or
                                   sl.get("criterion_id") in affected_criteria)))]
        if missing:
            if model_config.bulk_fills_enabled() and not selective:
                # Legacy one-call path (SPEC_GENERATOR_BULK=1): one streamed
                # call for every missing slot. Cheaper on prefill, but the
                # call's wall time grows with the slot count - 36 fills at
                # ~14s each is already at the 600s call budget, which is why
                # per-slot calls below are the default.
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
                          + json.dumps(missing_ids) + ". Each fill is {description, runner, definition}; definition is "
                            "{preconditions: [string], steps: [string], expected_outcomes: [string], cleanup: [string], "
                            "cases: [{name, inputs: [string], expected_outcomes: [string]}], execution: {url?, method?, "
                            "body_json?, expected_status?, command?, expected_exit?, expected_stdout_contains?, selector?}}. "
                            "Include all six outer fields; use [] or {} when inapplicable. runner and definition describe how "
                            "to test THE FOUNDER'S PRODUCT - "
                            "never this pipeline.", log, progress,
                              validate=_validate_new, schema=test_fill_schema(missing),
                              on_item=(_keep if partials else None))
                fills = {**done, **(out.get("fills") or {})}
            else:
                # ADR-0022: one fill per call, like the srs fills and spec
                # bodies - a weak fill fails one cheap call, a hung slot is
                # the visible last start line, and a kill reruns ONLY the
                # missing slots (partials persist per slot).
                for sl in missing:
                    tid = sl["test_id"]
                    link = sl.get("criterion_id") or sl.get("spec_id") or ""

                    def _validate_one(d, sl=sl):
                        for t in merge_test_fills([sl], d):
                            AcceptanceTest.model_validate(t)
                        return d

                    # Shared prefix FIRST (SRS+ITEMS are identical across
                    # slots - ollama's prefix cache then makes slot 2..N
                    # prefills nearly free), the varying slot LAST.
                    out = _call_json("tests",
                          STAGE_PROMPTS["tests"]
                          + "\n\nLINKED CONTEXT + ID GLOSSARY:\n" + json.dumps(_prompt_view(_compact_test_context(state, sl)))
                          + "\n\nTEST SLOT (fill its prose):\n" + json.dumps(sl)
                          + "\n\nFORMAT CONTRACT (final instruction - overrides anything above that conflicts): "
                            "return one STRICT JSON object {fills: {...}} with EXACTLY this one key: "
                          + json.dumps([tid]) + ". The fill is {description, runner, definition}; definition is "
                            "{preconditions: [string], steps: [string], expected_outcomes: [string], cleanup: [string], "
                            "cases: [{name, inputs: [string], expected_outcomes: [string]}], execution: {url?, method?, "
                            "body_json?, expected_status?, command?, expected_exit?, expected_stdout_contains?, selector?}}. "
                            "Include all six outer fields; use [] or {} when inapplicable. runner and definition describe how "
                            "to test THE FOUNDER'S PRODUCT - "
                            "never this pipeline.", log, progress,
                              validate=_validate_one, schema=test_fill_schema([sl]), stream_items=False,
                              item_label=f"{tid} ({sl['level']}: {link})")
                    done[tid] = (out.get("fills") or {})[tid]
                    if partials:
                        partials.save("tests_fills", done)
                fills = done
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
                           + "\n\nTESTS:\n" + json.dumps(_prompt_view(state["tests"])), log, progress,
                           validate=_normalized_review_fixes)
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
        fixes = _normalized_review_fixes(state["review_fixes"])
        req_targets = _fix_targets(fixes, "requirement")
        if req_targets is None:
            data = _run_srs_fill_stage(state, "REVIEW FIXES - apply EVERY one:\n" + json.dumps(fixes))
        elif req_targets:
            data = _run_srs_fill_stage(state, "SCOPED REVIEW FIXES:\n" + json.dumps(fixes), only=req_targets)
        else:
            data = state["srs"]
        data.setdefault("session_id", state["intent_model"]["session_id"])
        data.setdefault("intent_model_ref", state["intent_model"]["session_id"])
        if persist and data is not state["srs"]:
            persist("srs", data, state)
        return {"srs": data, "review_fixes": fixes}

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
    if only_stage:
        if only_stage not in ("srs", "architecture", "items", "tests", "review"):
            raise ValueError(f"unsupported only_stage: {only_stage}")
        g.set_entry_point(only_stage)
        g.add_edge(only_stage, END)
        return g.compile(checkpointer=checkpointer or MemorySaver(serde=checkpoint_serde()))

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



