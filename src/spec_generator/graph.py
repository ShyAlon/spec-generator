"""Generation pipeline as a LangGraph state machine.

LangGraph earns its place here (ADR-0003): staged drafts with independent
review passes (3 drafts + 2 review rounds is table stakes), resumable
background execution (Vercel's 300s cap means this runs as a job, not a
request), and a deterministic assembly tail that no LLM touches.
"""
from __future__ import annotations

import copy
import json
import re as _re
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

from .tech_candidates import (KIND_LEXICONS, PLATFORM_CHECKED_KINDS,
                              SLOT_KEY_SUBSTR, TECH_CANDIDATES,
                              TECH_CANDIDATES_AS_OF, primary_row_in_text,
                              tech_row, resolve_technology, TECH_ROWS)
from .tools import tools_enabled
from .skeleton import (ModelStructureError, assign_architecture_ids,
                                   build_srs_skeleton, build_test_skeleton,
                                   merge_srs_fills, merge_test_fills)
from .assembly import build_traceability, order_milestones, validate_traceability
from .prompts import METHODOLOGY, STAGE_PROMPTS
from .telemetry import UsageLog
from . import content_acceptance as _ca
from .errors import AssemblyBlocked, ContentRejected
from .findings_ledger import Ledger, LedgerError, finding_from_check, revision_of

# JEV-D10: review never triggers repair on its own. A scoped repair runs only on an explicit request
# (state["repair_request"], set by --repair) and is bounded by this ceiling. PLACEHOLDER: the founder
# left the number blank; reaching it leaves findings OPEN and assembly blocked, never "success".
REPAIR_ATTEMPT_CEILING = 2
REVIEW_CHUNK_MAX_FINDINGS = 24  # per-chunk bound by construction; overflow is recorded as incomplete


class GenState(TypedDict, total=False):
    intent_model: dict
    system_baseline: dict
    conops: dict               # founder-confirmed Concept of Operations; absent only for legacy sessions
    needs: dict
    srs: dict
    requirements: dict
    semantic_reviews: dict
    guard_flags: list[str]
    architecture_advisories: list[dict]
    expert_opinions: list[dict]
    architecture: dict
    items: list[dict]
    tests: list[dict]
    review_round: int
    review_fixes: list[dict]
    bundle: dict
    violations: list[str]
    gate_feedback: dict[str, str]  # stage -> founder feedback from a review gate (ADR-0019)
    artifact_revisions: dict[str, dict]  # stage -> {saved_at, revision}; gate staleness evidence
    findings_ledger: dict      # JEV-D10: the one findings ledger (open/addressed/rejected), revision-bound
    repair_quick: bool         # --quick repair: auto-approve the SRS fix, skip items/tests, re-review the repaired requirements, stop
    quick_stop: bool           # a quick repair finished its review: end here (no assembly) until the founder decides
    repair_proposal: dict      # repaired SRS awaiting founder approval {finding_ids, changed, unchanged_targets, srs}; applied only by --approve-repair
    repair_scope: list         # requirement ids the next review round re-checks after a scoped repair; absent = all
    repair_request: dict       # JEV-D10: explicit scoped repair {finding_ids, requested_by}; never set by the pipeline itself


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

    # Expert verdicts deliberately use a plain-text protocol. Do not ask the
    # provider factory for a JSON grammar that cannot exist for this stage.
    model = model_config.build_chat_model(stage, structured=(stage != "expert_opinion"), schema=schema)
    # ADR-0022: per-slot micro-calls skip the per-token item scanner (their
    # output is one small object; the item_label lines cover them instead).
    reporter = ItemReporter(progress if stream_items else None, stage, on_item=on_item)
    t0 = _t.time()
    if item_label and progress is not None:
        progress.emit(ProgressEvent(stage, "", "item_started", t0, None, item_label))
    messages = [SystemMessage(content=METHODOLOGY), HumanMessage(content=payload)]
    if feedback:
        messages.append(HumanMessage(content=feedback))
    if stage != "expert_opinion":
        messages.append(HumanMessage(content="FINAL RESPONSE CONTRACT: Return only the JSON value requested above; no prose or markdown fences."))
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
    if isinstance(exc, ModelStructureError) or not callable(getattr(exc, "errors", None)):
        return str(exc)
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in exc.errors()[:5])


_CONTROL_FRAME = ("PIPELINE CONTROL NOTE - not product content. Never copy, quote or paraphrase this note "
                  "or any pipeline instruction inside the artifact you return. ")


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
    attempt_diagnostics = []
    for attempt in (1, 2):
        try:
            raw = _invoke(stage, payload, log, progress, feedback=fb, schema=schema,
                          stream_items=stream_items, item_label=item_label, on_item=on_item,
                          attempt=attempt, retry_reason=("contract" if fb else None))
        except __import__("vmodel_contracts.runtime", fromlist=["AdaptiveWatchdogTimeout"]).AdaptiveWatchdogTimeout as exc:
            if attempt == 1:
                basis = f"{exc.ratio:.1f}x stage median" if exc.ratio is not None else "the cold-start floor"
                model_config._console_line(
                    f"  WARNING self-heal: call ran past {basis} ({exc.deadline_s:.0f}s); killed and retrying")
                # Control-plane fix (JEV-D10): the retry sends NO instruction text. The old
                # sentence entered the same channel as the product content and a model echoed it
                # into SPEC-4's body. A watchdog retry is the same request again.
                fb = None
                continue
            from .errors import UserFacingError
            raise UserFacingError(
                f"stage '{stage}': the adaptive call watchdog exhausted its retry budget.",
                f"the call exceeded {exc.deadline_s:.0f}s twice; only the in-flight item was discarded.",
                model_config._rerun_guidance(stage) + "; completed item partials are preserved.") from exc
        last_raw = raw
        attempt_diagnostics.append(model_config.call_diagnostic(stage, item_label, attempt))
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
            except (ValidationError, ModelStructureError, ValueError, KeyError, TypeError) as e:
                contract_err = e
        from .errors import UserFacingError

        snippet = raw.strip()[:200].replace("\n", " ")
        rejection = (_contract_errors(contract_err) if contract_err is not None
                     else f"invalid-json: {type(parse_err).__name__}: {str(parse_err)[:160]}")
        rejected_path = model_config.capture_rejected_completion(
            stage, item_label, attempt, raw, rejection)
        if attempt == 2:
            if contract_err is not None:
                raw_path = rejected_path
                details = "; ".join(
                    f"attempt {d.get('attempt')}: {d.get('duration_s', '?')}s, "
                    f"{d.get('tokens', '?')} tokens, {d.get('provider')}/{d.get('model')}"
                    for d in attempt_diagnostics)
                feedback_sent = fb or "(none on first attempt)"
                raise UserFacingError(
                    f"stage '{stage}': the model's output does not satisfy the contract (an automatic retry did not fix it).",
                    f"contract errors: {_contract_errors(contract_err)}. "
                    f"Retry feedback sent: {feedback_sent}. Call diagnostics: {details}. "
                    f"Full failed output: {raw_path}.",
                    f"{model_config._rerun_guidance(stage)}; inspect the full output path above. "
                    "Deterministic normalization handles known phi echoes; capture a new fixture so it can be repaired without a model upgrade.",
                ) from contract_err
            raise UserFacingError(
                f"stage '{stage}': the model returned output that is not valid JSON (an automatic retry did not fix it).",
                f"smaller/local models often wrap JSON in prose, drop structure, or truncate. Raw output began: {snippet!r}",
                f"{model_config._rerun_guidance(stage)}; if it repeats, route the stage "
                f"to a stronger model (SPEC_GENERATOR_MODEL_{stage.upper()} env override).",
            ) from parse_err
        if progress is not None:
            progress.emit(ProgressEvent(stage, "", "item", _t.time(), None,
                                        (("WARNING contract revision requested: " + _contract_errors(contract_err))
                                         if contract_err is not None else "WARNING JSON format revision requested")
                                        + " - retrying once with feedback"))
        model_config._console_line(
            f"     - boundary=contract-repair stage={stage} item={item_label or '(none)'} attempt={attempt} "
            f"decision=retry reason={rejection} rejected_output={rejected_path}")
        if contract_err is not None:
            fb = (_CONTROL_FRAME + "Your previous reply failed the output contract: " + _contract_errors(contract_err) + ". "
                  "Return ONLY the complete corrected JSON payload - every required field present at every "
                  "level, ids and enums exactly as specified.")
        else:
            fb = (_CONTROL_FRAME + "Your previous reply was not valid JSON (prose-wrapped, unbalanced, or truncated). "
                  "Return ONLY the complete JSON payload, no prose, no commentary, balanced to the end.")


def srs_document_data(intent_model: dict, srs_body: dict) -> dict:
    """Full SRSDocument data from the LLM-authored body: structure (definitions,
    constraints, assumptions, open questions) is carried DETERMINISTICALLY from
    the intent model - the LLM never re-copies it. Used at assembly AND at the
    review gate (validating hand edits)."""
    intent = IntentModel.model_validate(intent_model)
    data = dict(srs_body)
    # Architecture may answer an interview-time question after the SRS stage.
    # Keep the exact resolved questions as code-owned state, but never leak the
    # private marker into the SRSDocument contract.
    resolved = set(data.pop("_resolved_open_questions", []))
    if intent.founder_register is not None:data["founder_register"]=intent.founder_register.model_dump(mode="json")
    data["definitions"] = intent.glossary
    data["constraints"] = intent.constraints
    constraint_turns = intent.source_turns.get("constraints", [])
    if intent.constraints and not constraint_turns:
        raise ValueError("design constraints require source_turns['constraints'] provenance from the interview")
    data["design_constraints"] = [
        {"constraint_id": f"DC-{index}", "statement": statement,
         "source_field": "constraints", "source_turns": list(constraint_turns)}
        for index, statement in enumerate(intent.constraints, 1)
    ]
    data["assumptions"] = [a.model_dump(mode="json") for a in intent.assumption_items]
    data["open_questions"] = [q for q in intent.open_questions if q not in resolved]
    # Future-phase commitments (founder decision 2026-09-13): a need deferred
    # IN the interview drives NO current requirement (build_srs_skeleton skips
    # it) and is recorded HERE, in the scope, as what it is - a later
    # commitment, not a contradiction ('the app WILL have a community feature
    # - in the future' vs scope 'community: OUT' read as a pipeline bug in his
    # 2026-09-13 habit-app bundle). Code-owned block: stripped and rewritten
    # so the review gate's re-validation stays idempotent.
    from .skeleton import is_future_commitment
    deferred = [n.statement for n in intent.stakeholder_needs if is_future_commitment(n.statement)]
    if deferred:
        summary = _re.sub(r"\n*Deferred to a future phase.*$", "",
                          str(data.get("scope_summary", "")), flags=_re.S).rstrip()
        data["scope_summary"] = (summary + "\n\nDeferred to a future phase (interview "
                                 "commitments, NOT current scope - no requirements derive "
                                 "from these yet):\n"
                                 + "\n".join(f"- {st}" for st in deferred))
    return data


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


def _baseline_vocab(baseline: SystemBaseline) -> str:
    """Every baseline fact text a stack value may quote: the descriptive fields
    AND the structural facts (entry points, artifacts, contracts). The model is
    asked to cite module/artifact refs, so their verbatim text must be quotable
    - the 2026-09-12 module_entry_points rejection happened because the check
    excluded the very facts the entry cited, and the model's VERBATIM copy
    ('python -m spec_generator.cli') failed."""
    fields = ("languages", "runtime", "persistence", "integrations",
              "immutable_decisions", "package_entry_points", "module_entry_points",
              "existing_artifacts", "existing_contracts")
    return " ".join(v for r in baseline.repositories for f in fields
                    for v in getattr(r, f)).lower()


def _value_grounded(value: str, vocab: str) -> bool:
    """Grounded = the fact's verbatim text (substring), OR a composition of
    real fact tokens (a command assembled FROM the cited facts). Fabrication
    stays closed: any token that appears in no baseline fact (an invented
    name like 'auth0') fails. One-letter tokens (flags like -m) are ignored."""
    import re as _re

    v = str(value).lower()
    if v in vocab:
        return True
    tokens = {t for t in _re.split(r"[^a-z0-9]+", v) if len(t) > 1}
    vocab_tokens = set(_re.split(r"[^a-z0-9]+", vocab))
    return bool(tokens) and tokens <= vocab_tokens


# A value SHAPED like an entry point (a command invocation, a script path, a
# module:attr string) answers only an entry-point slot - never "framework",
# "ci" or "encryption" (the 2026-09-12 regen stuffed 'python -m
# spec_generator.cli' into all three). Deterministic kind check, ADR-0015
# doctrine: mechanical guards over prompt discipline.
_ENTRY_POINT_VALUE = _re.compile(
    r"^(python3?|node|ruby|php|java|dotnet|uvicorn|gunicorn|flask|npm|npx|pip|bash|sh)\s"  # command invocation
    r"|[\w\-.~]*/[\w\-./]*\.(sh|py|js|mjs|ts|rb)(\s|$)"   # script PATH (slash required)
    r"|[\w_][\w_.\-]*:[\w_][\w_.\-]*(\s|$)"  # module:attr (delivery_api.app:app)
    r"|\s-m\s", _re.IGNORECASE)                    # '-m module' anywhere
_ENTRY_POINT_KEY = _re.compile(
    r"entry[-_ ]?point|entrypoint|command|\bcli\b|launch|start[-_ ]?up|script", _re.IGNORECASE)


def _entry_point_shaped(value: str) -> bool:
    """Shaped like a command/script/module:attr. The script branch requires a
    path separator (2026-09-13 his live greenfield run): a bare 'D3.js' /
    'Vue.js' / 'Node.js' / 'Chart.js' is a technology NAME - script
    detection needs invocation structure (a slash, 'python -m', module:attr),
    not a bare extension. '.js/.ts' are how half the JS ecosystem spells its
    proper nouns."""
    return bool(_ENTRY_POINT_VALUE.search(value.strip()))


def _entry_point_slot(key: str) -> bool:
    return bool(_ENTRY_POINT_KEY.search(key))


def _experiment_cap(name: str, items: list, label: str,
                    progress=None) -> list:
    """VMODEL_CAP_<name> (env or the shared .env): deterministic scope cap for
    scope-limited experiment runs (his 2026-09-13 three-speed design: MEDIUM =
    capped end-to-end smoke in minutes, FULL = the uncapped acceptance run).
    A cap TRUNCATES the named list and announces itself - a silent cap would
    fake the very quality signal the experiment measures."""
    raw = model_config._env(f"VMODEL_CAP_{name}")
    if not raw:
        return items
    try:
        cap = int(raw)
    except ValueError:
        raise ModelStructureError(
            f"VMODEL_CAP_{name}={raw!r} is not an integer - fix the shared .env "
            "or unset it (caps are for experiment runs only)")
    if cap < 1 or len(items) <= cap:
        return items
    if progress is not None:
        from .progress import ProgressEvent
        import time as _t
        progress.emit(ProgressEvent("experiment", "", "item", _t.time(), None,
                                    f"experiment cap: {label} {len(items)} -> {cap} "
                                    f"(VMODEL_CAP_{name}); FULL runs stay uncapped"))
    return items[:cap]


# P3 layer 2, deterministic (founder directive 2026-09-12: deterministic logic
# over LLM judgment wherever possible). A stack value that matches a KNOWN
# technology lexicon of one kind cannot answer a slot of another kind
# ('Pydantic' for encryption, 'ts' for database). Values in NO lexicon (novel
# tech) pass - a lexicon miss is never a verdict. The lexicons are DERIVED
# from the tech candidate matrix (2026-09-13 matrix decision: ONE artifact
# owns all tech knowledge - kinds, platforms, aliases, notes, freshness).
_KIND_LEXICONS = KIND_LEXICONS
_SLOT_KIND = {
    "language": "language", "runtime": "language", "framework": "framework",
    "web_framework": "framework", "ui_framework": "framework", "database": "database", "persistence": "database",
    "orm": "orm", "ci": "ci", "ci_tool": "ci", "auth": "auth",
    "sso_integration": "auth", "authentication": "auth", "encryption": "encryption",
    "message_queue": "queue", "queue": "queue",
    "visualization": "visualization", "visualization_library": "visualization",
    "data_visualization_library": "visualization", "charts": "visualization",
    "charting_library": "visualization", "graphing_library": "visualization",
}


def _known_tech(value: str) -> bool:
    """A BARE recognized technology name (any kind lexicon). Membership takes
    precedence over the entry-point SHAPE rule: a recognized name is never a
    command (founder directive 2026-09-13, after 'D3.js' was rejected from
    his live run as 'an entry point'). Precedence applies to bare names only
    - anything carrying invocation structure (whitespace, '/', ':') is judged
    by its shape, so 'python -m spec_generator.cli' stays an entry point even
    though it CONTAINS the known tech 'python'."""
    if _re.search(r"[\s:/]", value.strip()):
        return False
    return any(_kind_match(value, kind) for kind in _KIND_LEXICONS)


def _kind_match(value: str, kind: str) -> bool:
    v = value.lower()
    return any(_re.search(rf"(?<![\w+.#]){_re.escape(term)}(?![\w])", v)
               for term in _KIND_LEXICONS[kind])


def _wrong_kind(key: str, value: str) -> str | None:
    """Name of the lexicon kind this value belongs to when it is NOT the slot's kind."""
    slot_kind = _SLOT_KIND.get(key)
    if not slot_kind or _kind_match(value, slot_kind):
        return None
    return next((k for k in _KIND_LEXICONS if k != slot_kind and _kind_match(value, k)), None)


# ---- Founder candidates + curated tech candidate lists (founder decision
# 2026-09-13): a founder-NAMED technology is adopted or rejected, never
# dropped from the record; the dated curated list seeds candidates so the
# model's training cutoff is not its only source. Suggestions, not
# constraints - off-list picks stay legal but an ADR must justify them.

_CANDIDATE_DISPLAY = {n.lower(): n for c in TECH_CANDIDATES.values() for n, _ in c}


def _display(tech: str) -> str:
    return _CANDIDATE_DISPLAY.get(tech, tech)


def _candidate_kind(key: str) -> str | None:
    """The slot's candidate kind: the fitness gate's exact map, then the
    matrix's substring rules for model-proposed keys."""
    if key in _SLOT_KIND:
        return _SLOT_KIND[key]
    kl = key.lower()
    for sub, kind in SLOT_KEY_SUBSTR:
        if sub in kl:
            return kind
    return None


# Product platform context, derived deterministically from the intent model
# (2026-09-13 matrix decision): an iOS mention (or an iOS-only integration
# like HealthKit) makes UI-kind slots require a matching-platform candidate.
_PLATFORM_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ios", ("ios", "iphone", "ipad", "healthkit", "app store")),
    ("android", ("android", "health connect", "google play")),
    ("web", ("web app", "web application", "webapp", "browser", "pwa")),
    ("desktop", ("desktop",)),
)


def _product_platforms(intent_model: dict) -> set[str]:
    text = json.dumps({k: intent_model.get(k) for k in
                       ("problem_statement", "target_users", "core_value_proposition",
                        "stakeholder_needs", "in_scope", "constraints",
                        "external_integrations")}, default=str).lower()
    return {plat for plat, kws in _PLATFORM_KEYWORDS if any(k in text for k in kws)}


def _tech_in_text(name: str, text: str) -> bool:
    return bool(_re.search(rf"(?<![\w+.#]){_re.escape(name.lower())}(?![\w])", text.lower()))


def _tech_kinds(name: str) -> list[str]:
    """Every kind a technology belongs to - lexicon kinds, then candidate
    kinds (supabase: database + auth + backend)."""
    kinds = [k for k in _KIND_LEXICONS if _kind_match(name, k)]
    n = name.lower()
    kinds += [k for k, cands in TECH_CANDIDATES.items() if k not in kinds
              and any(cn.lower() == n for cn, _ in cands)]
    return kinds



def _founder_required_tech(intent_model: dict, key: str, value: str) -> bool:
    """Only a sourced, exact founder choice bypasses platform contest.

    Preference areas use contract areas (framework), not arbitrary slot names
    (ui_framework). Needs must state an actual technology directive, not merely
    mention a technology while describing product behavior.
    """
    kind = _candidate_kind(key)
    area = "framework" if kind in ("framework", "mobile_framework") else kind
    if not area:
        return False
    from .tech_candidates import normalize_technology_alias
    desired = normalize_technology_alias(value)
    # Names are case-insensitive, but aliases and families do not grant an
    # exemption: "React Native" is not a founder choice of "React".
    for pref in intent_model.get("tech_preferences") or []:
        if (pref.get("source_turns") and str(pref.get("area", "")).casefold() == area
                and normalize_technology_alias(str(pref.get("preference", ""))) == desired):
            return True
    for need in intent_model.get("stakeholder_needs") or []:
        if not need.get("source_turns"):
            continue
        statement = str(need.get("statement", ""))
        # "Use React Native" must never exempt React by prefix. The need's
        # chosen technology is the longest matching registered name.
        directive = r"\b(?:use|using|require|requires|required|must use|must be built with)\s+"
        longer = any(r["name"].casefold().startswith(value.casefold() + " ")
                     and _re.search(directive + _re.escape(r["name"]) + r"(?=$|[\s.,;:!?])",
                                    statement, _re.I) for r in TECH_ROWS)
        if not longer and _re.search(directive + _re.escape(value) + r"(?=$|[\s.,;:!?])",
                                     statement, _re.I):
            return True
    return False

def _founder_named_techs(intent_model: dict) -> list[str]:
    """Technologies the founder NAMED inside their stated tech preferences
    ('free resources like Supabase' -> ['supabase']). Only names the pipeline
    itself recognizes (lexicon or candidate list) are extractable; a
    preference naming nothing known injects nothing and gates nothing."""
    vocab = {t for terms in _KIND_LEXICONS.values() for t in terms}
    vocab |= {n.lower() for c in TECH_CANDIDATES.values() for n, _ in c}
    # Founder tech mentions live in TWO places (2026-09-13 greenfield defect:
    # 'the app should use free resources like Supabase where possible' was
    # need N-8, not a tech_preference, and the extraction missed it entirely
    # - no candidate injection, no enforced verdict): stated preferences AND
    # need statements. A need naming a technology IS a founder constraint on
    # that technology.
    texts = [str(pref.get("preference", ""))
             for pref in (intent_model.get("tech_preferences") or [])]
    # Needs are founder technology evidence only when their wording explicitly
    # makes a technology choice. Bare dictionary collisions are not evidence:
    # "claims clerk", "it goes", and "react in one go" previously fabricated
    # Clerk, Go, and React mandates from ordinary product prose.
    tech_choice = _re.compile(
        r"\b(?:use|using|built (?:with|on)|implement(?:ed)? (?:with|in)|"
        r"run(?:s|ning)? (?:on|with)|host(?:ed)? on|deploy(?:ed)? (?:on|with)|"
        r"prefer(?:s|red)?|mandat(?:e|ed)|require(?:s|d)?|resources? like)\b",
        _re.I)
    texts += [statement for n in (intent_model.get("stakeholder_needs") or [])
              if tech_choice.search(statement := str(n.get("statement", "")))]
    out = set()
    for text in texts:
        for v in vocab:
            if _tech_in_text(v, text):
                out.add(v)
    return sorted(out)


_GENERIC_REJECTIONS = (
    "not suitable", "not a good fit", "not a fit", "does not fit", "doesn't fit",
    "better alternatives", "better alternatives exist", "better options exist",
    "better option", "less suitable", "not ideal", "not appropriate",
    "not the best", "inferior", "not needed", "unnecessary", "overkill",
)


def _specific_reason(reason: str) -> bool:
    """A rejection reason earns its place by naming WHAT the option fails -
    a requirement, a constraint, a missing capability. Bare verdicts
    ('not suitable') and sub-20-character shrugs re-ask (founder decision
    2026-09-13)."""
    r = reason.strip().lower().rstrip(".!")
    if len(r) < 20:
        return False
    return r not in _GENERIC_REJECTIONS


def _candidate_block(key: str, founder_techs: list[str]) -> str:
    """Per-stack-entry prompt injection: the dated curated candidates for the
    slot's kind, with founder-named technologies marked and weighted."""
    kind = _candidate_kind(key)
    cands = TECH_CANDIDATES.get(kind, ()) if kind else ()
    if key == "ui_framework":
        # An explicit UI slot includes both browser and native UI frameworks.
        # The broad framework kind also contains backend services, which are
        # never substitutes for the product's UI.
        cands = tuple((name, note) for name, note in
                      (cands + TECH_CANDIDATES.get("mobile_framework", ()))
                      if (row := tech_row(name))
                      and "backend" not in row["kinds"]
                      and ("web" in row["platforms"] or "ios" in row["platforms"]
                           or "android" in row["platforms"]))
        cands = tuple(dict.fromkeys(cands))
    lines = []
    if cands:
        lines.append(f"CANDIDATE TECHNOLOGIES for this slot's kind ({kind}) - curated list as of "
                     f"{TECH_CANDIDATES_AS_OF}; suggestions, NOT constraints (an off-list pick is "
                     "allowed, but an ADR must then name it and why it won):")
        lines += [f"- {name}: {note}" for name, note in cands]
    marked = [t for t in founder_techs if kind and kind in _tech_kinds(t)]
    if marked:
        lines.append("FOUNDER-NAMED CANDIDATE(S) for this slot: "
                     + ", ".join(_display(t) for t in marked)
                     + " - the founder named these in their stated preferences; weigh them heavily "
                     "(adopt unless a concrete technical reason says otherwise). The ADR covering "
                     "this slot records the verdict either way; a rejection must state the specific "
                     "requirement it fails ('not suitable' re-asks).")
    return ("\n\n" + "\n".join(lines)) if lines else ""


def _announce_candidate_lists(founder_techs: list[str]) -> None:
    """The freshness contract is VISIBILITY (founder decision 2026-09-13):
    every run log says how stale its candidate lists are."""
    model_config._console_line(
        f"   tech candidate lists as of {TECH_CANDIDATES_AS_OF} "
        "(suggestions, not constraints; refresh: python scripts/refresh_tech_candidates.py)")
    if founder_techs:
        model_config._console_line(
            "   founder-named candidates (weighted; verdicts tracked into the ADRs): "
            + ", ".join(_display(t) for t in founder_techs))


def _check_candidate_grounding(arch: ArchitectureOverview, intent_model: dict,
                               contested: dict | None = None) -> None:
    """Assembly gate (founder decision 2026-09-13): a founder-NAMED technology
    is adopted (a stack value names it) or rejected (an ADR's
    options_considered names it with a SPECIFIC reason) - never dropped from
    the record. And an off-list stack pick (no lexicon, no candidate list) is
    justified by an ADR naming it. Failures trigger ONE bounded ADR-set
    re-plan in node_architecture, then a human-readable stage error."""
    stack_values = [e.value for e in arch.stack_entries]
    all_options = [(d.decision_id, o.option, o.reason)
                   for d in arch.decisions for o in d.options_considered]
    for t in _founder_named_techs(intent_model):
        if any(_tech_in_text(t, v) for v in stack_values):
            continue  # adopted - the decision record carries the pick itself
        hits = [(did, o, r) for did, o, r in all_options if _tech_in_text(t, o)]
        if not hits:
            raise ModelStructureError(
                f"founder-named technology {_display(t)!r} is neither adopted in the stack nor "
                "rejected in any ADR's options_considered - the record must carry the verdict. "
                "Add it to the ADR covering its area, rejected, with the specific requirement it fails")
        for did, _o, r in hits:
            if not _specific_reason(r):
                raise ModelStructureError(
                    f"{did}: founder-named technology {_display(t)!r} rejected with the vague reason "
                    f"{r!r} - state the specific requirement it fails or the constraint it violates")
    # Contested off-platform picks (gate 3 insistence): the justification MUST
    # land in an ADR for founder veto - named in the decision, with the
    # platform trade-off argued explicitly in the rationale.
    for key, value in (contested or {}).items():
        hits = [d for d in arch.decisions if _tech_in_text(value, d.title + " " + d.decision)]
        if not hits:
            raise ModelStructureError(
                f"stack.{key}: contested off-platform pick {value!r} is justified in NO ADR - "
                "an ADR must own this decision and argue the platform trade-off explicitly "
                "(why it fits despite the product's platform)")
        if not any(_re.search(r"web|pwa|platform|mobile|cross.?platform|ios|android",
                              d.rationale.lower()) for d in hits):
            raise ModelStructureError(
                f"stack.{key}: the ADR(s) naming contested pick {value!r} never argue the "
                "platform trade-off - the rationale must state explicitly why a technology "
                "of its platform fits this product's platform")
    decisions_by_id = {d.decision_id: d for d in arch.decisions}
    for e in arch.stack_entries:
        if e.baseline_refs or _entry_point_slot(e.key) or _known_tech(e.value):
            continue
        if any(_tech_in_text(n.lower(), e.value) for c in TECH_CANDIDATES.values() for n, _ in c):
            continue  # on the curated list
        # The pipeline owns stack->ADR identity. Credit that structured link;
        # never require a long free-text pick to be repeated byte-for-byte in
        # prose (the insurer-adapter live run had the right migrate-stack slot
        # but failed this old substring test twice).
        decision = decisions_by_id.get(e.decision_id)
        selection = next((sel for sel in (decision.technology_selections if decision else [])
                          if sel.slot_id == "slot:" + _re.sub(r"[^a-z0-9]+", "-", e.key.casefold()).strip("-")), None)
        justification = (selection.off_list_justification if selection else None)
        if decision is None or selection is None:
            raise ModelStructureError(
                f"stack.{e.key}: off-list pick {e.value!r} has no structured owning ADR selection - "
                "the ADR must authorize this stack slot")
        if not justification or len(justification.strip()) < 20:
            raise ModelStructureError(
                f"stack.{e.key}: owning {decision.decision_id} links the off-list pick but does not say "
                "why it won over the curated candidates; provide a substantive off_list_justification")


def _duplicate_adr_pairs(arch: ArchitectureOverview) -> list[tuple[str, str, float, float]]:
    """Return only pairs meeting the deterministic duplicate threshold."""
    def words(text: str) -> set[str]:
        stop = {"a", "an", "and", "as", "at", "behind", "by", "for", "in", "of", "on",
                "the", "to", "use", "with", "adr", "decision"}
        return {w for w in _re.findall(r"[a-z0-9]+", text.casefold())
                if len(w) > 2 and w not in stop}
    def score(a: set[str], b: set[str]) -> float:
        return len(a & b) / max(1, len(a | b))
    found = []
    for i, left in enumerate(arch.decisions):
        for right in arch.decisions[i + 1:]:
            left_tech = {x.technology_id for x in left.technology_selections}
            right_tech = {x.technology_id for x in right.technology_selections}
            if left_tech and right_tech and left_tech.isdisjoint(right_tech):
                continue
            ls, rs = words(left.title + " " + left.decision), words(right.title + " " + right.decision)
            lo = {" ".join(sorted(words(o.option))) for o in left.options_considered}
            ro = {" ".join(sorted(words(o.option))) for o in right.options_considered}
            subject, options = score(ls, rs), score(lo, ro) if lo and ro else 0.0
            same_selection = bool(left_tech and left_tech == right_tech)
            duplicate = ((same_selection and subject >= .30)
                         or (len(lo & ro) >= 2 and subject >= .30 and options >= .65))
            near = len(lo & ro) >= 2 and subject >= .24 and options >= .52
            if duplicate or near:
                model_config._console_line(
                    f"     - boundary=adr-substance compared={left.decision_id},{right.decision_id} "
                    f"subject_overlap={subject:.2f} subject_threshold=0.30 alternatives_overlap={options:.2f} "
                    f"alternatives_threshold=0.65 shared_alternatives={len(lo & ro)} minimum=2 "
                    f"same_selection={same_selection} same_selection_subject_minimum=0.30 "
                    f"decision={'duplicate' if duplicate else 'pass-near-boundary'}")
            if duplicate:
                found.append((left.decision_id, right.decision_id, subject, options))
    return found


def _merge_duplicate_adr_pair(slots: list[dict], bodies: dict, left_id: str, right_id: str) -> None:
    """Losslessly fold proven duplicate metadata/ownership into the first ADR."""
    left_slot = next(slot for slot in slots if slot["decision_id"] == left_id)
    right_slot = next(slot for slot in slots if slot["decision_id"] == right_id)
    left, right = bodies[left_id], bodies[right_id]
    for key in ("authorizes", "required_concept_ids"):
        left_slot[key] = list(dict.fromkeys((left_slot.get(key) or []) + (right_slot.get(key) or [])))
    left_slot["migration"] = bool(left_slot.get("migration") or right_slot.get("migration"))
    for key in ("options_considered", "baseline_refs", "evidence", "affected_repositories",
                "executes_concept_ids", "rejected"):
        values = (left.get(key) or []) + (right.get(key) or [])
        # Dict/list values need structural identity, not hashability.
        seen=set(); merged=[]
        for value in values:
            identity=json.dumps(value,sort_keys=True,default=str)
            if identity not in seen: seen.add(identity); merged.append(value)
        left[key]=merged
    for key in ("title", "decision", "rationale"):
        if len(str(right.get(key) or "")) > len(str(left.get(key) or "")):
            left[key]=right[key]
    slots.remove(right_slot); bodies.pop(right_id, None)
    old_to_new={slot["decision_id"]:f"ADR-{i}" for i,slot in enumerate(slots,1)}
    rebuilt={}
    for slot in slots:
        old=slot["decision_id"]; slot["decision_id"]=old_to_new[old]; rebuilt[slot["decision_id"]]=bodies[old]
    bodies.clear(); bodies.update(rebuilt)


def _merge_duplicate_adrs_one_at_a_time(slots: list[dict], bodies: dict, pairs: list, assemble, label: str) -> int:
    """Merge proven duplicate ADR pairs losslessly, ONE pair per scan. A merge renumbers every later ADR id,
    so ids from the first scan are stale after it: re-assemble and re-scan before each merge.
    `assemble()` returns the current ArchitectureOverview. Returns how many merges ran."""
    done = 0
    for _ in range(len(slots)):
        if not pairs:
            break
        left_id, right_id, subject_score, alternatives_score = pairs[0]
        if left_id not in bodies or right_id not in bodies:
            break
        _merge_duplicate_adr_pair(slots, bodies, left_id, right_id)
        done += 1
        model_config._console_line(
            f"     - boundary={label} compared={left_id},{right_id} subject_overlap={subject_score:.2f} "
            f"alternatives_overlap={alternatives_score:.2f} "
            f"decision=merge-drop-{right_id}-retain-{left_id}-renumber reason=proven-duplicate-no-replan")
        try:
            pairs = _duplicate_adr_pairs(assemble())
        except (ValidationError, ModelStructureError, ValueError, KeyError):
            break
    return done


def _check_adr_substance(arch: ArchitectureOverview) -> None:
    """Selections and distinct ADR records must agree with their prose.

    Structured ownership alone is insufficient: a slot can be wired to an ADR
    whose body discusses a different decision. Duplicate records likewise hide
    ownership by making several ADRs appear to decide the same thing.
    """
    for decision in arch.decisions:
        substance = " ".join((decision.title, decision.decision, decision.rationale))
        for selection in decision.technology_selections:
            slot = selection.slot_id.casefold()
            high_risk = any(marker in slot for marker in (
                "orchestrat", "insurer", "integration", "interpret", "hosting", "telephon"
            ))
            if high_risk and not _tech_in_text(selection.display_name, substance):
                raise ModelStructureError(
                    f"{decision.decision_id}: technology selection {selection.display_name!r} "
                    "is absent from the ADR title, decision and rationale - the owning ADR must "
                    "substantively decide the selected technology, not only carry its slot link")

    for left_id, right_id, subject_overlap, option_overlap in _duplicate_adr_pairs(arch):
        raise ModelStructureError(
            f"{left_id} and {right_id} duplicate one architecture decision "
            f"(subject overlap {subject_overlap:.2f}, alternatives overlap {option_overlap:.2f}) - "
            "merge them or make each record own a genuinely distinct decision")


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
        baseline_vocab = _baseline_vocab(baseline)
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
                if not _value_grounded(e.value, baseline_vocab):
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
            offenders = ([f"component {c.name}" for c in arch.components
                          if c.migration or c.migration_adr_title]
                         + [f"{d.decision_id} {d.title}" for d in migration_list])
            raise ModelStructureError(
                "greenfield architecture is new design, not a migration - nothing exists to "
                f"migrate FROM; re-frame as initial decisions: {', '.join(offenders)}")
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


def _check_decision_drivers(arch: ArchitectureOverview, intent_model: dict, conops: dict | None = None) -> None:
    """ADR-0021: no arbitrary decisions. The contract requires the FIELDS
    present (driver nullable only for legacy migration); the stage requires a
    REAL driver on every new decision. A founder_constraint must cite either
    a recorded technology preference or the founder-confirmed CONOPS concept
    through executes_concept_ids; those are distinct sources of constraint.
    Options have no floor (founder call 2026-09-09): under a mandate, empty
    is honest."""
    prefs = [p.get("preference", "") for p in (intent_model.get("tech_preferences") or [])]
    for d in arch.decisions:
        if d.driver is None:
            raise ModelStructureError(
                f"{d.decision_id} '{d.title}': no driver - classify why the chosen option won "
                f"(founder_constraint | ecosystem_maturity | community_adoption | team_fit | simplicity).")
        if d.driver.value == "founder_constraint":
            prefs = [p for p in prefs if p]
            selected = _selected_concept(conops)
            executes_confirmed_concept = bool(selected and selected[0] in d.executes_concept_ids)
            text = (d.title + " " + d.decision + " " + d.rationale).lower()
            cites_tech_preference = any(p.lower() in text for p in prefs)
            if not executes_confirmed_concept and not cites_tech_preference:
                basis = (("recorded technology preferences: " + ", ".join(prefs)) if prefs else
                         "no recorded technology preference")
                raise ModelStructureError(
                    f"{d.decision_id} '{d.title}': driver founder_constraint requires either an "
                    "executes_concept_ids link to the founder-confirmed CONOPS concept or a cited "
                    f"technology preference ({basis})")


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


def _validate_component_plan(d, req_ids: list[str], existing: bool = True) -> list[dict]:
    """Plan call B result: component slots. The grammar enum-constrains
    requirement refs; here: unique names, real coverage, sane shape.
    Greenfield normalization (2026-09-13 his live run): the plan prompt says
    'migration=true for a new/replaced component', so on an EMPTY baseline
    the model marks every component migration=true - and the whole-arch gate
    then rejects 'greenfield architecture is new design, not a migration'.
    A correct run could never pass. Migration means departing from an
    existing baseline; with none, code owns the flag: forced False."""
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
                      "migration": bool(s.get("migration")) if existing else False})
    if len(names) != len(set(names)):
        raise ModelStructureError(f"duplicate component names: {sorted({n for n in names if names.count(n) > 1})}")
    covered = {r for s in slots for r in s["requirement_ids"]}
    missing = [r for r in req_ids if r not in covered]
    if missing:
        raise ModelStructureError(f"requirements covered by no component: {missing}")
    return slots


def _required_adr_slots(authorizable: list[str], migrating: set[str] | None = None) -> list[dict]:
    """Required ADR slots are pipeline-GENERATED, one per authorizable target
    (the 2026-09-12 live phi4 failure: the model planned a ci_tool-absence slot
    with migration:false and left stack:ci_tool unauthorized - a mechanical
    obligation the model forgot). Stable human-readable slugs derive from the
    target; the model only adds optional design ADRs and authors the prose.

    `migrating` = targets that really depart from an existing baseline. Only
    those get migration=True and a "migrate-" slug; every other target (all of
    them in a greenfield session) is an initial decision, slug "decide-". The
    slug is a label the model reads, so it must never contradict the
    code-owned flag (2026-10-02 greenfield run: "migrate-*" slugs on
    migration=False slots made the model fill migration-only fields).
    None = legacy behaviour, every target is a migration."""
    import re as _re

    slots, used = [], set()
    for target in authorizable:
        is_migration = True if migrating is None else target in migrating
        base = ("migrate-" if is_migration else "decide-") + _re.sub(r"[^a-z0-9]+", "-", target.lower()).strip("-")
        slug, n = base, 2
        while slug in used:
            slug, n = f"{base}-{n}", n + 1
        used.add(slug)
        slots.append({"slug": slug, "migration": is_migration, "authorizes": [target]})
    return slots


def _normalize_adr_slots(slots: list[dict], existing: bool) -> None:
    """Persisted ADR slots must satisfy the current mode's semantics on resume.
    Greenfield: nothing to migrate FROM, the flag is code-owned (forced False).
    A non-migration slot never carries a "migrate-" label (renamed "decide-")."""
    for slot in slots:
        if not existing:
            slot["migration"] = False
        slug = str(slot.get("slug", ""))
        if not slot.get("migration") and slug.startswith("migrate-"):
            slot["slug"] = "decide-" + slug.removeprefix("migrate-")


def _selected_concept(state_conops: dict | None) -> tuple[str, str] | None:
    """Return the founder-confirmed concept's stable id and display name."""
    if not state_conops:
        return None
    from vmodel_contracts import canonical_concept_id
    decision = state_conops.get("decision") or {}
    name = str(decision.get("selected_concept") or "").strip()
    concept_id = str(decision.get("selected_concept_id") or canonical_concept_id(name)).strip()
    return (concept_id, name) if concept_id and concept_id != "concept:" else None


def _required_concept_adr_slots(state_conops: dict | None) -> list[dict]:
    """A confirmed CONOPS choice is a mechanical architecture obligation.

    Give it a code-owned slot so one ADR body call can validate the exact link;
    do not hope an optional plan or a whole-architecture retry remembers it.
    """
    selected = _selected_concept(state_conops)
    if not selected:
        return []
    concept_id, _ = selected
    return [{"slug": "execute-" + concept_id.removeprefix("concept:"),
             "migration": False, "authorizes": [],
             "required_concept_ids": [concept_id]}]


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
        # The prompt necessarily names code-owned required slots so their ADR
        # bodies can be authored later. Small models sometimes parrot those
        # names as optional proposals. They add no decision and are safely
        # removed by identity; this is normalization, not a model retry.
        model_config._console_line(
            f"     - boundary=adr-slug-collision compared_optional={slugs} reserved_pipeline={taken_slugs} "
            f"intersection={clash} threshold=exact-identity decision=remove-echoed count={len(clash)}")
        slots = [slot for slot in slots if slot["slug"] not in set(clash)]
    return slots


def _merge_adr_slots(required: list[dict], optional: list[dict]) -> list[dict]:
    """Required migration slots first, then the model's optional slots; code
    assigns ADR ids in order (ADR-0020)."""
    slots = [dict(s) for s in required] + [dict(s) for s in optional]
    for i, s in enumerate(slots, start=1):
        s["decision_id"] = f"ADR-{i}"
    return slots


def _adr_subject(title: str) -> str:
    t = _re.sub(r"^(adopt|use|choose|select|migrate( to)?|switch to)\s+", "", title.strip().lower())
    return _re.sub(r"[^a-z0-9]+", " ", t).strip()


ADVISORY_TOKEN_JACCARD_THRESHOLD = 0.80


def _same_decision_tech(b1: dict, b2: dict) -> str | None:
    """Advisory only: registry identity first, normalized-token Jaccard >= .80."""
    r1 = primary_row_in_text(b1["title"])
    r2 = primary_row_in_text(b2["title"])
    if r1 and r2:
        return (f"same technology id: {r1['technology_id']}") if r1["technology_id"] == r2["technology_id"] else None
    if r1 or r2:
        return None
    t1, t2 = set(_adr_subject(b1["title"]).split()), set(_adr_subject(b2["title"]).split())
    if not t1 or not t2:
        return None
    score = len(t1 & t2) / len(t1 | t2)
    if score >= ADVISORY_TOKEN_JACCARD_THRESHOLD:
        return f"normalized-token Jaccard {score:.2f} >= {ADVISORY_TOKEN_JACCARD_THRESHOLD:.2f}"
    return None


def _dedupe_adr_slots(slots: list[dict], bodies: dict) -> list[dict]:
    """Return advisory duplicate candidates without changing decisions.

    Similarity is evidence for review, never authority to remove or rewrite an
    architecture decision. Migration decisions are excluded because each owns
    a distinct pipeline-computed departure.
    """
    findings=[]
    optional=[sl for sl in slots if not sl["migration"]]
    for i,left in enumerate(optional):
        for right in optional[i+1:]:
            # The fixed slot slug is the decision subject. Model-authored titles
            # may be accidentally repeated across distinct slots and cannot be
            # allowed to manufacture duplicate advisories.
            reason=_same_decision_tech({"title":left["slug"],"decision":""},
                                       {"title":right["slug"],"decision":""})
            if reason:
                findings.append({"kind":"possible_duplicate_adr","left_id":left["decision_id"],
                                 "left_slug":left["slug"],"right_id":right["decision_id"],
                                 "right_slug":right["slug"],"reason":reason,
                                 "action":"Review at the architecture gate; merge only if they express the same decision."})
    return findings


# Concept of Operations execution gate. The founder's concept-level decision
# outranks technology preferences: architecture implements it, never re-decides it.
def _check_conops_execution(arch: ArchitectureOverview, conops: dict | None) -> None:
    if not conops:
        return
    from vmodel_contracts import canonical_concept_id
    decision = conops.get("decision") or {}
    selected_id = str(decision.get("selected_concept_id") or canonical_concept_id(str(decision.get("selected_concept", "")))).strip()
    if not selected_id or selected_id == "concept:":
        raise ModelStructureError("CONOPS has no selected_concept_id")
    executed = {cid for d in arch.decisions for cid in d.executes_concept_ids}
    if selected_id not in executed:
        raise ModelStructureError(
            f"architecture does not execute founder-confirmed CONOPS concept id {selected_id!r}; "
            "an ADR must declare it in executes_concept_ids")


# Open-question resolution (2026-09-13 greenfield defect): the SRS carried
# 'Specific data storage solution for persisting user points' as OPEN after
# the architecture had decided PostgreSQL - a stale contradiction across
# artifacts. A question is answered when it (a) asks for a choice
# ('specific', 'which', 'what', 'choose') about (b) a technology topic the
# stack now owns. Conservative by design: feature questions ('which UI
# elements...') stay open - wrongly closing a real question is worse.
_QUESTION_TOPICS = {
    ("storage", "database", "data store", "persist"): ("database", "persistence", "storage"),
    ("framework",): ("framework",),
    ("library",): ("library", "visualization", "orm"),
    ("chart", "graph", "visualization"): ("visualization", "chart"),
    ("push", "notification"): ("notification", "push"),
    ("queue",): ("queue",),
    ("hosting", "provider"): ("hosting", "provider", "backend"),
    ("ci", "deployment"): ("ci", "deploy"),
}
_QUESTION_CHOICE_MARKERS = ("specific", "which", "what", "choose", "choice of")


def _resolve_answered_open_questions(state: dict, persist, *, architecture: dict | None = None) -> list[str]:
    """Drop architecture-answered open questions from the SRS artifact and
    say which stack entry answered each. Returns the resolution lines."""
    srs = state.get("srs") or {}
    # open_questions is code-owned by the intent model, so the LLM-authored
    # SRS state normally does not contain it. The old implementation looked
    # only in srs and therefore became a no-op in production even though its
    # unit fixture (incorrectly) put open_questions there.
    intent_questions = (state.get("intent_model") or {}).get("open_questions")
    questions = list(intent_questions if intent_questions is not None
                     else srs.get("open_questions") or [])
    already_resolved = set(srs.get("_resolved_open_questions") or [])
    questions = [q for q in questions if q not in already_resolved]
    if not questions:
        return []
    # During node_architecture, `data` is the newly assembled architecture but
    # LangGraph has not merged the node return into state yet. A resume from
    # architecture has deliberately cleared state["architecture"], so reading
    # state alone makes resolution a production-only no-op. The caller passes
    # the just-assembled artifact explicitly.
    entries = (architecture if architecture is not None
               else state.get("architecture") or {}).get("stack_entries") or []
    if not entries:
        return []
    keys = [str(e.get("key", "")).lower() for e in entries]
    resolved, kept = [], []
    for q in questions:
        ql = str(q).lower()
        answered_by = None
        if any(m in ql for m in _QUESTION_CHOICE_MARKERS):
            for topics, markers in _QUESTION_TOPICS.items():
                if any(t in ql for t in topics):
                    hit = next((e for e, k in zip(entries, keys)
                                if any(m in k for m in markers)), None)
                    if hit:
                        answered_by = hit
                        break
        if answered_by:
            resolved.append(f"'{q}' -> {answered_by['key']} = {answered_by['value']}")
        else:
            kept.append(q)
    if resolved:
        if intent_questions is None:  # backwards-compatible direct helper use
            srs["open_questions"] = kept
        srs["_resolved_open_questions"] = sorted(already_resolved | {
            q for q in questions if q not in kept
        })
        for line in resolved:
            model_config._console_line(f"     - open question resolved by architecture: {line}")
        if persist:
            persist("srs", srs, state)
    return resolved


def _assemble_architecture(*, session_id: str, baseline_id: str, entries: dict,
                           comp_slots: list, comp_bodies: dict, adr_slots: list,
                           adr_bodies: dict, nfr: dict) -> dict:
    """Code-owned projection: ADR selections are canonical; stack is derived."""
    decisions = []
    stack_entries = []
    stack_entry_dispositions = [dict(e) for e in entries.values() if e.get("status")]
    for slot in adr_slots:
        did = slot["decision_id"]
        body = {**adr_bodies[did]}
        selections = []
        for target in slot["authorizes"]:
            if not target.startswith("stack:"):
                continue
            key = target.split(":", 1)[1]
            e = entries[key]
            if e.get("status"):
                continue
            selection = {"slot_id": "slot:" + __import__("re").sub(r"[^a-z0-9]+", "-", key.casefold()).strip("-"),
                         "technology_id": e["technology_id"], "display_name": e["value"],
                         "baseline_refs": e["baseline_refs"],
                         "off_list_justification": e.get("off_list_justification")}
            selections.append(selection)
            stack_entries.append({"key": key, "value": e["value"], "technology_id": e["technology_id"],
                                  "decision_id": did, "baseline_refs": e["baseline_refs"],
                                  "migration_adr_id": did if slot["migration"] and not e["baseline_refs"] else None})
        body["technology_selections"] = selections
        decisions.append({"decision_id": did, **body})
    def covering(target):
        return next((s["decision_id"] for s in adr_slots if s["migration"] and target in s["authorizes"]), None)
    components=[]
    for c in comp_slots:
        aid=covering(f"component:{c['name']}") if c.get("migration") else None
        components.append({"name":c["name"],"responsibility":comp_bodies[c["name"]]["responsibility"],
                           "requirement_ids":c["requirement_ids"],"baseline_refs":c["baseline_refs"],
                           "migration":bool(c.get("migration")),"migration_adr_title":adr_bodies[aid]["title"] if aid else None})
    return {"session_id":session_id,"baseline_id":baseline_id,"stack_entries":stack_entries,
            "stack_entry_dispositions": stack_entry_dispositions,
            "components":components,"decisions":decisions,"non_functionals":nfr}



def _heal_slugs(entries: list) -> list:
    """Normalize slug shape and proven item echoes without inventing content.

    Exact duplicate records collapse first. Slugs are assigned per entry, not
    through a last-write-wins raw-slug map. Ambiguous dependency references
    remain errors rather than silently selecting one of several items.
    """
    import re as _re
    from collections import Counter, defaultdict

    def heal(value: str) -> str:
        return _re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")

    unique, seen = [], set()
    for entry in entries:
        if isinstance(entry, dict):
            identity = json.dumps(entry, sort_keys=True, separators=(",", ":"))
            if identity in seen:
                model_config._console_line("     - item echo removed: exact duplicate record")
                continue
            seen.add(identity)
        unique.append(entry)
    counts = Counter(entry.get("slug") for entry in unique
                     if isinstance(entry, dict) and isinstance(entry.get("slug"), str))
    stage_names = {"needs", "srs", "architecture", "items", "tests", "review", "assemble"}
    used, mapping, out = set(), defaultdict(list), []
    for entry in unique:
        if not isinstance(entry, dict):
            out.append(entry)
            continue
        entry = dict(entry)
        raw = entry.get("slug")
        missing = raw is None or (isinstance(raw, str) and not raw.strip())
        echoed = isinstance(raw, str) and (counts[raw] > 1 or heal(raw) in stage_names)
        if missing or echoed:
            title = entry.get("title")
            candidate = heal(title) if isinstance(title, str) else ""
        else:
            candidate = heal(raw) if isinstance(raw, str) else ""
        if candidate:
            base, suffix = candidate, 2
            while candidate in used:
                candidate, suffix = f"{base}-{suffix}", suffix + 1
            used.add(candidate)
            entry["slug"] = candidate
            if isinstance(raw, str):
                mapping[raw].append(candidate)
            if candidate != raw:
                model_config._console_line(f"     - slug healed: {raw!r} -> {candidate!r}")
        out.append(entry)
    for entry in out:
        if not isinstance(entry, dict) or not isinstance(entry.get("depends_on"), list):
            continue
        deps = []
        for dependency in entry["depends_on"]:
            choices = mapping.get(dependency, []) if isinstance(dependency, str) else []
            if len(choices) > 1:
                raise ModelStructureError(f"ambiguous dependency slug {dependency!r}: identifies multiple item specs")
            deps.append(choices[0] if choices else dependency)
        entry["depends_on"] = deps
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


def _previous_srs_fill(requirement: dict) -> dict:
    """Convert a contract-shaped requirement to the prose-only fill grammar."""
    from vmodel_contracts.normalize import normalize_list_string_fields
    acceptance = normalize_list_string_fields(
        {"acceptance": requirement.get("acceptance") or []},
        type("SRSFillBoundary", (), {"model_fields": {
            "acceptance": type("F", (), {"annotation": list[str]})()
        }})
    )["acceptance"]
    return {
        "title": requirement.get("title"),
        "statement": requirement.get("statement"),
        "priority": requirement.get("priority"),
        "acceptance": acceptance,
    }


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
        if any(not isinstance(x, str) or not x.strip() for x in f["acceptance"]):
            raise ModelStructureError(
                f"{rid}: acceptance items must be non-empty strings, not criterion objects; "
                "criterion ids are assigned by the pipeline")
        if not str(f.get("title") or "").strip() or not str(f.get("statement") or "").strip():
            raise ModelStructureError(f"{rid}: title and statement must be non-empty")
        return f  # normalized (unwrapped) fill
    return _v


def _items_plan_payload(state: GenState, req_ids: list[str]) -> str:
    return (STAGE_PROMPTS["items_plan"] + "\n\nSRS:\n" + json.dumps(_prompt_view(state["srs"]))
                   + "\n\nFOUNDER REGISTER (decided answers are binding; open/deferred inputs are not defaults):\n"
                   + json.dumps(_prompt_view(((state.get("intent_model") or {}).get("founder_register") or {})))
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


def _unquarantined_items(state: GenState) -> list[dict]:
    """Nothing derives from content with an OPEN finding (JEV-D10)."""
    held = Ledger.from_dict(state.get("findings_ledger")).quarantined()
    return [i for i in state["items"] if i.get("spec_id") not in held]

_CONTENT_RULES = {"echo": "echo.directive", "duplicate": "duplicate.body"}


def _accept_content(spec_id: str, body: str, siblings: dict[str, str], ledger: Ledger, sink=None) -> list:
    """The ONE content acceptance boundary (JEV-D10 rules 1 and 2). Every authoring
    path - bulk, per-spec, surgical repair, resumed partials - calls it before an
    item can be saved as accepted. Reject stops the stage loudly (inconclusive results are GRAY, never a stop) and
    the candidate is kept as evidence. Gray results become OPEN findings."""
    results = _ca.check_echo(body) + _ca.check_duplicate(spec_id, body, siblings)
    blocked = [r.to_dict() for r in results if r.status in (_ca.REJECT,)]
    if blocked:
        if sink:
            sink({"spec_id": spec_id, "body": body, "body_revision": revision_of(body), "results": blocked})
        raise ContentRejected(spec_id, blocked)
    revision = revision_of(body)
    fired = {}
    for r in results:
        if r.status == _ca.GRAY:
            fired[_CONTENT_RULES[r.rule]] = finding_from_check(ledger, r.to_dict(), spec_id, body)
    # an earlier finding on a re-authored body: closed only if its predicate no longer fires
    for f in list(ledger.open_findings()):
        if f["targets"] == [spec_id] and f["rule_id"] in _CONTENT_RULES.values() and f["artifact_revision"] != revision:
            if f["rule_id"] in fired:
                ledger.supersede_open(f["finding_id"], fired[f["rule_id"]])
            else:
                ledger.address(f["finding_id"], before_revision=f["artifact_revision"], after_revision=revision,
                               predicate_id=f["rule_id"], predicate_version=_ca.POLICY_VERSION,
                               predicate_still_fires=False)
    return results


def _accept_all(items: list[dict], ledger: Ledger, sink=None, only: set[str] | None = None) -> None:
    """Accept bodies in order against already accepted siblings (all specs, so a
    later duplicate of an earlier body is caught whatever the authoring path)."""
    seen: dict[str, str] = {}
    for spec in items:
        sid, body = spec.get("spec_id"), spec.get("body_markdown") or ""
        if only is None or sid in only:
            others = {k: v for k, v in {**{i.get("spec_id"): i.get("body_markdown") or "" for i in items}}.items()
                      if k != sid} if only is not None else dict(seen)
            _accept_content(sid, body, others, ledger, sink)
        seen[sid] = body
    ledger.refresh({i.get("spec_id"): revision_of(i.get("body_markdown") or "") for i in items})


def _content_sink(persist):
    fn = getattr(persist, "content_evidence", None)
    return fn if callable(fn) else None


def _items_payload(state: GenState, req_ids: list[str]) -> str:
    """The items prompt. The FORMAT CONTRACT closes it (the 14b follows the
    last frame it read): the reference enum, the coverage rule, and the
    id-discipline frame are the final words."""
    return (STAGE_PROMPTS["items"] + "\n\nSRS:\n" + json.dumps(_prompt_view(state["srs"]))
                   + "\n\nFOUNDER REGISTER (decided answers are binding; open/deferred inputs are not defaults):\n"
                   + json.dumps(_prompt_view(((state.get("intent_model") or {}).get("founder_register") or {})))
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


REVIEW_MAX_FINDINGS = REVIEW_CHUNK_MAX_FINDINGS
REVIEW_MAX_TARGET_IDS = 8
REVIEW_MAX_FIX_CHARS = 320


_REMOVAL_INTENT = _re.compile(
    r"\b(remove|replace|avoid|drop|eliminate|omit|delete|abstract|generali[sz]e|(?:do not|don't|should not|must not|never) "
    r"(?:name|mention|reference|specify|prescribe|use|include)|without (?:naming|mentioning|referencing)|instead of|"
    r"solution[- ]agnostic|implementation[- ]agnostic|technology[- ]neutral|vendor[- ]neutral)\b", _re.I)


def _decided_technologies(state: GenState) -> list[str]:
    """Names the architecture decided: stack values and the technologies its ADRs selected."""
    arch = state.get("architecture") or {}
    names: list[str] = []
    for e in arch.get("stack_entries") or []:
        v = str((e or {}).get("value") or "").strip()
        if len(v) >= 3:
            names.append(v)
    for d in arch.get("decisions") or []:
        for sel in (d or {}).get("technology_selections") or []:
            v = str((sel or {}).get("display_name") or "").strip()
            if len(v) >= 3:
                names.append(v)
    return list(dict.fromkeys(names))


def _decided_architecture_block(state: GenState) -> str:
    arch = state.get("architecture") or {}
    stack = {str((e or {}).get("key")): (e or {}).get("value") for e in arch.get("stack_entries") or []}
    titles = [str((d or {}).get("title") or "") for d in arch.get("decisions") or []]
    return ("\n\nDECIDED ARCHITECTURE (binding - already decided by the architecture stage; specs and tests MUST name these, "
            "never flag them):\n" + json.dumps({"stack": stack, "decisions": titles, "technologies": _decided_technologies(state)}))


def _filter_decided_stack_findings(fixes: list[dict], state: GenState) -> tuple[list[dict], list[dict]]:
    """Deterministic backstop for the review rubric: a spec/test finding that asks to remove, replace or
    avoid a technology the architecture DECIDED is not a defect. It is recorded (auditable), not opened."""
    techs = _decided_technologies(state)

    def _asks_to_drop(tech: str, text: str) -> bool:
        # the removal verb must govern the technology (within 60 chars before it), so "uses Firebase but the
        # architecture decided GitHub Pages; remove the contradiction" stays a real finding
        for m in _re.finditer(rf"(?<![\w+.#]){_re.escape(tech.lower())}(?![\w])", text.lower()):
            if _REMOVAL_INTENT.search(text[max(0, m.start() - 60):m.start()]):
                return True
        return False

    kept, filtered = [], []
    for fix in fixes:
        text = fix.get("fix") or ""
        hit = (next((t for t in techs if _asks_to_drop(t, text)), None)
               if fix.get("scope") in ("spec", "test") else None)
        if hit:
            filtered.append({**fix, "filtered": "filtered: decided-stack", "matched_technology": hit})
        else:
            kept.append(fix)
    return kept, filtered


# Review-noise classes (measured on the 2026-10-02 ledger, 96 spec/test findings the founder rejected, 0 of them
# kept by these rules; requirement and systemic findings are never touched). Each class is a deterministic phrase
# pattern; a finding that states a contradiction, an absence or an orphan is rescued and stays open.
_NOISE_RESCUE = _re.compile(
    r"contradict|conflict|inconsisten|mismatch|does not match|doesn't match|\bbut the (architecture|srs|requirement)|"
    r"\binstead of\b|\b(missing|omits?|omitted|absent|lacks?|no mention|not mentioned|never (states|mentions))\b|"
    r"\borphan|untraced|unreferenced|restat\w+ (the )?requirement", _re.I)
_NOISE_CLASSES = (
    ("A:anti-technology-doctrine", _re.compile(
        r"solution[- ]agnostic|technology[- ]agnostic|"
        r"(do not|don't|not|avoid|without)\s+(\w+\s+){0,2}(imply\w*|prescrib\w*|specif\w*|nam\w+|referenc\w*|mention\w*)"
        r"\s+(\w+\s+){0,3}(specific |particular )?(technolog|implementation|design solution|frameworks?\b)", _re.I)),
    ("E:solution-detail-in-spec-or-test", _re.compile(
        r"(does not|do not|not)\s+specify\b.{0,80}(UI elements|UI interactions|selectors?|solution details?|"
        r"performance metrics|GitHub)|"
        r"(remove|avoid)\b.{0,60}\b(solution[- ]specific|solution detail|solution language|implies? (a )?(solution|implementation)|"
        r"prescribes? a design)", _re.I)),
    ("F:cites-decided-architecture", _re.compile(
        r"(as per|aligning with|consistent with|per) the decided (architecture|storage|programming language|stack)", _re.I)),
)


def _noise_class(fix: dict) -> str | None:
    if fix.get("scope") not in ("spec", "test"):
        return None
    text = fix.get("fix") or ""
    if _NOISE_RESCUE.search(text):
        return None
    return next((name for name, pat in _NOISE_CLASSES if pat.search(text)), None)


def _filter_review_noise(fixes: list[dict]) -> tuple[list[dict], list[dict]]:
    """Archive (never open) spec/test findings in the measured noise classes. Auditable: the caller records them."""
    kept, filtered = [], []
    for fix in fixes:
        cls = _noise_class(fix)
        if cls:
            filtered.append({**fix, "filtered": f"filtered: noise-class {cls}", "noise_class": cls})
        else:
            kept.append(fix)
    return kept, filtered


def _normalized_review_fixes(value) -> list[dict]:
    out = []
    for fix in (value or []):  # no slice: every finding survives; per-chunk bounds are checked in node_review
        if isinstance(fix, str):
            out.append({"scope": "systemic", "target_ids": [], "fix": fix})
        elif isinstance(fix, dict) and str(fix.get("fix") or "").strip():
            scope = fix.get("scope") if fix.get("scope") in ("requirement", "spec", "test", "systemic") else "systemic"
            ids = [str(x) for x in (fix.get("target_ids") or []) if str(x)][:REVIEW_MAX_TARGET_IDS]
            if scope != "systemic" and not ids:
                scope = "systemic"
            fix_text = str(fix["fix"]).strip()
            if len(fix_text) > REVIEW_MAX_FIX_CHARS:
                fix_text = fix_text[:REVIEW_MAX_FIX_CHARS - 3].rstrip() + "..."
            out.append({"scope": scope, "target_ids": ids, "fix": fix_text})
    return out


def _target_revision(state: GenState, fix: dict) -> str:
    """Revision of exactly what a review finding points at (cleaned text hash)."""
    ids = set(fix.get("target_ids") or [])
    scope = fix.get("scope")
    if scope == "spec":
        text = "\n".join(i.get("body_markdown") or "" for i in state.get("items") or [] if i.get("spec_id") in ids)
    elif scope == "requirement":
        text = json.dumps([r for sec in ("functional", "non_functional") for r in (state.get("srs") or {}).get(sec) or []
                           if r.get("requirement_id") in ids], sort_keys=True)
    elif scope == "test":
        text = json.dumps([t for t in state.get("tests") or [] if t.get("test_id") in ids], sort_keys=True)
    else:
        text = json.dumps([state.get("srs"), state.get("items")], sort_keys=True, default=str)
    return revision_of(text)


def _ledger_review_findings(ledger: Ledger, state: GenState, fixes: list[dict], round_n: int,
                            raised_out: set | None = None) -> list[str]:
    """Every LLM review finding becomes ledger data. Dedup precedes any limit and the
    ledger is never sliced. These are semantic: they close only by founder disposition."""
    raised = set()
    for fix in fixes:
        raised.add(ledger.add(rule_id="review.llm", rule_version=f"review-round-{round_n}", source=f"review:round-{round_n}",
                   kind="semantic", scope=fix["scope"], targets=fix["target_ids"] or ["systemic"],
                   artifact_revision=_target_revision(state, fix), summary=fix["fix"],
                   evidence=[{"id": "review:finding", "sha256": revision_of(fix["fix"]), "excerpt": fix["fix"][:320]}],
                   detail={"scope": fix["scope"], "target_ids": fix["target_ids"], "fix": fix["fix"]}))
    if raised_out is not None:
        raised_out.update(raised)
    # a re-run of this round replaces its earlier output (not the founder's decisions, not other rounds)
    return ledger.retire_review_round(f"review:round-{round_n}", raised)


def _review_scope_requirements(state: GenState, fixes: list[dict]) -> list[str] | None:
    """After a scoped repair, the requirements whose review chunk must be re-run: the repaired requirements plus the
    requirements linked to repaired specs and tests. None = a systemic or unresolvable repair, review everything."""
    fixes = _normalized_review_fixes(fixes)
    if not fixes or any(f["scope"] == "systemic" for f in fixes):
        return None
    reqs = {x for f in fixes if f["scope"] == "requirement" for x in f["target_ids"]}
    spec_ids = {x for f in fixes if f["scope"] == "spec" for x in f["target_ids"]}
    test_ids = {x for f in fixes if f["scope"] == "test" for x in f["target_ids"]}
    for spec in state.get("items") or []:
        if spec.get("spec_id") in spec_ids:
            reqs |= set(spec.get("requirement_ids") or [])
    if test_ids:
        crit = {c.get("criterion_id"): r.get("requirement_id") for sec in ("functional", "non_functional")
                for r in (state.get("srs") or {}).get(sec) or [] for c in r.get("acceptance") or []}
        spec_req = {i.get("spec_id"): i.get("requirement_ids") or [] for i in state.get("items") or []}
        for t in state.get("tests") or []:
            if t.get("test_id") in test_ids:
                if t.get("criterion_id") in crit:
                    reqs.add(crit[t["criterion_id"]])
                reqs |= set(spec_req.get(t.get("spec_id"), []))
    return sorted(reqs) or None


def _repair_proposal(state: GenState, data: dict, fixes: list[dict], finding_ids: list[str]) -> dict:
    """Before/after of exactly the requirements the repair touched. Nothing is applied: the founder reads this,
    then --approve-repair applies it or --reject-repair drops it. `unchanged_targets` are repair targets whose text
    came back identical (the model changed nothing, or a reviewed-SRS decision restored the accepted text)."""
    def by_id(srs):
        return {r.get("requirement_id"): r for sec in ("functional", "non_functional")
                for r in (srs or {}).get(sec) or []}
    old, new = by_id(state.get("srs")), by_id(data)
    changed = {rid: {"before": old.get(rid), "after": new.get(rid)} for rid in sorted(new) if old.get(rid) != new.get(rid)}
    targets = _fix_targets(fixes, "requirement")
    unchanged = sorted(t for t in (targets or set()) if t in new and t not in changed)
    return {"finding_ids": list(finding_ids), "changed": changed, "unchanged_targets": unchanged, "srs": data}


def render_repair_proposal(proposal: dict) -> str:
    lines = ["# Repair proposal (not applied)", "",
             f"Findings: {', '.join(proposal.get('finding_ids') or [])}", ""]
    for rid, pair in (proposal.get("changed") or {}).items():
        for label in ("before", "after"):
            req = pair.get(label) or {}
            lines.append(f"## {rid} {label}")
            lines.append(str(req.get("statement") or "(none)"))
            for c in req.get("acceptance") or []:
                lines.append(f"- {c.get('criterion_id')}: {c.get('statement')}")
            lines.append("")
    if proposal.get("unchanged_targets"):
        lines += ["## Targets that came back unchanged",
                  ", ".join(proposal["unchanged_targets"]),
                  "(the model changed nothing, or a reviewed-SRS decision restored the accepted text)", ""]
    lines += ["Apply with --approve-repair, drop with --reject-repair."]
    return "\n".join(lines) + "\n"


def _check_ac_ids_stable(before: dict, after: dict) -> None:
    """Acceptance-criterion ids are numbered sequentially across the whole SRS. A repair that adds or removes a
    criterion shifts every later AC id, and tests are linked by criterion_id: they would silently point at the wrong
    criterion. Refuse such a repair before anything is persisted."""
    from .errors import UserFacingError

    def ids(srs):
        return {r.get("requirement_id"): [c.get("criterion_id") for c in r.get("acceptance") or []]
                for sec in ("functional", "non_functional") for r in (srs or {}).get(sec) or []}
    old, new = ids(before), ids(after)
    changed = sorted(r for r in set(old) | set(new) if old.get(r) != new.get(r))
    if changed:
        detail = "; ".join(f"{r}: {len(old.get(r) or [])} -> {len(new.get(r) or [])} criteria "
                           f"({(old.get(r) or ['-'])[0]}.. vs {(new.get(r) or ['-'])[0]}..)" for r in changed[:6])
        raise UserFacingError(
            "scoped repair refused: it would renumber acceptance criteria.",
            f"checked the criterion ids of every requirement before and after the repair; found changes in "
            f"{', '.join(changed)} ({detail}). Tests link by criterion id, so a shift would mis-link them. Nothing was applied.",
            "rerun the repair; keep the number of acceptance criteria per requirement unchanged, or rerun from 'srs' "
            "to regenerate tests with the new numbering.")


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


def _stage_validation_error(stage: str, exc: ValidationError):
    """Translate an escaped artifact-contract failure at the stage boundary."""
    from .errors import UserFacingError
    shape = "; ".join(
        f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']} "
        f"(got {type(err.get('input')).__name__})"
        for err in exc.errors()[:8])
    return UserFacingError(
        f"stage '{stage}' produced an artifact that fails its contract.",
        f"artifact '{stage}' has an offending field shape: {shape}",
        "rerun the same command after updating spec-generator; the checkpoint "
        "preserves all completed upstream stages and changing models cannot fix "
        "a pipeline-owned shape conversion.",
    )


def _raise_assembly_boundary_error(stage: str, state: dict, exc: Exception):
    from .errors import UserFacingError
    if isinstance(exc, KeyError):
        offending = repr(exc.args[0]) if exc.args else "unknown"
        expected = "registered pipeline stage/key and contract-owned field names"
    elif isinstance(exc, TypeError):
        offending = f"message={str(exc)}"
        expected = "the stage contract's declared JSON types"
    else:
        offending = str(exc)
        expected = "the deterministic stage validator requirements named in the message"
    state_shape = {key: type(value).__name__ for key, value in state.items()}
    model_config._console_line(
        f"  ERROR boundary=assembly-rejection producer_stage={stage} exception={type(exc).__name__} "
        f"offending={offending!r} expected={expected!r} input_state_shape={state_shape} "
        f"decision=reject-and-checkpoint reason={str(exc)!r}")
    return UserFacingError(
        f"stage '{stage}' could not assemble the model output.",
        f"pipeline boundary rejected an unexpected field shape: {type(exc).__name__}: {str(exc)[:500]}. "
        f"Offending value/key: {offending}. Expected: {expected}. Producer: {stage}. "
        f"Input state shape: {state_shape}.",
        f"{model_config._rerun_guidance(stage)}; the failed stage will retry from its checkpoint. "
        "The boundary evidence above is also in run.log.")


def _ground_items(state, ledger, provider, persist) -> None:
    """JEV-D10 rule 3 at the review boundary. Jev only; no provider means NOT CHECKED (recorded, never a pass)."""
    from . import grounding as G
    reqs = {r.get("requirement_id"): r for sec in ("functional", "non_functional")
            for r in (state.get("srs") or {}).get(sec) or []}
    excl = G.scope_exclusions(state.get("intent_model"))   # the SRS document has no "exclusions" field
    needs = ((state.get("needs") or {}).get("needs")) if isinstance(state.get("needs"), dict) else None
    for item in state.get("items") or []:
        bound = [reqs[r] for r in item.get("requirement_ids") or [] if r in reqs]
        if not bound:
            continue
        env = G.ground_spec(item["spec_id"], item.get("body_markdown") or "", bound, provider, exclusions=excl, needs=needs)
        G.record_grounding(ledger, env)
        getattr(persist, "grounding", lambda *_: None)(item["spec_id"], env)
    if provider is not None:
        body = {i["spec_id"]: i.get("body_markdown") or "" for i in state.get("items") or []}
        for f in ledger.open_findings():
            if f["rule_id"] == "echo.directive" and f["targets"] and f["targets"][0] in body \
                    and (f.get("detail") or {}).get("check", {}).get("status") == "gray":
                f["detail"]["jev_gray_band"] = G.ask_gray_echo(provider, f, body[f["targets"][0]])


def build_graph(log: UsageLog, checkpointer=None, progress: ProgressBus | None = None,
                review: bool = False, persist=None, partials=None,
                only_stage: str | None = None, stop_after: str | None = None,
                expert_opinion: bool = False, grounding=None):
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
                    try:
                        return fn(state)
                    except ValidationError as exc:
                        raise _stage_validation_error(stage, exc) from exc
                    except (ModelStructureError, KeyError, TypeError) as exc:
                        raise _raise_assembly_boundary_error(stage, state, exc) from exc
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
        # Previous SRS stores contract-shaped acceptance criteria
        # ({criterion_id, statement}); the fill grammar and merge function own
        # only prose ([statement]). Normalize at this boundary before an
        # apply_fixes pass reuses unaffected or checkpointed fills.
        done = {r["requirement_id"]: _previous_srs_fill(r)
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
        # A review-driven SRS rewrite happens after architecture. Preserve the
        # code-owned record of interview questions architecture already
        # answered, or deterministic assembly would resurrect them.
        if (state.get("srs") or {}).get("_resolved_open_questions"):
            merged["_resolved_open_questions"] = list(
                state["srs"]["_resolved_open_questions"])
        sid = state["intent_model"]["session_id"]
        SRSDocument.model_validate(srs_document_data(
            state["intent_model"], {**merged, "session_id": sid, "intent_model_ref": sid}))
        if partials:
            partials.clear("srs_fills")
        return merged

    def node_needs(state: GenState) -> dict:
        intent = IntentModel.model_validate(state["intent_model"])
        # The approved IntentModel already has every legitimate stakeholder
        # need. Re-authoring it added ungrounded "implied" N rows and duplicate
        # SRS slots in production. Carry the relation 1:1 in code instead.
        from .skeleton import needs_from_intent
        data = needs_from_intent(intent.model_dump(mode="json"))
        StakeholderNeedsSet.model_validate(data)
        model_config._console_line(
            f"   needs guard: carried {len(data['needs'])} approved SN need(s) 1:1; "
            "model-added implied needs disabled")
        capped = _experiment_cap("NEEDS", list(data.get("needs") or []), "needs", progress)
        if len(capped) != len(data.get("needs") or []):
            data["needs"] = capped
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
        from .skeleton import deferred_needs as _deferred_needs
        _def = _deferred_needs(state["needs"])
        if _def:
            model_config._console_line(
                f"   future-phase: {len(_def)} need(s) deferred in the interview "
                f"({', '.join(n.get('need_id', '?') for n in _def)}) - recorded in the SRS scope, "
                "driving no current requirements")
        data = _run_srs_fill_stage(state, feedback)
        data.setdefault("session_id", state["intent_model"]["session_id"])
        data.setdefault("intent_model_ref", state["intent_model"]["session_id"])
        # Retain original candidates before any semantic rewrite or deletion.
        # Schema validation still applies; accepting meaning is not permission
        # to bypass missing IDs/types or traceability.
        from vmodel_contracts.semantic_review import make_review
        # Every source of the founder's meaning: the interview, the needs derived from it, and the concept of operations.
        interview_corpus=json.dumps([state['intent_model'],state.get('needs'),state.get('conops')])
        lint=text_qa.srs_draft_findings(data,interview_corpus)
        findings=[]
        for section in ('functional','non_functional'):
            for index,req in enumerate(data.get(section) or []):
                rid=req.get('requirement_id','?')
                from .lint_reasons import describe,group_findings
                for n,group in enumerate(group_findings(lint,str(rid))):
                    # One finding per requirement field and span; every rule that fired is named in its reason.
                    # Review the complete requirement so IDs and acceptance
                    # relations are checked together after remove/replace.
                    findings.append({'finding_id':f'{rid}:{n}','item_id':str(rid),'path':[section,index],
                        'rule':'+'.join(dict.fromkeys(g['rule'] for g in group)),'reason':describe(str(rid),group),'original':req})
        review=make_review(state['intent_model']['session_id'],'srs',data,findings)
        reviews={**(state.get('semantic_reviews') or {}),'srs':review.model_dump(mode='json')}
        flags=[]
        revisions = dict(state.get("artifact_revisions") or {})
        prior_revision = revisions.get("srs") or {}
        revision = int(prior_revision.get("revision") or 0) + 1
        saved_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        revisions["srs"] = {"saved_at": saved_at, "revision": revision}
        if persist:
            persist("srs", data, {**state, "artifact_revisions": revisions,"semantic_reviews":reviews})
        return {"srs": data,"semantic_reviews":reviews, "guard_flags": state.get("guard_flags", []) + flags,
                "artifact_revisions": revisions,
                "gate_feedback": _clear_feedback(state, "srs")}

    def node_architecture(state: GenState) -> dict:
        pending=(state.get('semantic_reviews') or {}).get('srs')
        if pending and pending.get('findings'):
            from vmodel_contracts.semantic_review import SemanticReview
            from .semantic_review import reviewed_srs
            try:candidate,_=reviewed_srs(state['intent_model'],pending,pending)
            except ValueError as exc:
                from .errors import UserFacingError
                raise UserFacingError('SRS candidate review is not ready.',str(exc),'Review every flagged SRS item with Remove, Replace, or Accept before continuing.') from exc
            if candidate!=state['srs']:raise ValueError('reviewed SRS differs from the checkpoint')
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
                   + "\n\nFOUNDER REGISTER (decided answers are binding; open/deferred inputs are not defaults):\n"
                   + json.dumps(_prompt_view(((state.get("intent_model") or {}).get("founder_register") or {})))
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

        founder_techs = _founder_named_techs(state["intent_model"])
        product_platforms = _product_platforms(state["intent_model"])
        _announce_candidate_lists(founder_techs)
        if product_platforms & {"ios", "android"}:
            model_config._console_line(
                "   product platform context: " + ", ".join(sorted(product_platforms))
                + " - UI-kind slots require a matching-platform candidate (insistence "
                "lands its justification in an ADR for your veto)")

        # 1. Plan: which technology slots to decide.
        if "stack_keys" not in part:
            plan = _call_json("architecture", context + "\n\n" + ARCH_STACK_PLAN + feedback,
                              log, progress, validate=_validate_stack_plan,
                              schema=arch_stack_plan_schema(), stream_items=False,
                              item_label="plan: stack slots")
            part["stack_keys"] = _experiment_cap("STACK_SLOTS", plan["stack_keys"],
                                                 "stack slots", progress)
            _save()

        # 2. One micro-call per stack entry: value + grounding refs. The
        # value-vs-cited-facts check runs HERE (per-slot, retryable), not only
        # at assembly: an invented technology costs one retried entry call,
        # not the whole stage (the 2026-09-12 Auth0 acceptance failure -
        # 1282s of good calls lost to a gate that fired at the end).
        baseline_vocab = _baseline_vocab(baseline) if existing else ""

        def _v_entry(d, key):
            status = str(d.get("status") or "").strip()
            if status:
                if status == "research_required":
                    brief = str(d.get("research_brief") or "").strip()
                    if not brief:
                        raise ModelStructureError(f"stack.{key}: research_required requires a research_brief")
                    return {"key": key, "status": status, "research_brief": brief}
                if status == "not_applicable":
                    reason = str(d.get("reason") or "").strip()
                    if not reason:
                        raise ModelStructureError(f"stack.{key}: not_applicable requires a reason arguing genuine irrelevance")
                    return {"key": key, "status": status, "reason": reason}
                raise ModelStructureError(f"stack.{key}: unknown status {status!r}")
            value = str(d.get("technology", d.get("value", ""))).strip()
            refs = [str(r) for r in d.get("baseline_refs") or []]
            technology_id, technology_row = resolve_technology(value)
            raw_justification = d.get("off_list_justification")
            justification = ((str(raw_justification).strip() or None)
                             if raw_justification is not None else None)
            if not value:
                raise ModelStructureError(f"stack.{key}: an empty value decides nothing - propose a concrete technology; empty baseline_refs becomes a proposed migration")
            is_grounded_fact = bool(refs) and _value_grounded(value, baseline_vocab)
            if technology_row is None and not is_grounded_fact and not str(justification or "").strip():
                raise ModelStructureError(f"stack.{key}: off-list pick {value!r} requires off_list_justification")
            if technology_row is None and is_grounded_fact:
                technology_id = "tech:" + _re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
            if technology_row is not None and justification is not None:
                raise ModelStructureError(f"stack.{key}: off_list_justification is only for unresolved technology")
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
                    "this key, name a concrete technology anyway with baseline_refs []"
                    + ("; it becomes a proposed migration automatically and the pipeline "
                       "generates the authorizing ADR itself" if existing else
                       " - in greenfield mode that is simply the new design, no migration "
                       "framing"))
            entry = {"key": key, "value": value, "baseline_refs": refs}
            if existing and entry["baseline_refs"] and not _value_grounded(entry["value"], baseline_vocab):
                raise ModelStructureError(
                    f"stack.{key}: value {entry['value']!r} is not present in the cited baseline facts - "
                    "copy the fact text VERBATIM (if the fact is 'python -m spec_generator.cli', the "
                    "value is exactly that string), compose it only from cited fact text, or declare "
                    "a migration (empty baseline_refs)")
            if not existing and entry["baseline_refs"]:
                raise ModelStructureError(
                    f"stack.{key}: greenfield architecture cannot cite baseline refs")
            # P3 slot-kind fitness (2026-09-12 regen, founder-approved): a
            # grounded name must ANSWER the slot, not merely exist. Layer 1 is
            # deterministic: entry-point-shaped values are legal only for
            # entry-point slots.
            if _entry_point_shaped(value) and not _known_tech(value) and not _entry_point_slot(key):
                raise ModelStructureError(
                    f"stack.{key}: value {value!r} is an entry point (a command, script or "
                    "module:attr), which cannot answer this slot - name the technology of the "
                    "slot's KIND (a framework name for a framework slot, a CI tool for a ci "
                    "slot), or declare a migration (empty baseline_refs)")
            # Layer 2 is deterministic too (founder directive 2026-09-12): a
            # value in a KNOWN lexicon of the wrong kind cannot answer the slot
            # ('Pydantic' for encryption). Novel-tech values match no lexicon
            # and pass - a lexicon miss is never a verdict; no LLM judge call.
            wrong = _wrong_kind(key, value) if not _entry_point_slot(key) else None
            if wrong:
                raise ModelStructureError(
                    f"stack.{key}: {value!r} is a {wrong}, which does not answer this "
                    "slot - name a technology that directly answers this slot's question, "
                    "or declare a migration (empty baseline_refs)")
            # Gate 3 (2026-09-13 matrix decision, his React-for-mobile defect):
            # ATTRIBUTE fitness. A wrong-PLATFORM technology cannot silently
            # answer a UI-kind slot when the product context is mobile.
            # Deterministic reject with the matching-platform alternatives;
            # ONE insistence is allowed, and then the covering ADR must carry
            # the explicit platform justification for founder veto (checked
            # at assembly) - no silent pass, no soft hint. Baseline-grounded
            # values are facts about the repo, not proposals: not re-judged.
            kind3 = _candidate_kind(key)
            if (kind3 in PLATFORM_CHECKED_KINDS and not entry["baseline_refs"]
                    and not _founder_required_tech(state["intent_model"], key, value)
                    and product_platforms & {"ios", "android"}):
                row = tech_row(value)
                need = product_platforms & {"ios", "android"}
                if row and "any" not in row["platforms"] and not (set(row["platforms"]) & need):
                    conflicts = part.setdefault("platform_conflicts", {}).setdefault(key, [])
                    if value.lower() in (c.lower() for c in conflicts):
                        part.setdefault("contested", {})[key] = value
                        model_config._console_line(
                            f"     - contested: stack.{key}={value} ({'/'.join(row['platforms'])} technology, "
                            f"{'/'.join(sorted(need))} product) - an ADR must carry the explicit justification")
                    else:
                        conflicts.append(value)
                        candidate_rows = TECH_CANDIDATES.get(kind3, ())
                        if key == "ui_framework":
                            candidate_rows += TECH_CANDIDATES.get("mobile_framework", ())
                        fitting = list(dict.fromkeys(
                            n for n, _ in candidate_rows
                            if (r2 := tech_row(n))
                            and (set(r2["platforms"]) & need or "any" in r2["platforms"])
                            and (key != "ui_framework" or "backend" not in r2["kinds"])))
                        raise ModelStructureError(
                            f"stack.{key}: {value!r} is a {'/'.join(row['platforms'])} technology, but the "
                            f"product context is {'/'.join(sorted(need))} - it cannot answer this slot "
                            f"directly. Matching-platform candidates: {', '.join(fitting[:6]) or '(none on file)'}. "
                            f"If {value!r} is a DELIBERATE call (e.g. web-first PWA), repeat it with "
                            "off_list_justification: null (this is a registered technology; that field is only "
                            "for an unresolved technology). The covering ADR must name it in title/decision "
                            "and argue the platform trade-off in its rationale for founder veto.")
            entry["technology_id"] = technology_id
            entry["off_list_justification"] = justification
            return entry

        entries = part.setdefault("stack_entries", {})
        for key in list(entries):
            # persisted slots must satisfy the CURRENT contract; a stale entry
            # (e.g. Auth0 saved before this check existed) is dropped and
            # re-asked - never silently trusted, never a rerun from zero.
            try:
                entries[key] = _v_entry(entries[key], key)
            except ModelStructureError as e:
                del entries[key]
                model_config._console_line(
                    f"     - persisted stack entry {key!r} failed the current contract ({e}) - re-asking that slot")
        for key in part["stack_keys"]:
            if key in entries:
                continue  # per-slot resume
            entries[key] = _call_json(
                "architecture",
                context + f"\n\nTHE STACK KEY TO DECIDE: {key}\n"
                + _candidate_block(key, founder_techs) + "\n\n" + ARCH_STACK_ENTRY + feedback,
                log, progress, validate=lambda d, key=key: _v_entry(d, key),
                schema=arch_stack_entry_schema(valid_refs, [r["technology_id"] for r in TECH_ROWS]),
                stream_items=False, item_label=f"stack:{key}")
            _save()

        # 3. Plan: component slots (coverage checked here, not at assembly).
        if "component_slots" not in part:
            part["component_slots"] = _call_json(
                "architecture",
                context + "\n\nREQUIREMENT IDS TO COVER: " + json.dumps(req_ids)
                + "\n\n" + ARCH_COMPONENT_PLAN + feedback,
                log, progress, validate=lambda d: _validate_component_plan(d, req_ids, existing),
                schema=arch_component_plan_schema(req_ids, valid_refs),
                stream_items=False, item_label="plan: component slots")
            _save()

        # Persisted component slots must ALSO satisfy the current mode's
        # semantics on resume (his 2026-09-13 rerun: slots persisted with
        # migration=true under the old prompt failed the greenfield gate
        # immediately, with no re-ask). Greenfield: nothing exists to migrate
        # FROM - the flag is code-owned, normalized, never re-asked.
        if not existing:
            for slot in part.get("component_slots", []):
                slot["migration"] = False

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

        # Founder-candidate grounding can only be judged at assembly (which ADR
        # covers which area is the model's call), so the ADR plan+fill runs in a
        # bounded re-plan loop: ONE re-plan with the gate failure as feedback,
        # then a human-readable stage error (founder decision 2026-09-13).
        candidate_feedback = ""
        founder_plan_block = ("" if not founder_techs else
            "\n\nFOUNDER-NAMED TECHNOLOGIES (from the founder's stated preferences): "
            + ", ".join(_display(t) for t in founder_techs)
            + ". Your plan MUST leave room for the decision where each of these is adopted "
            + "or rejected - one slug per affected area.")
        won = {t for t in founder_techs
               if any(_tech_in_text(t, str(e.get("value") or e.get("technology") or ""))
                      for e in entries.values() if isinstance(e, dict))}
        lost = [t for t in founder_techs if t not in won]
        lost_block = ("" if not lost else
            "\n\nFOUNDER-NAMED TECHNOLOGIES NOT ADOPTED IN THE STACK: "
            + ", ".join(_display(t) for t in lost)
            + ". If THIS decision covers one of their areas, that technology MUST appear "
            + "in options_considered with the specific requirement it fails - vague reasons "
            + "('not suitable') are rejected and re-asked. If unrelated, ignore this line.")
        for _adr_attempt in (1, 2):
            # 5. Plan: ADR slots. Departures are CODE-COMPUTED from the accepted
            # entries/slots and the required migration slots are CODE-GENERATED
            # from them (the 2026-09-12 live phi4 failure: the model forgot to
            # authorize stack:ci_tool). The model only proposes OPTIONAL design
            # decisions; ids are assigned after the merge, required slots first.
            departures = [k for k, e in entries.items() if not e.get("status") and not e["baseline_refs"]] if existing else []
            migrating = [c["name"] for c in part["component_slots"] if c["migration"]] if existing else []
            authorizable = [f"stack:{k}" for k, e in entries.items() if not e.get("status")] + [f"component:{n}" for n in migrating]
            if "adr_slots" not in part:
                migrating_targets = {t for t in authorizable
                                     if (t.startswith("stack:") and t.removeprefix("stack:") in departures)
                                     or (t.startswith("component:") and t.removeprefix("component:") in migrating)}
                required = (_required_adr_slots(authorizable, migrating_targets)
                            + _required_concept_adr_slots(state.get("conops")))
                req_block = ("\n\nPIPELINE-REQUIRED ADRS (already planned - do not re-propose them; "
                             "these include migration obligations and the founder-confirmed CONOPS "
                             "execution obligation; author their bodies against these fixed slots):\n"
                             + json.dumps(required)) if required else \
                            "\n\nPIPELINE-REQUIRED ADRS: none."
                optional = _call_json(
                    "architecture",
                    context + "\n\nDECIDED STACK (key: value):\n"
                    + json.dumps({k: (e.get("value") or e.get("status")) for k, e in entries.items()})
                    + "\n\nCOMPONENT SLOTS:\n" + json.dumps(part["component_slots"])
                    + req_block + founder_plan_block + candidate_feedback + "\n\n" + ARCH_ADR_PLAN + feedback,
                    log, progress,
                    validate=lambda d: _validate_adr_plan(d, [s["slug"] for s in required]),
                    schema=arch_adr_plan_schema(),
                    stream_items=False, item_label="plan: decision slots")
                part["adr_slots"] = _merge_adr_slots(required, optional)
                _save()

            _normalize_adr_slots(part["adr_slots"], existing)  # resumed sessions: flag + label are code-owned
            _save()

            # 6. One micro-call per ADR: the decision substance.
            adr_bodies = part.setdefault("adr_bodies", {})
            for slot in part["adr_slots"]:
                did = slot["decision_id"]
                if did in adr_bodies:
                    continue
                def _v_adr(d, slot=slot):
                    raw_title = str(d.get("title", "")).strip()
                    # The renderer owns the visible ADR id. Models often echo the fixed
                    # id from the slot ("ADR-3: ..."), which used to render twice.
                    title = _re.sub(
                        rf"^\s*{_re.escape(slot['decision_id'])}\s*[:.\-]?\s*",
                        "", raw_title, flags=_re.IGNORECASE,
                    ).strip()
                    body = {"title": title,
                            "decision": str(d.get("decision", "")).strip(),
                            "rationale": str(d.get("rationale", "")).strip(),
                            "driver": d.get("driver"),
                            "options_considered": d.get("options_considered") or [],
                            "baseline_refs": [str(r) for r in d.get("baseline_refs") or []],
                            "migration": slot["migration"],
                            # code owns the migration flag: a non-migration ADR carries neither
                            # field, whatever the model wrote (never a retry over it)
                            "evidence": [str(x) for x in d.get("evidence") or []] if slot["migration"] else [],
                            "affected_repositories": ([str(x) for x in d.get("affected_repositories") or []]
                                                      if slot["migration"] else []),
                            "executes_concept_ids": [str(x) for x in d.get("executes_concept_ids") or []],
                            "technology_selections": [],
                            "rejected": []}
                    if not body["title"] or not body["decision"] or not body["rationale"]:
                        raise ModelStructureError(f"{slot['decision_id']}: title, decision and rationale are all required")
                    # phi4 often prefixes otherwise concrete prose with the fixed id
                    # in decision/rationale. The id adds no substance and code already
                    # owns it, so remove that mechanical echo locally. A field that
                    # becomes empty still fails below rather than fabricating content.
                    healed_self_refs = []
                    for name in ("decision", "rationale"):
                        before = body[name]
                        body[name] = _re.sub(
                            rf"\b{_re.escape(slot['decision_id'])}\b\s*(?:[:.,;\-]|(?:is|selects?|chooses?|decides?)\b)?\s*",
                            "", before, flags=_re.IGNORECASE).strip()
                        if body[name] != before:
                            healed_self_refs.append(name)
                    if healed_self_refs:
                        model_config._console_line(
                            f"     - boundary=deterministic-repair stage=architecture item={slot['decision_id']} "
                            f"repair=remove-self-reference fields={healed_self_refs} decision=continue")
                    if not body["decision"] or not body["rationale"]:
                        raise ModelStructureError(
                            f"{slot['decision_id']}: removing the self-reference left no concrete decision substance")
                    if any(not isinstance(opt, dict) or not str(opt.get("option", "")).strip()
                           or not str(opt.get("reason", "")).strip()
                           for opt in body["options_considered"]):
                        raise ModelStructureError(
                            f"{slot['decision_id']} '{body['title']}': options_considered must contain "
                            "objects {option: non-empty string, reason: non-empty string}; bare strings "
                            "cannot preserve why an alternative lost")
                    required_concepts = set(slot.get("required_concept_ids") or [])
                    missing_concepts = sorted(required_concepts - set(body["executes_concept_ids"]))
                    if missing_concepts:
                        raise ModelStructureError(
                            f"{slot['decision_id']} '{body['title']}': this code-owned CONOPS execution slot "
                            f"must declare these exact ids in executes_concept_ids: {missing_concepts}")
                    # A founder_constraint
                    # driver must cite its real source: either a technology
                    # preference or the confirmed CONOPS concept. The latter is a
                    # product/operations decision, not a technology preference.
                    adr_text = (body["title"] + " " + body["decision"] + " " + body["rationale"]).lower()
                    founder_prefs = [str(p.get("preference", "")) for p in
                                     (state["intent_model"].get("tech_preferences") or []) if p.get("preference")]
                    if body["driver"] == "founder_constraint":
                        selected = _selected_concept(state.get("conops"))
                        executes_confirmed_concept = bool(selected and selected[0] in body["executes_concept_ids"])
                        names_tech_preference = any(pref.lower() in adr_text for pref in founder_prefs)
                        if not executes_confirmed_concept and not names_tech_preference:
                            raise ModelStructureError(
                                f"{slot['decision_id']} '{body['title']}': driver founder_constraint requires "
                                "either executes_concept_ids linking the founder-confirmed CONOPS concept or "
                                "the named founder technology preference. A CONOPS choice is a product/operations "
                                "constraint, not a technology preference; link it instead of deleting it")
                    elif _re.search(r"mandat\w* by the founder|founder(-|\')s? (mandate|preference|requirement|decision)", adr_text):
                        raise ModelStructureError(
                            f"{slot['decision_id']} '{body['title']}': the text claims a founder mandate but the "
                            "driver is not founder_constraint - delete the mandate language or justify the driver")
                    # Founder-NAMED technologies rejected here need a SPECIFIC
                    # reason (2026-09-13 founder decision) - vague verdicts
                    # re-ask this ADR call, far cheaper than an assembly re-plan.
                    for opt in body["options_considered"]:
                        if not isinstance(opt, dict):
                            continue
                        oname, oreason = str(opt.get("option", "")), str(opt.get("reason", ""))
                        for t in founder_techs:
                            if _tech_in_text(t, oname) and not _specific_reason(oreason):
                                raise ModelStructureError(
                                    f"{slot['decision_id']} '{body['title']}': founder-named technology "
                                    f"{_display(t)!r} is rejected with the vague reason {oreason!r} - state "
                                    "the specific requirement it fails or the constraint it violates")
                    if slot["migration"]:
                        if not body["evidence"]:
                            raise ModelStructureError(f"{slot['decision_id']} {body['title']}: migration requires evidence")
                        if not body["affected_repositories"]:
                            raise ModelStructureError(f"{slot['decision_id']} {body['title']}: migration requires affected_repositories")
                    return body
                adr_bodies[did] = _call_json(
                    "architecture",
                    context + "\n\nTHE ADR SLOT (fixed - ids, migration flag and authorizations are pipeline-owned):\n"
                    + json.dumps({"decision_id": did, "slug": slot["slug"],
                                  "migration": slot["migration"], "authorizes": slot["authorizes"]})
                    + "\n\nDECIDED STACK (key: value):\n"
                    + json.dumps({k: (e.get("value") or e.get("status")) for k, e in entries.items()})
                    + lost_block + candidate_feedback + "\n\n" + ARCH_ADR_BODY + feedback,
                    log, progress, validate=_v_adr,
                    schema=arch_adr_body_schema(valid_refs, repo_names, [__import__("vmodel_contracts").canonical_concept_id(str(o.get("name", ""))) for o in (state.get("conops") or {}).get("concept_options", [])]),
                    stream_items=False, item_label=f"{did} {slot['slug']}")
                _save()

            # P2 ADR dedupe (deterministic, his 2026-09-13 pick): merge
            # same-technology ADRs BEFORE assembly; the merge log is visible.
            _advisories = _dedupe_adr_slots(part["adr_slots"], adr_bodies)
            for _a in _advisories:
                model_config._console_line(
                    f"     - ADR review advisory: {_a['left_id']} '{_a['left_slug']}' and "
                    f"{_a['right_id']} '{_a['right_slug']}' may overlap ({_a['reason']}); "
                    "nothing was merged")

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
                _check_decision_drivers(arch, state["intent_model"], state.get("conops"))  # ADR-0021
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
            gate_failures=[]
            try:
                _check_conops_execution(arch, state.get("conops"))
            except ModelStructureError as e:
                gate_failures.append(("CONOPS-execution", str(e)))
            try:
                _check_candidate_grounding(arch, state["intent_model"],
                                           contested=part.get("contested"))
            except ModelStructureError as e:
                gate_failures.append(("candidate-grounding", str(e)))
            try:
                _check_adr_substance(arch)
            except ModelStructureError as e:
                gate_failures.append(("ADR-substance", str(e)))
            if gate_failures and _adr_attempt == 1 and all(name == "ADR-substance" for name, _ in gate_failures):
                # Duplicate-only failure: the deterministic, lossless merge fixes it without throwing away every
                # ADR body (a full re-plan cost ~10-15 min and produced NEW duplicates in the 2026-10-02 run).
                # Anything the merge leaves, or any other gate it disturbs, falls through to the full re-plan.
                pairs = _duplicate_adr_pairs(arch)
                if pairs:
                    snapshot = (copy.deepcopy(part["adr_slots"]), copy.deepcopy(adr_bodies))
                    _merge_duplicate_adrs_one_at_a_time(
                        part["adr_slots"], adr_bodies, pairs,
                        lambda: ArchitectureOverview.model_validate(_assemble_architecture(
                            session_id=sid, baseline_id=baseline.baseline_id, entries=entries,
                            comp_slots=part["component_slots"], comp_bodies=comp_bodies,
                            adr_slots=part["adr_slots"], adr_bodies=adr_bodies, nfr=nfr)),
                        label="adr-duplicate-merge-first")
                    try:
                        merged_try = _assemble_architecture(
                            session_id=sid, baseline_id=baseline.baseline_id, entries=entries,
                            comp_slots=part["component_slots"], comp_bodies=comp_bodies,
                            adr_slots=part["adr_slots"], adr_bodies=adr_bodies, nfr=nfr)
                        arch_try = ArchitectureOverview.model_validate(merged_try)
                        _check_decision_drivers(arch_try, state["intent_model"], state.get("conops"))
                        _check_baseline_reconciliation(arch_try, baseline)
                        _check_conops_execution(arch_try, state.get("conops"))
                        _check_candidate_grounding(arch_try, state["intent_model"], contested=part.get("contested"))
                        _check_adr_substance(arch_try)
                    except (ValidationError, ModelStructureError, ValueError, KeyError) as exc:
                        part["adr_slots"][:] = snapshot[0]
                        adr_bodies.clear(); adr_bodies.update(snapshot[1])
                        model_config._console_line(
                            f"     - boundary=adr-duplicate-merge-first decision=fall-back-to-replan reason={str(exc)[:200]!r}")
                    else:
                        _save()
                        merged, arch = merged_try, arch_try
                        gate_failures = []
            if gate_failures:
                trace="; ".join(f"{name}: {detail}" for name,detail in gate_failures)
                if _adr_attempt == 1:
                    model_config._console_line(
                        "     - WARNING architecture gates requested revisions: " + trace
                        + " - re-planning the ADR set once with all requirements")
                    part.pop("adr_slots", None)
                    part.pop("adr_bodies", None)
                    _save()
                    candidate_feedback = ("\n\nTHE PREVIOUS ARCHITECTURE FAILED THESE ACTIVE "
                                          "DETERMINISTIC GATES. Fix ALL of them in this ADR set; "
                                          "do not trade one for another:\n- "
                                          + "\n- ".join(f"{name}: {detail}" for name,detail in gate_failures))
                    continue
                for name, detail in gate_failures:
                    model_config.capture_rejected_completion(
                        "architecture", f"gate:{name}", _adr_attempt,
                        json.dumps({"adr_slots": part.get("adr_slots"), "adr_bodies": adr_bodies}, indent=2), detail)
                duplicate_pairs = _duplicate_adr_pairs(arch)
                if gate_failures and all(name == "ADR-substance" for name, _ in gate_failures) and duplicate_pairs:
                    # One pair per scan: each merge renumbers ADR ids, so the first scan's pairs go stale.
                    _merge_duplicate_adrs_one_at_a_time(
                        part["adr_slots"], adr_bodies, duplicate_pairs,
                        lambda: ArchitectureOverview.model_validate(_assemble_architecture(
                            session_id=sid, baseline_id=baseline.baseline_id, entries=entries,
                            comp_slots=part["component_slots"], comp_bodies=comp_bodies,
                            adr_slots=part["adr_slots"], adr_bodies=adr_bodies, nfr=nfr)),
                        label="adr-duplicate-fallback")
                    _save()
                    merged = _assemble_architecture(
                        session_id=sid, baseline_id=baseline.baseline_id, entries=entries,
                        comp_slots=part["component_slots"], comp_bodies=comp_bodies,
                        adr_slots=part["adr_slots"], adr_bodies=adr_bodies, nfr=nfr)
                    arch = ArchitectureOverview.model_validate(merged)
                    _check_adr_substance(arch)
                    break
                from .errors import UserFacingError
                raise UserFacingError(
                    "stage 'architecture': the repaired architecture still failed deterministic gates (" + ", ".join(name for name,_ in gate_failures) + ").",
                    trace,
                    f"{model_config._rerun_guidance('architecture')} - inspect the named gate requirements "
                    "above; completed stack/component/NFR work is preserved and no model upgrade is required.")
            break
        if partials:
            partials.clear("architecture")
        data = merged
        revisions = dict(state.get("artifact_revisions") or {})
        prior_revision = revisions.get("architecture") or {}
        revision = int(prior_revision.get("revision") or 0) + 1
        saved_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        revisions["architecture"] = {"saved_at": saved_at, "revision": revision}
        if persist:
            persist("architecture", data, {**state, "artifact_revisions": revisions})
        resolved_questions = _resolve_answered_open_questions(state, persist, architecture=data)
        result = {"architecture": data, "architecture_advisories": _advisories,
                  "artifact_revisions": revisions,
                  "gate_feedback": _clear_feedback(state, "architecture")}
        # Nested mutation is not a LangGraph state update. Return the changed
        # SRS explicitly so the resolution marker survives the checkpoint and
        # deterministic assembly, including --from architecture resumes.
        if resolved_questions:
            result["srs"] = state["srs"]
        return result

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
                _accept_content(entry["spec_id"], entry["body_markdown"],
                                {d["spec_id"]: d["body_markdown"] for d in data},
                                Ledger.from_dict(state.get("findings_ledger")), _content_sink(persist))
                data.append(entry)
                if partials:
                    done[entry["spec_id"]] = entry
                    partials.save("items", list(done.values()))
        # JEV-D10: one content acceptance boundary for every path above (bulk, micro,
        # surgical, resumed partials) before anything is persisted as accepted.
        ledger = Ledger.from_dict(state.get("findings_ledger"))
        _accept_all(data, ledger, _content_sink(persist), only=(affected if (selective and state.get("items")) else None))
        out_state = {**state, "findings_ledger": ledger.to_dict()}
        if persist:
            persist("items", data, out_state)
            getattr(persist, "ledger", lambda *_: None)(ledger.to_dict())
        if partials:
            partials.clear("items")
        return {"items": data, "findings_ledger": ledger.to_dict()}

    def node_tests(state: GenState) -> dict:
        """Two V pairings in one pass: acceptance tests pair with SRS criteria,
        unit/module tests pair with item specs."""
        # ADR-0020: one slot per SRS acceptance criterion (acceptance level)
        # + one per item spec (unit level); T-ids and links are code-owned.
        slots = _experiment_cap("TESTS", build_test_skeleton(state["srs"], _unquarantined_items(state)),
                                "test slots (traceability gaps are expected under a cap)",
                                progress)
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
                   + "\n\nFOUNDER REGISTER (decided answers are binding; open/deferred inputs are not defaults):\n"
                   + json.dumps(_prompt_view(((state.get("intent_model") or {}).get("founder_register") or {})))
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
        from .schemas import for_stage as output_schema

        n = state.get("review_round", 0) + 1
        ledger = Ledger.from_dict(state.get("findings_ledger"))
        if state.get("findings_ledger") is None and state.get("items"):
            # A checkpoint that predates JEV-D10 has no ledger: it is UNREVIEWED, not clean.
            _accept_all(state["items"], ledger, _content_sink(persist))
        if state.get("repair_quick"):
            model_config._console_line("     - boundary=grounding decision=skipped-quick-repair (specs and tests unchanged by an SRS-only repair)")
        else:
            _ground_items(state, ledger, grounding, persist)
        # One requirement per review call bounds output by construction. The
        # prior two-requirement batches could exceed 8192 completion tokens on
        # dense vertical slices. Requirement remains the semantic join key.
        requirements = [r for sec in ("functional", "non_functional")
                        for r in state["srs"].get(sec) or []]
        chunks = [[requirement] for requirement in requirements] or [[]]
        scope_reqs = state.get("repair_scope")
        if scope_reqs:
            # after a scoped repair only the repaired requirements' clusters are re-reviewed (cheap, same prompt)
            chunks = [c for c in chunks if c and c[0].get("requirement_id") in set(scope_reqs)] or chunks
            model_config._console_line(
                f"     - boundary=review-scope round={n} requirements={sorted(scope_reqs)} chunks={len(chunks)} "
                "decision=re-review-repaired-requirements-only")
        saved = (partials.load("review") if partials else None) or {}
        findings = []
        for index, req_chunk in enumerate(chunks, 1):
            chunk_id = f"round-{n}-chunk-{index}"
            if chunk_id not in saved:
                req_ids = {r.get("requirement_id") for r in req_chunk}
                criterion_ids = {c.get("criterion_id") for r in req_chunk
                                 for c in r.get("acceptance") or []}
                chunk_items = [i for i in state["items"]
                               if req_ids & set(i.get("requirement_ids") or [])]
                spec_ids = {i.get("spec_id") for i in chunk_items}
                chunk_tests = [t for t in state["tests"]
                               if t.get("criterion_id") in criterion_ids or t.get("spec_id") in spec_ids]
                projected_input_chars = len(json.dumps(req_chunk)) + len(json.dumps(chunk_items)) + len(json.dumps(chunk_tests))
                model_config._console_line(
                    f"     - boundary=review-batch round={n} batch={index}/{len(chunks)} requirements={len(req_chunk)} "
                    f"linked_items={len(chunk_items)} linked_tests={len(chunk_tests)} projected_input_chars={projected_input_chars} "
                    "max_findings=24 max_fix_chars=320 decision=bounded-single-requirement-review")
                saved[chunk_id] = _call_json(
                    "review", STAGE_PROMPTS["review"].format(n=n)
                    + f"\n\nREVIEW CHUNK {index}/{len(chunks)}. Return ONLY concrete findings whose primary target "
                      "is in this one-requirement cluster. Never restate requirements, specs, tests, or rationale. "
                      "At most 24 findings. Each fix is one imperative sentence, at most 320 characters. "
                      "Cross-chunk consistency is checked deterministically after aggregation."
                    + _decided_architecture_block(state)
                    + "\n\nSRS REQUIREMENTS:\n" + json.dumps(_prompt_view(req_chunk))
                    + "\n\nLINKED ITEMS:\n" + json.dumps(_prompt_view(chunk_items))
                    + "\n\nLINKED TESTS:\n" + json.dumps(_prompt_view(chunk_tests)),
                    log, progress, validate=_normalized_review_fixes,
                    schema=output_schema("review"), stream_items=False,
                    item_label=f"review {index}/{len(chunks)} ({len(req_chunk)} requirements)")
                if partials:
                    partials.save("review", saved)
            chunk_findings = _normalized_review_fixes(saved[chunk_id])
            if len(chunk_findings) > REVIEW_CHUNK_MAX_FINDINGS:
                # Never silently drop: keep them all and mark the chunk as over its bound.
                ledger.mark_incomplete(f"review:round-{n}-chunk-{index}",
                                       f"chunk returned {len(chunk_findings)} findings (bound {REVIEW_CHUNK_MAX_FINDINGS})",
                                       returned=len(chunk_findings))
            findings.extend(chunk_findings)
        # Deterministic stable dedup: overlapping specs can appear in adjacent
        # requirement chunks, but one correction must be applied only once.
        fixes = []
        seen = set()
        for fix in _normalized_review_fixes(findings):
            key = (fix["scope"], tuple(sorted(fix["target_ids"])), fix["fix"].strip().casefold())
            if key not in seen:
                seen.add(key); fixes.append(fix)
        fixes, filtered = _filter_decided_stack_findings(fixes, state)
        fixes, noise_filtered = _filter_review_noise(fixes)
        if noise_filtered:
            by_class = {}
            for f in noise_filtered:
                by_class[f["noise_class"]] = by_class.get(f["noise_class"], 0) + 1
            ledger.history.append({"event": "review_filtered", "round": n, "reason": "filtered: noise-class",
                                   "classes": by_class, "count": len(noise_filtered),
                                   "fixes": [f["fix"][:200] for f in noise_filtered]})
            model_config._console_line(
                f"     - boundary=review-filter round={n} filtered={len(noise_filtered)} reason=noise-class "
                f"classes={by_class} decision=archived-not-opened (spec/test only; contradiction/absence findings are kept)")
        if filtered:
            ledger.history.append({"event": "review_filtered", "round": n, "reason": "filtered: decided-stack",
                                   "count": len(filtered), "fixes": [f["fix"][:200] for f in filtered]})
            model_config._console_line(
                f"     - boundary=review-filter round={n} filtered={len(filtered)} reason=decided-stack "
                "decision=archived-not-opened (spec/test findings asking to remove a technology the architecture decided)")
        raised_now = set()
        _ledger_review_findings(ledger, state, fixes, n, raised_out=raised_now)
        ledger.close_repaired(
            f"review:round-{n}", raised_now,
            lambda f: _target_revision(state, {"scope": f["scope"], "target_ids": f["detail"].get("target_ids") or f["targets"]}),
            [(fx["target_ids"], fx["fix"]) for fx in fixes])
        # Archive every chunk before partial cleanup.
        if persist and hasattr(persist, "review"):
            persist.review({"round": n, "chunks": {k: v for k, v in saved.items()
                           if k.startswith(f"round-{n}-chunk-")},
                           "all_chunk_findings": findings, "retained_fixes": fixes, "filtered_decided_stack": filtered, "filtered_noise_class": noise_filtered,
                           "aggregation_omitted_count": max(0, len(findings) - len(fixes)),
                           "disposition": "proposed; resolution not verified",
                           "ledger": ledger.to_dict(),
                           "state": {**state, "review_round": n, "review_fixes": fixes}})
        if persist is not None:
            getattr(persist, "ledger", lambda *_: None)(ledger.to_dict())
        if partials:
            partials.clear("review")
        return {"review_round": n, "review_fixes": fixes, "findings_ledger": ledger.to_dict(), "repair_scope": None,
                "quick_stop": bool(state.get("repair_quick")), "repair_quick": False}

    def route_review(state: GenState) -> str:
        # JEV-D10: no unattended replay. Repair runs only when explicitly requested for
        # named findings and only under the attempt ceiling. Otherwise assembly decides,
        # and its gate blocks while anything is open.
        request = state.get("repair_request") or {}
        if request.get("finding_ids"):
            ledger = Ledger.from_dict(state.get("findings_ledger"))
            attempts = sum(1 for h in ledger.history if h.get("event") == "repair")
            if attempts < REPAIR_ATTEMPT_CEILING:
                return "apply_fixes"
        if state.get("quick_stop"):
            return "stop"          # a quick repair ends at its review evidence; assembly is a separate, later run
        return "assemble"

    def node_apply_fixes(state: GenState) -> dict:
        request = state.get("repair_request") or {}
        ledger = Ledger.from_dict(state.get("findings_ledger"))
        chosen = [ledger.findings[f] for f in request.get("finding_ids") or []
                  if f in ledger.findings and ledger.findings[f]["state"] == "OPEN"
                  and ledger.findings[f]["rule_id"] == "review.llm"]
        if not chosen:
            raise AssemblyBlocked(["scoped repair requested for findings that are not open review findings"],
                                  list(request.get("finding_ids") or []))
        ledger.history.append({"event": "repair", "finding_ids": [f["finding_id"] for f in chosen],
                               "requested_by": request.get("requested_by", "founder"),
                               "before": {f["finding_id"]: f["artifact_revision"] for f in chosen}})
        # Only the selected findings go to the author calls (never the whole review output).
        state = {**state, "review_fixes": [f["detail"] for f in chosen], "findings_ledger": ledger.to_dict()}
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
        reviewed=(state.get('semantic_reviews') or {}).get('srs')
        if reviewed and reviewed.get('decisions'):
            from vmodel_contracts.semantic_review import SemanticReview
            record=SemanticReview.model_validate(reviewed)
            accepted={d.item_id for d in record.decisions if d.action=='accept'}
            originals={r['requirement_id']:r for section in ('functional','non_functional') for r in state['srs'].get(section) or []}
            for section in ('functional','non_functional'):
                data[section]=[originals[r['requirement_id']] if r.get('requirement_id') in accepted else r for r in data.get(section) or []]
        data.setdefault("session_id", state["intent_model"]["session_id"])
        data.setdefault("intent_model_ref", state["intent_model"]["session_id"])
        _check_ac_ids_stable(state["srs"], data)
        if data is state["srs"]:
            # spec/test-only repair: no SRS text changes, so there is nothing to approve - continue to items
            return {"review_fixes": fixes, "repair_request": None,
                    "repair_scope": _review_scope_requirements(state, fixes),
                    "findings_ledger": state["findings_ledger"]}
        proposal = _repair_proposal(state, data, fixes, [f["finding_id"] for f in chosen])
        if state.get("repair_quick"):
            # --quick: the founder asked for no stops. Apply the SRS fix, keep the before/after on disk as evidence
            # (never a gate), skip items/tests, and let the scoped review re-check the repaired requirements.
            ledger.history.append({"event": "repair_auto_approved", "finding_ids": proposal["finding_ids"],
                                   "changed": sorted(proposal["changed"]), "unchanged_targets": proposal["unchanged_targets"],
                                   "mode": "quick"})
            if persist is not None:
                getattr(persist, "repair_proposal", lambda *_: None)(proposal, render_repair_proposal(proposal))
                persist("srs", data, state)
            model_config._console_line(
                f"     - boundary=repair-quick changed={sorted(proposal['changed'])} "
                f"unchanged_targets={proposal['unchanged_targets']} decision=auto-approved-no-stop "
                "(items and tests NOT regenerated; this run ends at the scoped review)")
            return {"srs": data, "review_fixes": fixes, "repair_request": None,
                    "repair_scope": _review_scope_requirements({**state, "srs": data}, fixes),
                    "findings_ledger": ledger.to_dict()}
        if persist is not None:
            getattr(persist, "repair_proposal", lambda *_: None)(proposal, render_repair_proposal(proposal))
        model_config._console_line(
            f"     - boundary=repair-proposal changed={sorted(proposal['changed'])} "
            f"unchanged_targets={proposal['unchanged_targets']} decision=stopped-for-founder-approval "
            "(--approve-repair applies it, --reject-repair drops it)")
        # the repaired SRS is NOT applied here: the founder approves the before/after first
        return {"repair_proposal": proposal, "review_fixes": fixes, "repair_request": None,
                "repair_scope": _review_scope_requirements({**state, "srs": data}, fixes),
                "findings_ledger": state["findings_ledger"]}

    def node_assemble(state: GenState) -> dict:
        """Deterministic tail. Validate hard; ship nothing with orphan traces."""
        if state.get("findings_ledger") is None:
            raise AssemblyBlocked(["no findings ledger: this checkpoint was never content-reviewed (unreviewed, not clean)"], [])
        gate_ledger = Ledger.from_dict(state["findings_ledger"])
        gate = gate_ledger.gate({i.get("spec_id"): revision_of(i.get("body_markdown") or "") for i in state.get("items") or []})
        if gate["blocked"]:
            getattr(persist, "ledger", lambda *_: None)(gate_ledger.to_dict())
            raise AssemblyBlocked(gate["reasons"], gate["open"])
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

    stoppable = ("needs", "srs", "architecture", "items", "tests")
    if stop_after is not None and stop_after not in stoppable:
        raise ValueError(f"unsupported stop_after: {stop_after}")

    def edge(source: str, destination: str) -> None:
        g.add_edge(source, END if stop_after == source else destination)

    g.set_entry_point("needs")
    edge("needs", "srs")
    if review:
        from langgraph.types import interrupt

        def _make_gate(stage_key: str, artifact_key: str):
            def node_gate(state: GenState) -> dict:
                advisories=state.get("architecture_advisories", []) if stage_key == "architecture" else []
                if expert_opinion:
                    all_prior_regenerations = [o for o in (state.get("expert_opinions") or [])
                                               if o.get("gate") == stage_key and o.get("action") == "regenerate"]
                    # Pre-structured checkpoints cannot be meaningfully rendered,
                    # scoped, or counted against today's retry budget. Re-review
                    # the current artifact under the current protocol on resume.
                    prior_regenerations = [o for o in all_prior_regenerations
                                           if o.get("review_protocol") == 2]
                    legacy_regenerations = len(all_prior_regenerations) - len(prior_regenerations)
                    if legacy_regenerations:
                        model_config._console_line(
                            f"     - boundary=expert-checkpoint-migration gate={stage_key} "
                            f"legacy_regenerations={legacy_regenerations} current_protocol_regenerations={len(prior_regenerations)} "
                            "decision=re-review-current-artifact reason=stored-opinions-predate-structured-objection-scope")
                    if len(prior_regenerations) >= 2:
                        from .errors import UserFacingError
                        from vmodel_contracts.runtime import format_expert_review_rounds
                        objections = format_expert_review_rounds(prior_regenerations)
                        model_config._console_line(
                            f"  WARNING boundary=expert-park gate={stage_key} regeneration_rounds={len(prior_regenerations)} "
                            "limit=2 decision=park-for-human-review")
                        # Keep console/run.log to one summary line. The structured
                        # objections belong once in the Error/Likely cause block.
                        raise UserFacingError(
                            f"expert review parked the {stage_key} gate for human review after two regeneration rounds.",
                            "the independent reviewer requested changes twice; another automatic full-stage rewrite "
                            "would be an unbounded token loop, not stronger evidence.\n\n" + objections,
                            "rerun the same command without --expert-opinion to inspect and decide this gate manually; "
                            "the latest artifact and checkpoint are preserved, along with the reviewer objections above.")
                    artifact_label = {
                        "srs": "Software Requirements Specification (SRS): requirements and acceptance criteria, not executable tests",
                        "architecture": "system architecture: stack, components, ADRs, and non-functional positions",
                    }[stage_key]
                    artifact_snapshot = state.get(artifact_key)
                    artifact_revision = (state.get("artifact_revisions") or {}).get(stage_key) or {"revision": 0, "saved_at": "legacy-unknown"}
                    artifact_fingerprint = __import__("hashlib").sha256(
                        json.dumps(artifact_snapshot, sort_keys=True, default=str).encode()).hexdigest()[:12]
                    model_config._console_line(
                        f"     - boundary=expert-input gate={stage_key} artifact_key={artifact_key} "
                        f"artifact_sha256={artifact_fingerprint} artifact_revision={artifact_revision['revision']} "
                        f"artifact_saved_at={artifact_revision['saved_at']} prior_regenerations={len(prior_regenerations)} "
                        "decision=review-current-checkpoint-artifact")
                    payload=(f"Act as an independent expert reviewer. You are judging this exact artifact type: {artifact_label}. "
                             "Judge it for its own V-model stage; do not demand artifacts owned by a later stage. "
                             "First line: APPROVE or REGENERATE. APPROVE when there is no "
                             "blocking contradiction, missing trace, unsafe assumption, or build-stopping ambiguity. "
                             "Non-blocking improvements and polish MUST stay as suggestions in RATIONALE under APPROVE; "
                             "they are not grounds to regenerate. REGENERATE only for a named blocking defect, and write "
                             "FEEDBACK as one or more lines in this exact form: OBJECTION: artifact=<reviewed artifact name> | rule=<named rule or criterion> | location=<section, id, field, or quoted fragment> | resolution=<concrete change>. Every objection must include all four fields. For invention or containment claims, location must quote concrete verbatim examples from the artifact. "
                             "The SRS is solution-agnostic except its design_constraints section: accept mandated solution limits there only when each has valid interview source_turns. Algorithms and algorithm families (AES-256, Argon2, bcrypt, SHA-256) may state required properties and are not design objections. Solution mentions introduced as non-binding examples with 'like', 'such as', 'e.g.', or 'for example' are allowed; imperative choices such as 'use OAuth' and API directives remain defects. Flag untraced constraints and binding solution detail in requirement bodies. Then write RATIONALE: free prose. "
                             "You may read the full context for coverage, but REGENERATE may name only defects fixable in this reviewed artifact. "
                             "For a defect whose fix belongs in an upstream artifact, use the same OBJECTION line with that upstream artifact name; "
                             "it will be reported as an advisory and must not be a REGENERATE ground. Never claim founder approval. "
                             "Advisories must be considered, but are not themselves blocking.\nARTIFACT:\n" + json.dumps(state.get(artifact_key))
                             + "\nADVISORIES:\n" + json.dumps(advisories))
                    from .errors import UserFacingError
                    text="";verdict=None;objections=[];upstream_advisories=[];specificity_errors=[];retry_feedback=None
                    low_specificity_feedback="";low_specificity_rationale="No separate rationale supplied."
                    for attempt in (1,2):
                        try:
                            raw=_invoke("expert_opinion",payload,log,progress,
                                        feedback=retry_feedback,
                                        stream_items=False,item_label=f"{stage_key} gate",attempt=attempt,
                                        retry_reason=("verdict" if attempt==2 else None))
                        except __import__("vmodel_contracts.runtime", fromlist=["AdaptiveWatchdogTimeout"]).AdaptiveWatchdogTimeout as exc:
                            if attempt == 1:
                                basis = f"{exc.ratio:.1f}x stage median" if exc.ratio is not None else "the cold-start floor"
                                model_config._console_line(
                                    f"  WARNING self-heal: call ran past {basis} ({exc.deadline_s:.0f}s); killed and retrying")
                                continue
                            raise UserFacingError(
                                f"expert opinion watchdog exhausted its retry budget at the {stage_key} gate.",
                                f"the call exceeded {exc.deadline_s:.0f}s twice.",
                                "rerun the same command; the checkpoint and completed work are preserved.") from exc
                        except UserFacingError:
                            raise
                        except Exception as e:
                            raise UserFacingError(
                                f"expert opinion failed at the {stage_key} gate.",
                                f"unexpected {type(e).__name__}: {str(e)[:240]}",
                                "rerun the same command; the checkpoint is preserved and no model upgrade is required.") from e
                        text=raw
                        from vmodel_contracts.runtime import parse_expert_verdict
                        verdict = parse_expert_verdict(raw, ("approve", "regenerate"))
                        specificity_errors = []
                        objections = []
                        if verdict is not None and verdict["action"] == "regenerate" and verdict["feedback"]:
                            from vmodel_contracts.runtime import classify_expert_objections
                            known_artifacts = ("intent", "srs") if stage_key == "architecture" else ("intent",)
                            objections, upstream_advisories, specificity_errors = classify_expert_objections(
                                verdict["feedback"], stage_key, known_artifacts,
                                json.dumps(artifact_snapshot, sort_keys=True, default=str))
                            fabricated_evidence = [error for error in specificity_errors if error.startswith("fabricated evidence")]
                            specificity_errors = [error for error in specificity_errors if not error.startswith("fabricated evidence")]
                            for error in fabricated_evidence:
                                model_config._console_line(
                                    f"     - boundary=expert-evidence gate={stage_key} decision=invalidate-objection reason={error}")
                            if specificity_errors:
                                low_specificity_feedback = verdict["feedback"]
                                low_specificity_rationale = verdict["rationale"]
                                retry_feedback = ("SPECIFICITY ERROR: " + "; ".join(specificity_errors) + ". Return REGENERATE with "
                                                  "one OBJECTION line per blocker using artifact, rule, location, and resolution fields.")
                                verdict = None
                            elif not objections and (upstream_advisories or fabricated_evidence):
                                verdict["action"] = "approve"
                                model_config._console_line(
                                    f"     - boundary=expert-scope gate={stage_key} own_blockers=0 "
                                    f"upstream_advisories={len(upstream_advisories)} decision=approve-with-upstream-advisories")
                        if verdict is not None and (verdict["action"] == "approve" or verdict["feedback"]):
                            if verdict.get("extracted_from") == "json_object":
                                model_config._console_line(
                                    f"     - boundary=expert-verdict gate={stage_key} input=json-object "
                                    f"allowed=APPROVE,REGENERATE action={verdict['action']} "
                                    f"feedback_present={bool(verdict['feedback'])} decision=accept-deterministic-extraction")
                            break
                        reason = ("low-specificity objection: " + "; ".join(specificity_errors) if specificity_errors
                                  else "regenerate verdict missing FEEDBACK" if verdict else "unparseable expert verdict")
                        if attempt == 1 and not retry_feedback:
                            retry_feedback = ("FORMAT ERROR: return APPROVE or REGENERATE on the first nonblank line. "
                                              "For REGENERATE, add FEEDBACK: with one structured OBJECTION line per blocker, "
                                              "then RATIONALE:.")
                        path = model_config.capture_rejected_completion("expert_opinion", f"{stage_key} gate", attempt, raw, reason)
                        model_config._console_line(
                            f"     - boundary=expert-verdict gate={stage_key} attempt={attempt} allowed=APPROVE,REGENERATE "
                            f"parsed={bool(verdict)} feedback_present={bool(verdict and verdict['feedback'])} "
                            f"decision={'format-retry' if attempt == 1 else 'reject'} reason={reason} output={path}")
                        verdict = None
                    if verdict is None or (verdict["action"]=="regenerate" and not verdict["feedback"]):
                        from .errors import UserFacingError
                        if specificity_errors:
                            from vmodel_contracts.runtime import format_expert_review_rounds
                            low_specificity = format_expert_review_rounds([{
                                "rationale": low_specificity_rationale,
                                "feedback": low_specificity_feedback or text,
                                "artifact_revision": artifact_revision["revision"],
                                "artifact_saved_at": artifact_revision["saved_at"],
                                "artifact_sha256": artifact_fingerprint,
                            }])
                            raise UserFacingError(f"expert opinion returned low-specificity {stage_key} objections twice.",
                                "LOW-SPECIFICITY: " + "; ".join(specificity_errors) + "\n\n" + low_specificity,
                                "inspect the numbered objections above; each blocker must name artifact, rule, location, and resolution.")
                        raise UserFacingError(f"expert opinion could not parse the {stage_key} gate verdict.",
                            f"Expected first line APPROVE or REGENERATE (with structured OBJECTION feedback). Output began: {text[:200]!r}",
                            "rerun the same command; the checkpoint is preserved and no model upgrade is required.")
                    record={"gate":stage_key,"mode":"simulated_expert_opinion","human_review_required":True,"review_protocol":2,
                            "action":verdict["action"],"feedback":verdict.get("feedback","")+"", "rationale":verdict["rationale"],
                            "objections": objections, "upstream_advisories": upstream_advisories,
                            "artifact_sha256": artifact_fingerprint, "artifact_revision": artifact_revision["revision"],
                            "artifact_saved_at": artifact_revision["saved_at"]}
                    model_config._console_line(f"   SIMULATED EXPERT OPINION at {stage_key}: {verdict['action']} - human review still required")
                    model_config._console_line(f"     rationale: {verdict['rationale']}")
                    if verdict["action"] == "regenerate":
                        model_config._console_line(f"     regeneration feedback: {verdict['feedback']}")
                    if upstream_advisories:
                        from vmodel_contracts.runtime import format_expert_review_rounds
                        advisory_text = format_expert_review_rounds([record]).split("Upstream advisories:", 1)[1]
                        model_config._console_line("     Upstream advisories:" + advisory_text)
                    out={"expert_opinions":state.get("expert_opinions",[])+[record]}
                    if verdict["action"]=="regenerate":
                        if stage_key == "architecture" and partials and hasattr(partials, "regenerate"):
                            partials.regenerate("architecture", "simulated expert review requested architecture regeneration")
                        out["gate_feedback"]={stage_key:verdict["feedback"]}
                    return out
                verdict = interrupt({"gate": stage_key, "artifact": state.get(artifact_key),"advisories":advisories})
                if verdict.get("action") == "regenerate":
                    if stage_key == "architecture" and partials and hasattr(partials, "regenerate"):
                        partials.regenerate("architecture", "founder review requested architecture regeneration")
                    return {"gate_feedback": {stage_key: verdict["feedback"]}}
                return {}
            return node_gate

        # NOT wrapped in track(): interrupt() raises through the progress
        # contextmanager and would print a spurious FAILED line per pause.
        g.add_node("gate_srs", _make_gate("srs", "srs"))
        g.add_node("gate_architecture", _make_gate("architecture", "architecture"))
        if stop_after == "srs":
            g.add_edge("srs", END)
        else:
            g.add_edge("srs", "gate_srs")
        g.add_conditional_edges("gate_srs",
                                lambda st: "srs" if _gate_feedback(st, "srs") else "architecture",
                                {"srs": "srs", "architecture": "architecture"})
        if stop_after == "architecture":
            g.add_edge("architecture", END)
        else:
            g.add_edge("architecture", "gate_architecture")
        g.add_conditional_edges("gate_architecture",
                                lambda st: "architecture" if _gate_feedback(st, "architecture") else "items",
                                {"architecture": "architecture", "items": "items"})
    else:
        edge("srs", "architecture")
        edge("architecture", "items")
    edge("items", "tests")
    edge("tests", "review")
    g.add_conditional_edges("review", route_review, {"apply_fixes": "apply_fixes", "assemble": "assemble", "stop": END})
    g.add_conditional_edges(
        "apply_fixes",
        lambda st: "stop" if st.get("repair_proposal") else ("review" if st.get("repair_quick") else "items"),
        {"stop": END, "items": "items", "review": "review"})
    g.add_edge("assemble", END)
    return g.compile(checkpointer=checkpointer or MemorySaver(serde=checkpoint_serde()))



