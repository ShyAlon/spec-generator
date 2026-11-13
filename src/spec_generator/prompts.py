"""Generation prompts. The methodology context is stable across runs and
across stages: put it first and cache it (prompt caching is the single
biggest generation-side lever in the cost model)."""

METHODOLOGY = """You are generating a systems-engineering specification following the V model.
The artifact hierarchy: stakeholder needs -> SRS (system requirements) -> architecture ->
item-level specs -> test hierarchy (unit, module, system, acceptance) with full traceability.

Authoring rules (these are what separate this from a free PRD generator):
- Every requirement traces to at least one stakeholder need. No orphans.
- The SRS is SOLUTION-AGNOSTIC: it says WHAT must be verifiably true, never HOW.
  Requirement statements and acceptance criteria are observable, well-formed
  shall-statements: 'user data shall be encrypted in transit', 'a seeded
  disagreement shall produce exactly one alert'. NEVER endpoints, HTTP verbs,
  status codes, URL paths, JSON shapes, database or UI widgets, and NEVER named
  technologies or algorithms (no 'AES-256', 'OAuth', 'Kafka' - those are design
  decisions belonging to the architecture stage). NEVER invent artifacts,
  features, or numbers the stakeholder did not state.
- The executable HOW that the architecture and specs define (http GET /health, CLI
  commands, db probes) is authored at the TEST stage - never in the SRS. UI markup
  (selectors, ids, test ids) is authored nowhere in the spec: the developer defines it.
- Downstream artifacts REFERENCE requirements by id (R-n) through the traceability
  matrix. Restating or regrouping a requirement's text downstream is a defect.
- Item specs are written FOR A CODING AGENT: interface, data model, behaviors, error cases,
  non-functionals. No marketing language.
- Numbering is owned by the pipeline: ids and trace links are assigned in code; fill exactly the slots you are given, keyed by the given id.
- Output STRICT JSON matching the requested schema. No prose around it."""

STAGE_PROMPTS = {
    "needs": "From this intent model, write the stakeholder needs set (N-1..). "
             "The intent model already carries interview-level needs (SN-ids, founder language, "
             "classified functional/non_functional): carry them forward - every SN-id maps to at "
             "least one N-id (intent_need_ids), with kind preserved and the stakeholder named. "
             "Add needs the founder implied but did not state (mark their intent_need_ids []).",
    "srs": "Author the SOFTWARE REQUIREMENTS SPECIFICATION prose. The skeleton is FIXED: "
           "one requirement slot per need is given with requirement_id, kind, need_ids, and "
           "intent_fields already assigned - never change, renumber, add, or drop them. "
           "Write purpose (why the system exists, stakeholder language), scope_summary (the "
           "boundary), and for EACH slot: title, statement (well-formed shall-statement; shall followed by a space, never a comma; "
           "solution-agnostic), priority (must/should/could), and acceptance criteria as plain "
           "statements that are OBSERVABLE outcomes - no implementation mechanics, no named "
           "technologies, no invented numbers or artifacts (ids are assigned by the pipeline). "
           "purpose and scope_summary describe THE FOUNDER'S PRODUCT, in stakeholder language, "
           "from the INTENT below - never the V model, this pipeline, or these instructions. "
           "Non-functional slots must NOT silently vanish into prose: give each its own real, "
           "verifiable requirement text. Return one JSON object: {purpose, scope_summary, "
           "fills: [{requirement_id, title, statement, priority, acceptance: [statements]}]} "
           "- exactly one fill per given slot.",
    "architecture": "See ARCH_* prompts: the architecture stage is a micro-pipeline "
                    "(ADR-0022 applied to the 2026-09-11 phi4 failure) - plan calls propose "
                    "slots, one small call fills each slot, code assembles and validates.",
    "items": "Write item-level specs (SPEC-...) covering EVERY requirement in the SRS: each "
             "requirement id must appear in at least one spec's requirement_ids. Reference "
             "requirements by id through requirement_ids - never restate or regroup requirement "
             "text, and never invent ids: use only requirement ids that exist in the SRS below. "
             "One JSON array.",
    "validation": "You are a validation engineer. Machine tests already check that the system was built right; you list what only a PERSON can judge, so the founder can confirm the right system was built for ONE stakeholder need. "
                  "For the need below, return zero or more validation items. One item per distinct human judgment: what the person does, what they look or listen for, what counts as a pass, how the result is recorded, and the device or setting required. "
                  "Describe it in product terms the founder understands. NEVER write a selector, element id, CSS class, data-testid or other markup. "
                  "Do not restate a check a machine test already makes (the criteria below list their test_ids). When the need has no machine-verifiable criterion at all (machine_covered is false), at least one item is required: the need is otherwise unvalidated. "
                  "criterion_id is optional: set it to one of this need's criteria when the item judges what that criterion's machine proxy measures, otherwise null. components: pick only from the architecture's component names. "
                  "Add nothing the need does not state. Return one STRICT JSON object {items: [{criterion_id, components, what_the_human_does, what_to_observe, pass_when, evidence, environment}]}.",
    "tests": "You are a QA engineer writing tests for ONE acceptance criterion or spec at a time. Each test asserts "
             "ONLY what its one criterion or spec states: add no scenarios, failure modes, security, encryption, "
             "localization, accessibility or latency numbers that the cited criterion or spec does not state. "
             "expected_outcomes must be observable. runner http needs execution.url, "
             "method and expected_status, cli needs execution.command. A runner ui test describes ONE interaction at the system boundary and the measurable response, in product terms: steps are what the user or system does, expected_outcomes state a value or state that can be true or false ('the score screen shows 10 of 10'), and the elements involved are named in product language ('the score display'). NEVER write a selector, element id, CSS class, data-testid or other markup: the developer owns the markup and binds it to the test later. A check only a person can judge (how something looks, feels or sounds, a physical device or setting) is NOT a test: write the measurable part of it as a machine test (a value, a state, an output a check can fail on) and leave the human judgment to the validation stage, which runs after this one. Author the test prose. The skeleton is FIXED: one acceptance-level slot per SRS "
             "acceptance criterion (criterion_id linked) and one unit-level slot per item spec "
             "(spec_id linked), test_ids already assigned - never change, renumber, add, or drop "
             "them. For EACH slot write description, components (the names of the architecture components this test exercises, copied exactly from the architecture's component list; pick from candidate_components when it is given), runner ('http' | 'ui' | 'cli' | 'static'; machine-verifiable only), and definition using the canonical procedure shape: "
             "{preconditions: [string], steps: [string], expected_outcomes: [string], cleanup: "
             "[string], cases: [{name, inputs: [string], expected_outcomes: [string]}], execution: "
             "{url?, method?, body_json?, expected_status?, command?, expected_exit?, "
             "expected_stdout_contains?}}. Always include all six outer definition "
             "fields; use [] where a phase does not apply and {} when no execution field applies. "
             "Put readable technical detail in the procedure strings, and duplicate only the "
             "small machine-readable subset into execution when it applies. Use cases only for a "
             "real data/scenario matrix - never "
             "flatten or discard a valuable matrix. HERE you author the executable HOW, because "
             "the architecture now exists. Return one JSON object: {fills: [{test_id, description, "
             "runner, definition}]} - exactly one fill per given slot.",
    "srs_overview": "Author the SRS opening: purpose (why the system exists, stakeholder language) "
                    "and scope_summary (the boundary). Describe THE FOUNDER'S PRODUCT from the INTENT "
                    "below - never the V model, this pipeline, or these instructions.",
    "srs_slot": "Author ONE requirement fill for the given slot. The skeleton is FIXED - never change, "
                "renumber, add, or drop slots; you fill exactly the slot given here. Write title, "
                "statement (well-formed shall-statement; shall followed by a space, never a comma; solution-agnostic), priority "
                "(must/should/could), and acceptance criteria as plain statements that are OBSERVABLE "
                "outcomes - no implementation mechanics, no named technologies, no invented numbers "
                "or artifacts.",
    "items_plan": "Plan the item-level specs (SPEC-...) covering EVERY requirement in the SRS. The "
                  "module grouping is yours to design - that IS the analysis - but every requirement "
                  "id must appear in at least one plan's requirement_ids. Reference requirements by "
                  "id - never restate or regroup their text, never invent ids. This call plans ONLY: "
                  "spec bodies are written separately, one spec per call.",
    "items_body": "Write the body of ONE item spec, for a coding agent to execute against: interface, "
                  "data model, behaviors, error cases, non-functionals. Markdown. No marketing "
                  "language. Reference requirements by id; never restate their text.",
    "review": "Review pass {n}: find gaps, contradictions, orphan requirements and uncheckable criteria. "
              "Rules depend on the artifact's scope. "
              "requirement (the SRS): flag solution language - named technologies, UI widgets, APIs - because requirements stay "
              "solution-agnostic. "
              "spec and test: the architecture has ALREADY DECIDED the stack (the DECIDED ARCHITECTURE block is binding). "
              "Naming the decided technologies, APIs and HTTP methods in a spec or a test is REQUIRED. Never flag it "
              "and never ask to remove, replace, abstract or avoid it. For spec and test flag ONLY: missing behavior, "
              "contradictions with the SRS or with the decided architecture, orphan items, and uncheckable items. "
              "A test is flagged ONLY for contradicting its own criterion or spec, for an expected result that cannot be "
              "observed, or for naming a selector, element id, CSS class or test id (UI markup is the developer's). A spec 'add behavior' finding must cite a requirement or AC id. "
              "A technology that is NOT in the decided architecture may be flagged as a contradiction. "
              "Return a JSON list of typed fixes {{scope, target_ids, fix}}, or []. scope is requirement|spec|test|systemic. "
              "Use exact R-/SPEC-/T- ids for a scoped fix; use systemic with [] only when no safe target exists.",
}


# 0.13.0 architecture micro-pipeline (ADR-0022 doctrine applied to the
# 2026-09-11 phi4 failure: the single giant architecture call asked a small
# local model to keep two free-form maps key-synchronized - a relation no
# grammar can enforce. Now: tiny planning calls propose SLOTS, one shallow
# call fills each slot, and code owns ids, grounding checks, and assembly.
# Every call shares the same context block first (prompt-caching friendly),
# then its own small task, then a final FORMAT CONTRACT - the model follows
# the last frame it read.)

ARCH_CONTEXT = """You are reconciling an SRS against a SystemBaseline to produce a system architecture.
The baseline is the source of truth; NEVER infer a stack from an archetype or a generic
product pattern. Rules that apply to every architecture call:
- In existing mode, every grounded claim cites ONLY the supplied canonical baseline refs,
  exactly as written. A technology choice with NO baseline refs is a MIGRATION (a departure
  from the recorded system) and must be authorized by a migration ADR.
- In greenfield mode there is no baseline: refs stay empty and nothing is a migration.
- Components and decisions REFERENCE requirement ids (R-n) - never restate requirement text.
- Ids are assigned by the pipeline: ADR numbers, and the links between a departure and the
  ADR authorizing it, are wired in code. You never write them.
- Output STRICT JSON matching the requested schema. No prose around it."""

ARCH_STACK_PLAN = """List the technology slots this architecture must decide (e.g. "language",
"framework", "database", "model_runtime", "ci"). Judge from the SRS, the founder's
constraints, and the baseline - one short snake_case key per slot, no values yet.
FORMAT CONTRACT (final instruction - overrides anything above that conflicts): return one
STRICT JSON object {stack_keys: [strings]} - only the keys, 3 to 12 of them, no duplicates."""

ARCH_STACK_ENTRY = """Decide ONE stack slot. Return exactly one legal JSON shape:
1. Concrete selection: {technology: non-empty string, off_list_justification: string|null, baseline_refs:[refs]}.
2. Founder-delegated or ambiguous discovery: {status:"research_required", research_brief: non-empty string}. This is the DEFAULT when the right answer depends on current external facts or the founder delegated the choice. State what current registries, vendor docs, web sources, criteria, and requirement must be checked. Never answer from model memory.
3. Genuine irrelevance: {status:"not_applicable", reason: non-empty string}. Use this rarely, only when the slot truly does not apply to the concept; the reason must argue why.
An empty technology without an explicit status is illegal. A missing baseline fact does not make a slot not_applicable. If a concrete technology is grounded in baseline facts, cite every ref and copy the fact text verbatim. Otherwise baseline_refs is [] and the choice is new design or an authorized migration. Off-list concrete picks require a substantive justification."""

ARCH_STACK_OPTIONS = """Frame ONE technical architecture choice for a non-technical founder.
Return a proposed design default, never a founder answer or a researched recommendation.
Use the supplied requirements, explicit founder constraints and baseline as binding evidence.
Do not change a founder-required technology or preserved baseline fact. If a constraint makes an
option illegal, exclude it. Never silently choose an answer to an OPEN/DEFERRED product input.
Write a short plain-language title, one-sentence purpose, brief requirement-grounded reason,
and two or three options: concrete value, plain-language label, one material implication/tradeoff.
Set default_value explicitly to one option's value, not arbitrary first-option order.
Cite real requirement_ids. No markdown, investigation instructions, prices, versions, or claims
about current vendor capabilities: external research has NOT run. Code marks research pending.
Return STRICT JSON {title,purpose,default_value,reason,requirement_ids,options:[{value,label,implication}]}.
"""

ARCH_COMPONENT_PLAN = """List the software components of this architecture. For each: a short
unique name, the requirement ids (R-n) it realizes, the baseline refs proving where it already
exists (existing mode: a preserved component NEEDS refs), and migration=true only for a
new/replaced component IN EXISTING MODE. In greenfield mode (empty baseline) every component
is new design: migration is ALWAYS false there - migration means departing from an existing
baseline, and there is none. Cover every requirement id across the components.
FORMAT CONTRACT (final instruction - overrides anything above that conflicts): return one
STRICT JSON array of {name, requirement_ids, baseline_refs, migration} - names unique, ids and
refs only from the supplied lists."""

ARCH_COMPONENT_BODY = """Write the responsibility of ONE component: one or two sentences naming
what it owns and what it deliberately does not own, grounded in its requirements and baseline
refs. Reference requirement ids; never restate their text.
FORMAT CONTRACT (final instruction - overrides anything above that conflicts): return one
STRICT JSON object {responsibility: string}."""

ARCH_ADR_PLAN = """Plan the OPTIONAL architecture decision records (ADRs): the load-bearing
design choices that are NOT already forced. Forced migration decisions are pipeline-planned
and listed under PIPELINE-REQUIRED MIGRATION ADRS - never re-propose them. No arbitrary
choices, but no strawmen either. One short slug per decision. The pipeline assigns ADR numbers
in order, required migration ADRs first.
FORMAT CONTRACT (final instruction - overrides anything above that conflicts): return one
STRICT JSON array of {slug} objects - slugs only, no other fields ([] when there are no
optional decisions)."""

ARCH_ADR_BODY = """Write ONE architecture decision record against its fixed slot: title, the
decision itself, the rationale, its driver, the rejected options. Never use the ADR id as the
decision (for example, "consolidate into ADR-2"); name the concrete architecture choice.
Slots shown as research_required are unresolved: do not claim, imply, or select a technology for them.
DRIVER DISCIPLINE (hard rule, checked mechanically): founder_constraint needs one of three sources:
- A recorded founder technology preference, named in the ADR.
- The founder-confirmed CONOPS concept. Code inserts mandatory executes_concept_ids from the fixed slot; choose only optional concept links.
- A decided founder-register item with decision and reason, linked by exact founder_decision_refs id.
For a register source, select its exact ref and explain how this ADR implements that constraint.
Do not copy its sentence, source snapshot, revision or hash: code inserts the exact recorded source.
This is a system architecture choice, not a founder-decided architecture. The source only constrains
what it actually says; it does not approve technologies or implementation choices it never names.
Source support is a separate human review question, not proved by a quote.
Open, deferred, unknown or unrelated register items are not sources.
Use [] for founder_decision_refs when this ADR does not use a register decision. Never invent a link.
Without one of these sources use the real non-founder driver, not founder-mandate language.
The rationale is load-bearing - why this option won. driver is exactly one of
founder_constraint - the founder mandated/confined it, cite the preference; ecosystem_maturity;
community_adoption; team_fit; simplicity), options_considered (the real leading alternatives
you rejected, each with one sentence on why it lost - not strawmen; empty only when a founder
mandate genuinely leaves no alternative), baseline_refs supporting it, and - ONLY for a
migration ADR - evidence (concrete justification for departing) and affected_repositories
(baseline repository names that change). Non-migration ADRs leave evidence and
affected_repositories empty.
FORMAT CONTRACT (final instruction - overrides anything above that conflicts): return one
STRICT JSON object {title, decision, rationale, driver, options_considered, baseline_refs,
evidence, affected_repositories, executes_concept_ids, founder_decision_refs} - refs and repository names only from the supplied lists."""

ARCH_NFR = """Write ONE non-functional position: the checkable stance this architecture takes
for the given non-functional requirement (how the design makes the requirement observable -
budgets, mechanisms, limits), referencing the requirement id without restating its text.
FORMAT CONTRACT (final instruction - overrides anything above that conflicts): return one
STRICT JSON object {position: string}."""


TOOL_PROMPTS = {
    # ADR-0020 escape hatch: where structure IS analysis the model may PROPOSE
    # via typed tools; code disposes. Runs before the prose-fill call.
    "srs": "Here is the requirement skeleton for this system (one slot per need, "
           "ids/kinds/links fixed) with the needs and the intent model. BEFORE prose "
           "is written, you may propose STRUCTURAL changes via the tools - but only "
           "where structure is genuinely analysis: split_requirement when one need "
           "decomposes into 2+ independently verifiable requirements, "
           "link_requirement_to_need when a requirement truly realizes several needs. "
           "Most honest skeletons need NO changes: 1:1 is the norm, splitting is the "
           "exception. If nothing decomposes, call no tools and say so briefly.\n\n"
           "SKELETON:\n{skeleton}\n\nNEEDS:\n{needs}\n\nINTENT:\n{intent}",
    "tests": "Here is the test skeleton (one acceptance slot per SRS criterion, one "
             "unit slot per item spec, ids/links fixed) with the SRS and the item "
             "specs. Act as a QA engineer: each test asserts only what its one "
             "criterion or spec states, with observable expected_outcomes, and no new scenarios, failure modes, "
             "security, encryption, localization, accessibility or latency numbers. Do NOT call add_test unless "
             "a criterion or spec has no test at all and the skeleton cannot cover it. If the skeleton covers "
             "the V, call no tools and say so briefly.\n\n"
             "SKELETON:\n{skeleton}\n\nSRS:\n{srs}\n\nITEMS:\n{items}",
}


