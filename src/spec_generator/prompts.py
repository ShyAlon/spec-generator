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
- The executable HOW (http GET /health, DOM selectors, db probes) is authored at
  the TEST stage, where the architecture exists - never in the SRS.
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
           "boundary), and for EACH slot: title, statement (well-formed shall-statement, "
           "solution-agnostic), priority (must/should/could), and acceptance criteria as plain "
           "statements that are OBSERVABLE outcomes - no implementation mechanics, no named "
           "technologies, no invented numbers or artifacts (ids are assigned by the pipeline). "
           "purpose and scope_summary describe THE FOUNDER'S PRODUCT, in stakeholder language, "
           "from the INTENT below - never the V model, this pipeline, or these instructions. "
           "Non-functional slots must NOT silently vanish into prose: give each its own real, "
           "verifiable requirement text. Return one JSON object: {purpose, scope_summary, "
           "fills: [{requirement_id, title, statement, priority, acceptance: [statements]}]} "
           "- exactly one fill per given slot.",
    "architecture": "Given the SRS and the target archetype, propose the architecture: stack, "
                    "components, ADRs, non-functionals. Components and decisions REFERENCE "
                    "requirement ids (R-n) - never restate requirement text. Do NOT number the "
                    "decisions: ADR ids are assigned by the pipeline in order. "
                    "Every decision MUST carry: driver - exactly one of founder_constraint (the "
                    "founder mandated/confined it; cite the preference), ecosystem_maturity, "
                    "community_adoption, team_fit, simplicity - and options_considered: the real "
                    "leading alternatives you rejected, each with one sentence on why it lost "
                    "(not strawmen; empty list only when a founder mandate genuinely leaves no "
                    "alternative). No decision is arbitrary.",
    "items": "Write item-level specs (SPEC-...) covering every requirement in the SRS. Reference "
             "requirements by id through requirement_ids - never restate or regroup requirement text. "
             "One JSON array.",
    "tests": "Author the test prose. The skeleton is FIXED: one acceptance-level slot per SRS "
             "acceptance criterion (criterion_id linked) and one unit-level slot per item spec "
             "(spec_id linked), test_ids already assigned - never change, renumber, add, or drop "
             "them. For EACH slot write description, runner ('http' | 'dom' | 'cli' | 'static' | "
             "'semantic'), and definition (the runner-specific payload: URL + expected status, "
             "selector, command, ...) - HERE you author the executable HOW, because the "
             "architecture now exists. Return one JSON object: {fills: [{test_id, description, "
             "runner, definition}]} - exactly one fill per given slot.",
    "review": "Review pass {n}: find gaps, contradictions, orphan requirements, uncheckable criteria, "
              "solution language in the SRS, downstream text that restates requirements. "
              "Return a JSON list of concrete fixes, or [].",
}


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
             "specs. You may propose ADDITIONAL tests via add_test - only for real "
             "gaps the skeleton cannot cover: a module/system test spanning several "
             "specs, or an acceptance angle with no criterion. If the skeleton covers "
             "the V, call no tools and say so briefly.\n\n"
             "SKELETON:\n{skeleton}\n\nSRS:\n{srs}\n\nITEMS:\n{items}",
}
