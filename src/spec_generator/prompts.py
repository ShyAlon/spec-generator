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
- Numbering is assigned deterministically after generation; use the placeholder ids given.
- Output STRICT JSON matching the requested schema. No prose around it."""

STAGE_PROMPTS = {
    "needs": "From this intent model, write the stakeholder needs set (N-1..). "
             "The intent model already carries interview-level needs (SN-ids, founder language, "
             "classified functional/non_functional): carry them forward - every SN-id maps to at "
             "least one N-id (intent_need_ids), with kind preserved and the stakeholder named. "
             "Add needs the founder implied but did not state (mark their intent_need_ids []).",
    "srs": "From these needs + intent, author the SOFTWARE REQUIREMENTS SPECIFICATION body: "
           "purpose (why the system exists, stakeholder language), scope_summary (the boundary), "
           "and the requirements as two arrays - functional[] (behaviors) and non_functional[] "
           "(qualities/constraints on behavior: security, performance, privacy, auditability, cost). "
           "Each requirement: requirement_id placeholder (R-1..), title, statement (well-formed "
           "shall-statement, solution-agnostic), priority (must/should/could), kind, need_ids, "
           "intent_fields, and acceptance criteria (AC-1..) that are OBSERVABLE outcomes - "
           "no implementation mechanics, no named technologies, no invented numbers or artifacts. "
           "Non-functional needs must NOT silently vanish: each becomes its own verifiable "
           "requirement, never absorbed into prose. Every requirement traces to at least one N-id.",
    "architecture": "Given the SRS and the target archetype, propose the architecture: stack, "
                    "components, ADRs, non-functionals. Components and decisions REFERENCE "
                    "requirement ids (R-n) - never restate requirement text.",
    "items": "Write item-level specs (SPEC-...) covering every requirement in the SRS. Reference "
             "requirements by id through requirement_ids - never restate or regroup requirement text. "
             "One JSON array.",
    "tests": "Derive the test hierarchy in two halves. acceptance_tests: one per SRS acceptance "
             "criterion (criterion_id set) - HERE you author the executable_hint (http GET ..., "
             "DOM selector, db probe) because the architecture now exists. unit_module_tests: "
             "per item spec (spec_id set). One JSON object: {\"acceptance_tests\": [...], "
             "\"unit_module_tests\": [...]}.",
    "review": "Review pass {n}: find gaps, contradictions, orphan requirements, uncheckable criteria, "
              "solution language in the SRS, downstream text that restates requirements. "
              "Return a JSON list of concrete fixes, or [].",
}
