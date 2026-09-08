"""Generation prompts. The methodology context is stable across runs and
across stages: put it first and cache it (prompt caching is the single
biggest generation-side lever in the cost model)."""

METHODOLOGY = """You are generating a systems-engineering specification following the V model.
The artifact hierarchy: stakeholder needs -> system requirements -> architecture ->
item-level specs -> test hierarchy (unit, module, system, acceptance) with full traceability.

Authoring rules (these are what separate this from a free PRD generator):
- Every requirement traces to at least one stakeholder need. No orphans.
- Every acceptance criterion is MACHINE-CHECKABLE: an HTTP status, a DOM assertion,
  a CLI result, a schema conformance. Write 'GET /health returns 200', not 'system is healthy'.
  Machine-checkable criteria are what make cheap deterministic verification possible.
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
    "requirements": "From these needs + intent, write system requirements (R-1..) with acceptance "
                    "criteria (AC-1..). Every requirement carries kind: 'functional' (a behavior) or "
                    "'non_functional' (a quality/constraint on behavior - security, performance, "
                    "privacy, auditability, cost). Non-functional needs must NOT silently vanish: "
                    "each becomes its own verifiable requirement (e.g. 'User data shall be encrypted "
                    "at rest' with a checkable criterion), never absorbed into prose.",
    "architecture": "Given requirements and the target archetype, propose the architecture: stack, components, ADRs, non-functionals.",
    "items": "Write item-level specs (SPEC-...) covering every requirement listed. One JSON array.",
    "tests": "Derive the test hierarchy (T-1..): acceptance tests for every AC, plus unit/module/system tests per item spec.",
    "review": "Review pass {n}: find gaps, contradictions, orphan requirements, uncheckable criteria. Return a JSON list of concrete fixes, or [].",
}
