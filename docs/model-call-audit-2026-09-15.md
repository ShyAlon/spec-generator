# Model-call audit - 2026-09-15

Scope: every model invocation in `interview-engine` and `spec-generator`. Classification:
A = derive by code, B = model with code guard, C = genuinely model-owned language/judgment.

## interview-engine

| Site | Current behavior and inputs | Class | Action |
|---|---|---:|---|
| `graph.py:159,181` probing | Chooses/phrases the next interview question from transcript, coverage and parked topics; retries malformed turn output. | C/B | Kept. Question selection and phrasing are semantic. `TurnOutput` schema, coverage policy, repeat ceiling, source-turn state, checked invocation and one bounded retry own shape and process. |
| `graph.py:289` distillation | Turns founder dialogue into substantive intent fields. | C/B | Kept. Distillation is intrinsic model work. Code owns staged field inventory, enum normalization, provenance checks, delegated/parked decisions, contract validation, deterministic merge and bounded retries. |
| `scope_gate.py:82` scope tiebreak | Only classifies mixed/silent deterministic signals. | C/B | Strengthened now. Deterministic signals still decide clear cases; ambiguous semantics stays model-owned. Added balanced wrapper extraction plus a strict typed result before `ScopeVerdict` construction. |
| `graph.py:613` CONOPS per-field calls | Fills only missing prose in deterministic stakeholder/scenario identities. | B (A already extracted) | Kept derivation-first design from `6dddb21`. Code derives stakeholder/scenario containers, IDs, names and provenance; each model call owns one missing prose field; wrappers, list shapes, nulls, whitelist, provenance and contract are guarded. Concept-option prose remains model-generated because no validated artifact already states alternatives/tradeoffs. |
| `model_config.py:454` structured adapter | Internal adapter delegates to the provider runnable. | Infrastructure | No product call site. It is the constrained-output implementation behind `build_chat_model`, not an additional semantic request. |

## spec-generator

All authoring calls go through `graph.py:_call_json` / `_invoke`, except the optional structural tool loop noted below.

| Site | Current behavior and inputs | Class | Action |
|---|---|---:|---|
| `node_needs` | Reframes validated interview needs as a stakeholder-needs document. | C/B | Kept: wording and grouping are semantic. Code overwrites session identity; strict contract validation and retry own shape. |
| `node_srs` plan/fills | Authors purpose/scope and per-requirement prose. | B | Kept. Code deterministically carries definitions, constraints, assumptions, open questions; builds R IDs, kinds, links and section slots. Model only fills prose, guarded per slot. |
| `node_architecture` plan/entries/components/optional sections/ADRs/NFRs | Chooses architecture and authors rationale from intent/SRS plus real candidate data. | B/C | Kept. Technology choice and rationale require judgment. Code owns component/ADR IDs, slot identities, selected-candidate provenance, trace links and final contract; candidate registry/download ranking is deterministic and never redrafted by the model. |
| `node_items` plan/body fills | Decomposes architecture into item specs and authors implementation text. | B/C | Kept. Decomposition/text are substantive design work. Code owns spec IDs, slugs, ordering, requirement/component links and contract validation; body calls cannot replace identities. |
| `node_tests` plan/body fills | Derives acceptance-test content from requirements and specs. | B/C | Kept. Test definitions require semantic design. Code owns test IDs, levels, criterion/spec links, completeness checks and contracts. |
| `node_review` | Reviews assembled artifacts and proposes bounded fixes. | C/B | Kept. Review is judgment. Code normalizes the fix wrapper, validates paths/operations, and applies only the whitelist; the model never mutates state directly. |
| `tools.py:run_tool_loop` | Optional model proposes skeleton changes via typed tools. | B | Fixed now. Proposal/disposal and typed handlers were sound, but direct `model.invoke` bypassed standard timeout, truncation, telemetry and provider-error handling. Every round now uses `invoke_checked`; code remains sole owner of IDs/links/mutations. |
| `model_config.py` provider `.invoke` / `.stream` | Guard implementation that actually calls the provider. | Infrastructure | No separate semantic calls. `_invoke` uses `stream_checked`; tool loop now uses `invoke_checked`. |

## Failure-species closure

- Dict constraint / shape drift: authoring outputs are contract-validated; pipeline-owned dictionaries and containers are rebuilt by code. Tool proposals pass typed handlers.
- Wrapper leak: structured authoring uses balanced JSON extraction. Scope tiebreak was the remaining ad-hoc parser and now uses that extractor plus typed validation.
- Null repair fields: CONOPS validated substance is derived before generation; model calls are one field at a time, normalize wrappers/list shapes, reject null, whitelist the requested field and preserve code-owned identity/provenance. Spec authoring validates every output and does not accept nullable pipeline identities.
- Unguarded provider invocation: all production semantic calls now enter `invoke_checked` or `stream_checked`. Provider calls inside those functions are the guard implementation itself.

No additional A-class conversion was made: the remaining model-owned values are distillation, question phrasing, prose, decomposition, architectural/test judgment or ambiguous classification. Replacing those with copied artifact text would lose the requested synthesis rather than remove model-generated data.
