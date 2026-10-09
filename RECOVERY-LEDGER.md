# Recovery ledger

The frozen wheel scaffold is not an authoritative substitute for remote main. New commits preserve the existing main ancestry and recovered later work.

## Main architecture reconciliation
Conflict: frozen v3 omitted main's validated_architecture, technical-choice defaults/continuation metadata, founder ADR source checks and plain-call DeepSeek thinking policy. Resolution: restore current main architecture_decisions/prompts/schemas/render, graph and CLI architecture behavior, retaining only later recovered Stop guards/publication adapters in those files. Keep recovered model runtime/admission code while restoring main's all-call thinking-disabled behavior. No implicit legacy healing remains in graph/CLI. Example baseline stamp moved to the recovered current contract0.28.0, not silently migrated user content.

Validation: full current tracked suite767 passed. Additional lost-unit test asset recovery and wheel packaging comparison remain open; this is not yet full pre-loss equivalence.
