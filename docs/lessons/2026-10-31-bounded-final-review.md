# Final review must not reserialize settled phases

The installed paid-child all-phase acceptance reached validation, but the bounded CLI stopped before content review, so final export had no findings ledger. Running the unbounded tail was unsafe: terminal assembly reserialized settled architecture/tests and the publication substitution guard rejected it.

Add --final-review only with the bounded validation boundary (and optional validation-only redo). It runs the existing content review, retains its findings/history, and stops before repair or assembly. It neither closes findings nor substitutes approved phase outputs. Final handoff stays blocked by open findings.

The same real export found a typed comparison mismatch: a valid semantic candidate omits nested optional defaults, while canonical contract parsing adds them. Compare the candidate through its contract type, still refusing changed content and unresolved decisions.

Validation: bounded-review topology and handoff nested-default/tamper regression, existing resume/lineage/crash tests. Delivery's single-task fake HTTP acceptance reaches authenticated package create/download with hash verification. No paid provider call or activation.
