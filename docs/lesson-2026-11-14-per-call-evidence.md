# Preserve code-owned call labels without changing call identity

The parent broker can export token counts and charges, but cannot reliably infer an ADR/item label from a prompt or timestamp after the call. Pass the existing optional `item` label through the checked invocation/stream and the child call context instead. Labels that are absent or outside the bounded diagnostic alphabet remain unknown; fields are empty because a generator item label is not proof of a particular field.

The cursor hash still uses exactly the old session/job/step/node/stage/ordinal inputs. Diagnostic labels do not create new calls or change replay identity. No prompt, model choice, validation, automatic repair, cost policy, or provider dispatch changes. The exact child-contract pin becomes 0.4.0; vmodel-contracts remains 0.29.0.

Local proof: 1134 tests passed. One unchanged baseline failure remains in `test_bound_lifecycle_lineage_and_docs_have_no_file_authority`: its Documents fake lacks `recent_lifecycle_events`. The same test fails on clean main 6889900b3ebc63fba2bb50cc97992baf74581889 with the same error. No assertion was removed or weakened. Hosted CI and a live paid run are not claimed.

Merge with the delivery and interview-engine companions and rebuild the combined runtime only after the active owner run has settled. The new parent accepts the optional metadata; an old parent does not. Historical missing item labels are not reconstructed.

Lesson: the human-readable item label belongs beside, not inside, the existing durable call identity.
