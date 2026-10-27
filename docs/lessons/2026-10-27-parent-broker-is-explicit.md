# Keep parent broker binding separate from local provider wiring

A private child SDK transport must not inherit endpoint/key or settle/release
power. Use the shared pinned broker transport only under an explicit scope;
keep local/in-process defaults unchanged. Stable logical IDs come from durable
execution, not random retries. A dead channel stays dead until reconciliation.
Model registrations belong to the broker scope and exact stage/provider/model.
Engine's per-job lazy cache also needs that scope identity, or a later child
can retain the first child's model/descriptor. That regression failed first.
Hosted CI and complete production cutover remain unverified.
