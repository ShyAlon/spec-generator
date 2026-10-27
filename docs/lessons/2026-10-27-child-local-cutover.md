# 2026-10-27: local child cutover must carry application state, not closures

Problem: isolated transports alone do not move a graph. The actual durable
request, checkpoint, reviewed input, lease and provider-call cursor must cross
the boundary without lending a live parent's graph or lease.

Fix under review: typed one-step dispatch with immutable owner-bound refs,
compare-and-claim, a private configuration FD, parent-owned raw-result staging,
exact monotonic evidence, and the existing publication recovery owner.

Regression found while implementing: generator saver-latest can advance after
an interrupted node. A call identity keyed to that changing checkpoint can
repeat provider work. Use durable job/generation plus cumulative graph step,
node/stage/ordinal. The new crash-cursor regression failed before the fix.

Local acceptance uses disposable migrated Postgres and fake HTTP only. Never
turn missing completion evidence into a rerun, zero charge or cancellation.
Draft limitations are recorded in phase-4-local-cutover.md. Hosted CI unverified.
