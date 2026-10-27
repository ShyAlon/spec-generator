# Phase 4 draft: explicit local child cutover

2026-10-27. Work in progress, not ready to merge or activate.

Defaults remain unchanged. DELIVERY_CHILD_EXECUTION=local also requires
DELIVERY_LOCAL_PREVIEW=1 and an Ollama-only reviewed gateway. Paid child
execution remains refused. No deploy, restart, live provider call or live DB
schema/data access was performed. Hosted CI and Docker execution are unverified.

Delivery owns provider dispatch/ledger, immutable operation inputs, publication
and recovery. Fresh children receive bounded configuration through a private FD,
not credential-bearing argv or environment. Typed interview start/answer and
generator phase operations compare-and-claim a new lease, publish durably and
release it before terminal lifecycle success. Parent reads remain claimless.
One shared local slot includes reap and broker reconciliation. Parent pool 8
plus active child 4 measured at saturation, compatibility service 0, total 12.

Local tests exercise real graph start and answer in separate PIDs, real generator
Needs and SRS/SSE against fake HTTP, Stop, Auth revocation, SIGKILL after dispatch,
no blind replay, immutable refs, durable gate answer restart, publication-only
recovery with no spawn, graph cursor stability and 8+4 pool accounting.

Pending in this draft: model-route changes and question-quality config writes in
child mode, expanded recovery acceptance and final review. These currently fail
closed instead of silently falling back to an in-process graph. Paid readiness
is not claimed. Shy owns merge; no merge request until these drafts are ready.
