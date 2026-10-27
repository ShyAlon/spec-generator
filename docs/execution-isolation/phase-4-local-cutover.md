# Phase 4: explicit local child cutover

2026-10-27. Opt-in implementation; no activation performed.

Defaults remain unchanged. DELIVERY_CHILD_EXECUTION=local also requires
DELIVERY_LOCAL_PREVIEW=1 and an Ollama-only reviewed gateway. Paid child
execution remains refused. No deploy, restart, live provider call or live DB
schema/data access was performed. Hosted CI and Docker execution are unverified.

Delivery owns provider dispatch/ledger, immutable operation inputs, publication
and recovery. Fresh children receive bounded configuration through a private FD,
not credential-bearing argv or environment. Typed interview start/answer/route and
generator phase operations compare-and-claim a new lease, publish durably and
release it before terminal lifecycle success. Parent reads remain claimless.
One shared local slot includes reap and broker reconciliation. Parent pool 8
plus active child 4 measured at saturation, compatibility service 0, total 12.

Local tests exercise real graph start and answer in separate PIDs, real generator
Needs and SRS/SSE against fake HTTP, Stop, Auth revocation, SIGKILL after dispatch,
no blind replay, immutable refs, durable gate answer restart, publication-only
recovery with no spawn, graph cursor stability and 8+4 pool accounting.

Local route changes now run as a separate typed child operation. The child
atomically validates the published checkpoint, sequence and input, then uses
the engine's existing switch logic without invoking a provider. Durable retries
return the original route event without another child. A subsequent answer
uses the new model through the broker. Quality edits validate through the engine
ConfigStore, write atomically under the artifacts base, and snapshot into each
new child. Reads and writes stay in the parent control plane; graph work does not.

Validation: 227 delivery selected tests, 43 engine tests, 55 generator tests,
no skips in those selections. All three project wheels build with uv. Shared
contract 0.3.0 has identical bytes in all three repos; wheel source/distribution
tests and dependency locks are checked. Hosted CI, Docker image execution and
live-provider/production-DB behavior remain unverified. Two earlier opt-in test
debts remain separate, not fixed or hidden here. Paid readiness is not claimed.
Shy owns merge. This phase stops here; no deploy, restart or next phase.
