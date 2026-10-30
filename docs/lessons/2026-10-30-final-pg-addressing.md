# Remove backend-shaped addresses after retiring the backend

The PG cutover removed the file reader but left a dummy checkpoints.sqlite name
in revision/rework/preflight and kept old driver packages in uv.lock. Host workflow
addresses now use owned directories directly, preserving canonical-path, namespace,
review-phase and exact-checkpoint checks. The delivery integration test uses the
same directory contract. Merge this change before the delivery cleanup PR.

Keep historical lessons and the technology catalog. Legacy private-file ignore
patterns remain denial guards, not active persistence or migration support.
A stale Linux-only skip on a pure lineage test was removed, widening coverage.

Validation: complete spec suite, 1,113 passed without skips. Hosted CI is unverified.
