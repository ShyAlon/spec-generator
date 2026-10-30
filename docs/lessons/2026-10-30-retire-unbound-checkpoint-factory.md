# Retire the standalone checkpoint fallback, keep the host API

Delivery's bound child imports cli.main. It supplies authenticated worker
binding, PG checkpoint factory, document sink, broker and stop fence. Retiring
standalone runs must not delete those imported helpers or close the host saver.

checkpoint/read_state refuse an unbound runtime before opening anything. CLI
main rejects unbound execution before parsing commands or calling a provider.
worker_graph, revision_worker and SRS rework use only the host factory; local
connection imports/close finalizers are gone. Namespace ownership and lineage,
revocation, stop, source-immutability and fresh-review checks remain.

Tests own a deliberate MemorySaver host fixture for graph/command behavior.
That is not an alternative supported durable runtime. The 9 revision-worker
executions retain all behavior assertions, now comparing exact source saver
state instead of file bytes. Crash recovery compares saver checkpoints/writes;
model counts, lifecycle, gates and no-repeat assertions remain. CLI argv/error,
resume/version, replay, handoff and architecture cases run with explicit host
bindings rather than silently bypassing the new guard. No test removed/skipped.
The only retired coverage is execution against the removed unbound local saver.
Real PG host persistence/isolation remains delivery-api integration coverage.

The first unbound checkpoint/CLI tests failed before the change. Full suite:
1120 passed, no skips/deselections. Earlier intermediate host fixture failures
were fixed by passing explicit models/settings and sharing the host saver.
One full attempt crashed with Python Bus error while reporting a pinned-reader
failure. Its temporary fixture had not closed its connection. Closed it and
reran: pinned reader 10 pass and full suite 1120 pass. Do not call that failed
attempt a green run.

revision_source and its 10 pinned-file tests remain for now: delivery main still
imports that module until delivery #515 merges. Do not break a cross-repo import
or claim all SQLite code is removed. The test-only dependency and other legacy
saver fixtures are a separate retirement hop. No live writes/reset, deploy,
spend, auth policy change or flake-tracing removal. Hosted CI unverified.
