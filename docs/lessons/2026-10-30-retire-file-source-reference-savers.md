# Retire the pinned file reader after its caller replacement merges

Verified delivery #516 head cf19c191faa19e8353bd21c826c797c108947fdd is
an ancestor of freshly fetched main 90ce78728c4d8f5d396c45a799c4786df8f0fc2c.
No delivery main Python import of spec_generator.revision_source remains.
Spec #68 host-checkpoint replacement was already merged. Only then remove the
obsolete file reader and its runtime dependency.

The absence test fails before removal. Retired test_revision_source.py contains
10 expanded executions: exact state/read-only/no-new-sidecars; four sidecar or
source-symlink refusals; namespace swap; same-inode modification; ancestor
symlink; exact edited staged gate; arbitrary checkpoint gate refusal. These
file descriptor, inode, sidecar and path guards are retired with the file
capability, not claimed retained in PG. PG source snapshot/gate identity and
lineage tests in merged delivery #516 remain the durable authority.

Reconstructed the lost reference-saver follow-up: graph-only fixtures use a
per-test keyed MemorySaver registry. Preserve all graph/resume/gate/rewind,
serialization, provider-zero, fence/digest/approval assertions. Reset before
and after each test. A direct registry identity/key/reset test covers sharing.
Fresh graph instances share a reference saver, not a file connection. These
fixtures no longer test cross-process or on-disk durability. Delivery's real
PG lifecycle suites own that behavior. Never claim MemorySaver is production
persistence. Preserve the literal checkpoints.sqlite file-copy exclusion test:
that filename is a denied stale artifact, not a persistence implementation.
Technology candidate catalog entries are product content, not a storage driver.

Focused gate: 140 passed. Full suite: 1112 passed, no skips or deselections
(1120 prior executions minus 10 retired plus 2 new). Hosted CI unverified.
No live DB mutation/reset, deploy, spend, auth change or flake tracing removal.

After reference-key naming cleanup, a mechanical variable rename produced one
NameError. Fixed the mismatched key and reran the full suite: 1112 passed.
Delivery revision/source/stage/manifest integration: 37 passed.
