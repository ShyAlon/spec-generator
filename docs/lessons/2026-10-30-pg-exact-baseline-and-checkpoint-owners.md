# 2026-10-30 - Preserve exact published inputs and real checkpoint owners

A PG baseline already at the output document key must keep its exact approved
bytes. Revalidating it is correct; normalizing and rewriting it silently changes
settled evidence. A focused regression preserves non-normalized valid JSON.

LangGraph versions_seen can include __interrupt__ bookkeeping beside the real
latest graph node. Exclude internal bookkeeping before choosing as_node, while
still refusing genuinely ambiguous real-node owners. The regression uses
LangGraph's bookkeeping helper rather than an invented metadata shape.

Local full suite: 1,115 tests passed after these changes. Hosted CI is unverified.

A host worker's own UsageLog does not prove the default CLI sink is PG-backed.
The CLI constructs its own jsonl_sink, so that shared sink must recognize the
bound document runtime too. Test two exact rows with no local copy and read the
same bytes through a fresh real PG document reader. Plain run logs and raw
rejected/fitted diagnostics still need the separate diagnostic-storage audit;
canonical/checkpoint parity is not proof that every diagnostic file is gone.

Follow-up: bound generator rejected/fitted raw bytes, their manifests, per-call
metadata and plain run logs now use the document backend too. Keep exact raw
whitespace, collision-safe names and ANSI-free plain logs. Legacy standalone
sinks stay unchanged. Fail-first sink regressions and a real PG readback cover
this addition; the interview diagnostic sinks are a separate outstanding audit.
