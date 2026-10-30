# 2026-10-30 - Preserve exact published inputs and real checkpoint owners

A PG baseline already at the output document key must keep its exact approved
bytes. Revalidating it is correct; normalizing and rewriting it silently changes
settled evidence. A focused regression preserves non-normalized valid JSON.

LangGraph versions_seen can include __interrupt__ bookkeeping beside the real
latest graph node. Exclude internal bookkeeping before choosing as_node, while
still refusing genuinely ambiguous real-node owners. The regression uses
LangGraph's bookkeeping helper rather than an invented metadata shape.

Local full suite: 1,115 tests passed after these changes. Hosted CI is unverified.
