# Recovery ledger

The frozen wheel scaffold is not an authoritative substitute for remote main. New commits preserve the existing main ancestry and recovered later work.

## Main architecture reconciliation
Conflict: frozen v3 omitted main's validated_architecture, technical-choice defaults/continuation metadata, founder ADR source checks and plain-call DeepSeek thinking policy. Resolution: restore current main architecture_decisions/prompts/schemas/render, graph and CLI architecture behavior, retaining only later recovered Stop guards/publication adapters in those files. Keep recovered model runtime/admission code while restoring main's all-call thinking-disabled behavior. No implicit legacy healing remains in graph/CLI. Example baseline stamp moved to the recovered current contract0.28.0, not silently migrated user content.

Validation: full current tracked suite767 passed. Additional lost-unit test asset recovery and wheel packaging comparison remain open; this is not yet full pre-loss equivalence.

## Admission/runtime/wire regression tests
Recovered missing initial attempt admission/wire/runtime fixtures and chronological transport tests from archived source writes. Preserved real SDK serialization and fake-only transport assertions. Expanded suite810 passed; no production change in this test unit. Further Stop and publication tests/build assets remain open.

Recovered initial Stop fence tests; current shared same-host writer is asserted STOP_REQUESTED while active, not falsely quiescent in a second reader. Targeted Stop tests passed (engine4, generator6 including publication/late-thread guards). Further chronological Stop test additions remain to recover.

## Chronological Stop regression recovery
Recovered later source-written Stop/registry/publication test additions. Duplicate writer expects current distinct SessionWriterBusy exception; no false Stop completion. Full expanded suites at this checkpoint: engine643 passed,2 warnings; generator816 passed. These are reconstructed current suites, not an assertion of complete pre-loss equivalence.

## Recovery dependency pin and lock
Restored explicit contracts0.28.0 metadata/UV local-wheel source from archived Oct9 pin preparation semantics, but built a new wheel from current recovered contracts5bdc1a0, with fresh SHA provenance in vendor/RECOVERY-WHEEL.md. Old wheel records retained as history. Regenerated uv.lock without blanket upgrades; uv lock --check passed. Offline project wheel builds passed; all three metadata files require contracts==0.28.0. Frozen-v3 artifacts untouched. Fresh installed rebuilt-wheel UI verification remains the next check, not claimed done.

## Later Stop/publication and explicit worker admission tests
Recovered exact chronological Oct9 07:37,07:38,07:50,08:05 Stop stream draining, raw/progress suppression, adapter accounting, real graph checkpoint, CLI docs/audit/partials/lifecycle tests plus08:10 explicit required admission context test. One archived usage assertion expected no total_tokens; current main normalizes total, so assertion reconciled to existing documented normalized value5, production unchanged. Targeted Stop/admission suite17 passes. Wider suite remains to re-run after this inventory expansion.

## Recovered SRS producer rework test file and strict-main fixture conflict
Exact initial/model/Stop test bodies from Oct9 08:01:54,08:02:24,08:03:21 recovered. Archived minimal intent fixture omitted six fields now required by main's strict IntentModel path through srs_document_data. Expanded the fake fixture with explicit target users/value proposition/scope and a typed fake stakeholder need with explicit source-turn reference; did not change production validation or invent a compatibility API. The first expanded fixture attempt still failed (2 failed/828 passed) because stakeholder_needs cannot be empty. Prior success wording was premature and corrected in an additive commit; retest result follows below. Original archived fixture failure preserved in this ledger as a newer-main conflict, not worked around through healing.

Corrected result verified from completed commands: SRS rework7 pass; expanded generator830 pass. Fake stakeholder need now meets strict main contract. These results supersede91eecbb's premature green text. Production validation unchanged.

## Post-recovery Gate/Edit sub-unit: strict phase seed and fresh checkpoint
Added revision_checkpoint preparation for SRS/architecture only. It validates current-version typed intent/needs/baseline/SRS, session and need/requirement trace references; architecture uses main validated_architecture and baseline/session checks. New seed retains approved upstream state, invalidates downstream state/old affected feedback and semantic approvals, injects exact phase-scoped founder guidance, and preserves reviewed prior architecture only where the existing feedback prompt needs it. Unsupported intent/ConOps/needs/specs/tests/validation adapters fail explicitly.

Fresh checkpoint helper requires bound cooperative execution, exact reviewed source checkpoint identity and exact session configuration, rejects any nonempty destination, seeds without invoking the producer, and checks durable guidance readback. Real LangGraph/MemorySaver with FencedCheckpointer fake test proves the following invocation consumes scoped guidance while source state stays unchanged. Current-version baseline fixture created explicitly; old-version rejection tested, no source restamping.

Completed targeted16 tests and full generator846 tests pass; git diff check passed. This is private proposal preparation only, not source checkpoint mutation, artifact commit, CAS authority, restart dispatch, approval or a public route. Caller-owned filesystem namespace/fenced graph integration, output typed/trace validation, current artifact/checkpoint evidence CAS and durable restart intent remain next. Draft-only Save and cost acknowledgement behavior unchanged.

## Gate/Edit sub-unit: regenerated typed/trace proposal validation
Added non-publishing validate_revision_output for SRS/architecture. Revalidates source seed, rejects unreviewed upstream drift, uncleared phase feedback and carried/generated downstream artifacts; validates current contract, session/need/requirement links. Architecture calls main's baseline reconciliation and founder-decision driver/source checks on both source and proposal; legacy parallel stack input rejected without healing. Returns normalized artifact digest and fresh-review requirement, never acceptance/commit/dispatch. Targeted30 and full generator860 tests completed and passed; diff check passed. Owned filesystem/fenced graph runner, atomic evidence CAS and durable restart intent remain open.

## Gate/Edit private proposal runner
Added run_revision_proposal: requires bound job runtime/captured task-owner-generation-root, explicit phase model and exact same-session Stop marker. Reads exact source checkpoint using read-only SQLite, seeds unique fresh checkpoint under owned spec/revision-proposals, builds Stop-fenced graph stopped at edited phase with no live artifact persistence/old partial reuse, validates output, rechecks source and authority, and retains fsynced proposal metadata. No public route, source mutation, acceptance or restart. Late revocation/Stop prevent proposal publication. Targeted35 and expanded full865 tests completed/pass; diff check clean. Source SQLite fixture initially lacked explicit checkpoint_ns and was corrected, not production API invented.

Path resolution/namespace checks are defenses, not atomic filesystem evidence CAS or full race-safe descriptor containment. Proposal helper remains private and requires the trusted same-host writer to serialize source work. Durable source artifact/checkpoint/generation CAS and reviewed commit/restart integration remain open. No gate/Save/cost acknowledgement behavior changed.

## Gate/Edit real-generator feedback consumption check
Added fake-provider test using real build_graph plus real source/destination SQLite checkpoints. It verifies edited SRS guidance reaches the model prompt and run stops at edited phase. This caught graph channel versus serialized artifact shape: new graph SRS body has no contract_version until normal contract serialization. Runner now serializes only freshly generated unstamped body through current SRSDocument, not source inputs or explicitly old-stamped documents; strict source/output compatibility checks remain. Targeted36/full866 tests completed and passed, diff check clean. No actual provider calls, source mutation or approval. Atomic evidence CAS/durable restart still not implemented.

## Gate/Edit source checkpoint descriptor containment
The closed-revision worker now reads the exact source SQLite checkpoint through a no-follow ancestor walk and pinned regular-file descriptor. Read-only immutable SQLite is restricted to closed sources with no WAL/SHM/journal sidecars; active sidecars fail closed and require a separate quiescent snapshot. Before/after inode, size, mtime and ctime checks plus namespace re-open reject changed contents or swapped directories. Existing writer-lease requirement remains; this is not a replacement for writer exclusion or an atomic filesystem/DB transaction. Linux proc descriptor support is required, otherwise no proposal is produced.

Targeted44 pass; full generator874 pass, diff check clean. Added source-preservation, sidecar/symlink refusal, ancestor symlink, directory swap and same-inode-change regressions. Real graph revision tests still pass. No live artifact/session/generation mutation, deployment, Docker or gate lifting.

## Gate/Edit semantic lineage checkpoint seed
Private lineage adapter now separates immutable semantic contract validation from a fresh execution checkpoint thread. Explicit parent/next generation and identities required, original source contracts validated without restamping, destination must be empty, cooperative fence generation must match planned execution generation and guidance/semantic readback must match. Caller still owns authenticated lineage and destination namespace containment; this helper installs no session binding, artifacts or active generation. Existing proposal worker behavior unchanged.

Targeted53 pass; full generator883 pass; diff check clean. Tests cover preserved source bytes/IDs in a different thread, no old-thread state, wrong lineage/source/generation, populated destination and wrong execution fence. This is a seed adapter, not a promotion journal or reviewed generation installation. Public commit/restart stays disabled.

## Private reviewed-output checkpoint staging
Added isolated staging helper that revalidates semantic lineage/source and typed proposal output, matches regenerated artifact digest to lineage, rejects unreviewed expert decisions and seeds only the cleared approved source plus newly validated phase artifact into an empty execution thread. Unknown proposal channels/pending writes/old checkpoint metadata are not copied. Reads staged artifact/upstream/guidance back. Graph is never invoked, no phase accepted and fresh gate review remains required.

Targeted lineage checkpoint11 pass; full generator885 pass, diff check clean. Tests prove no producer callback/graph invocation, retained semantic ids, no destination write on mismatched lineage digest. Trusted caller must provide pinned source/proposal state and namespace containment. This helper is staging, not filesystem promotion, installed execution authority or a replay mechanism. Existing source run is unchanged; public commit/restart off.

## Staged checkpoint must stop at fresh edited gate
Follow-up tightening: staging now verifies the compiled graph has the edited phase's mandatory review edge and readback schedules exactly gate_srs or gate_architecture, never a downstream producer. Real build_graph(review=True) test proves fresh SRS gate boundary without invoking a model; a graph that bypasses review fails before destination seed. This closes a private-helper misuse gap in the preceding staging unit, not a public runtime incident.

Targeted12/full886 pass, diff check clean. No installed execution session/generation, graph invocation or gate lifting. Staged namespace/caller authority obligations unchanged.

## Private execution-anchor StopFence binding
StopFence optionally requires typed authenticated installed anchor and pinned unchanged bytes. Initializes later execution generation from exact binding, keeps Stop as one-way requested marker at anchor generation+1, rejects stale/false/conflicting/removed markers and changed/missing anchors. An anchor namespace without explicit binding cannot fall back to legacy generation1. Existing non-lineage behavior unchanged. No production caller switched, writer/installation route or public restart added; installation must authenticate expected binding separately.

Full generator896 pass with current contracts source; targeted prior21 before extra fallback regression.

## Coordinated anchor contracts wheel refresh
Rebuilt vendor contracts wheel from contracts95b41c1 with private execution-anchor module. SHA256 bc8cb30c63719e106451d514532b28a30d8e69fbcce8a771121eab2369f0391e, version0.28.0 unchanged. Offline uv lock regeneration/check and project wheel build passed, diff check clean. This records build/lock checks only; clean installed current four-wheel context verification is next, not claimed complete. Frozen v3 untouched, no deployment. Existing PR description correctly records pre-refresh packaged limitation at creation time.
