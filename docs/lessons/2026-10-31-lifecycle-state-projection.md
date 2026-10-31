# Read lifecycle state from a durable latest-event projection

Configured host document runtimes now supply recent_lifecycle_events. The host
verifies and consumes only new log chunks and retains the last two matching
events across process restarts. Two events preserve the immediate FINALIZE seal
required by RESUME retry validation. All existing lifecycle transition, content,
anchor and Stop checks remain in place. File mode keeps its existing semantics.

Deploy with the companion delivery-api document projection schema/runtime, after
owner review. Do not silently fall back to reading a host document's entire log
if the required host capability is missing.
