# Parent provider broker adapter (phase 3)

Explicit wire_binding.broker_clients installs only the shared private-descriptor
SDK clients. It does not enable execution or create an endpoint/credential.
Stable logical call IDs are supplied by trusted durable-operation wiring.
The child sends exact SDK JSON bytes plus stage/model identity; parent alone
validates, reserves, dispatches, persists raw bytes/usage, settles and replies.
Unknown outcomes close/poison the channel, no reconnect or automatic replay.

The default local/in-process wire is unchanged. Never nest broker and local
wire scopes. Registered models are scoped to that broker; mismatches refuse.
No child settle/release API and no provider keys are needed. Provider waits
cross the separate sequenced compute channel; missing finish stays unresolved.
Paid-child refusal is still retained by delivery. Full app cutover is phase 4.
Hosted CI, live models and Docker execution are unverified.
