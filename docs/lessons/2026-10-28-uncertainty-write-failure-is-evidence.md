# Uncertainty write failure is evidence

Preserving the original provider error does not require silently losing the
secondary ledger outage. Record safe error class at that boundary, never attempt
payload, credentials or exception text. A diagnostic sink failure must not replace
the original error either. Never infer refund or retry from missing ledger state.
