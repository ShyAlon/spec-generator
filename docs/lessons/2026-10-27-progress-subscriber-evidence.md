# Nonfatal progress failures still need evidence

A broken subscriber must not abort a successful run or stop later subscribers.
Count failures per subscriber index and warn at most once per index per minute.
The diagnostic includes a bounded machine-stage identifier, category and count,
never the event label/detail, exception message or callable repr. Logging failure
is also nonfatal. Generator stop-fence behavior remains unchanged.

Regression tests failed before the fix. They cover continued fan-out, cumulative
counts, rate limiting and resumption, safe single-line context and broken logging.
