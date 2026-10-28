# Nonfatal log loss is still evidence

A call-log sink failure used to disappear behind except OSError: pass.
Keep telemetry failure nonfatal, but emit one bounded warning per error category
in the process. Never log the destination path, event, exception text or model
content. The full service routes this warning into its operator diagnostic file;
standalone commands use their configured warning logger. A restart resets the
suppression, so missing older evidence must remain explicit in Debug exports.
