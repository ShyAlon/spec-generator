# Exact source citations need actionable repair feedback

The retained ADR-1 outputs both named the correct decided founder-register ref
but paraphrased its recorded no-op decision. The exact-citation validator
correctly rejected them. The existing body prompt already asked for a verbatim
quote, but the one repair message repeated only the ref and generic rejection.

Keep citation validation strict. Include the missing ref and its exact recorded
decision as explicitly labeled source data in the existing repair message. Let
the model return a corrected complete body; do not append a citation for it,
infer source applicability, erase the ref, or add another retry.

Regression evidence retains both rejected bodies and the minimal source fields.
Tests reproduce rejection, accept only the exact quotation, reject changed or
unrelated wording, and drive a fake streaming model through the real repair
message. Unknown, undecided and reasonless sources remain invalid and do not
receive a repair quotation. Only refs that failed the citation check enter the
feedback.

After adding only the exact source sentence, both retained bodies also passed
the existing full ADR-body validator offline. Attempt1's simplicity driver is
not a subsequent contract failure. No driver rule needs changing for this case.

Evidence limit: the debug export retains rejected outputs, but its Intent/ConOps
are current-at-export rather than a historical failure-time prompt snapshot.
