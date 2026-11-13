# Citation-copy measurement

## Measured before (retained debug export)

- Two ADR-1 quote-copy rejections, attempts1 and2. The second is the existing automatic repair, not another new retry.
- Rejected call durations27.30s and27.81s in the rejection manifest. These are recorded call durations, not stage wall time.
- Architecture failed after1236.7s in spec/run.log. The export contains28 architecture usage rows and13 SRS rows. Counts alone do not prove unnecessary calls.
- The source/export is redacted and is not an exact historical prompt snapshot. Token counts are redacted. Do not calculate per-retry dollar savings.

## Measured offline after

The two retained model answers independently pass source-identity validation in one fake-model call each. The server inserts the recorded source snapshot without changing their rationale. Both still receive a support-not-yet-reviewed finding. This proves elimination of the sentence-transcription retry condition, not that the architectures are semantically sound or that a live run is faster.

A fake-model graph reaches the existing architecture review with those findings and no new model stage. Human finding disposition allows completion; an unreviewed or changed source/ADR cannot reuse that disposition at assembly or package export. Tests cover an unchanged resume and a changed source, rationale and concept link.

## Owner-run after measurement

Run scripts/citation-metrics.py on the existing debug export and on the next owner-run export. Compare citation rejection counts, repair rejections, usage rows by stage, recorded rejected-call durations, and available stage elapsed times. The script reads existing records only and emits no prompts or source text. Missing evidence is null, never zero.

Record model/provider, generator/contracts revisions, route, input source revision, resume point and whether upstream work was reused. Compare equivalent runs, not a resumed ADR against a full fresh architecture. Expected: zero sentence-copy rejection reasons and zero added support-review calls. Any retry due to unknown refs, bad structure or invalid source remains legitimate and separate.

The removed retry is an estimated future saving until an equivalent live owner-run is measured. No live model call was made for this PR.
