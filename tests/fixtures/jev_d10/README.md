# JEV-D10 frozen distance fixture

- `distance-corpus-v1.json`: 28 pair cases (13 echo, 15 duplicate). Echo: A is the control directive, B the candidate. Duplicate: A and B are two spec bodies. Provenance per case is in each `source` field (real SPEC bodies from the failed Duolingo run, documented synthetic variants, authored calibration cases).
- `distance-labels-v1.json`: the founder's yes/no labels (zero unsure) exported from the labeling tool on 2026-10-01.
- The labels file declares `corpus_sha256` 2b20978f4d95f6ffc55e4be1454664ad36b41e5123290008dae0573e0294fb50. The labeler copies that digest as a literal; it does not compute it, and the raw bytes of this corpus file hash to 7be8e9b4b0ee8860f393728152c3564dc7f99776f424a327cfc498f83430ac52. All 28 case IDs and texts were compared with the corpus embedded in the labeler and match. `tests/test_content_acceptance.py` pins the raw-byte hash of this file, so any edit is a visible change.
- Treat the thresholds calibrated on this set as provisional. It is a small designed fixture.
