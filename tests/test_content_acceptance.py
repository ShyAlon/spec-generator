"""JEV-D10 rules 1 and 2 against the founder-labeled frozen fixture."""
import hashlib
import json
from pathlib import Path

import pytest

from spec_generator import content_acceptance as ca

FX = Path(__file__).parent / "fixtures" / "jev_d10"
CORPUS_BYTES = (FX / "distance-corpus-v1.json").read_bytes()
CORPUS = json.loads(CORPUS_BYTES)
LABELS = {x["case_id"]: x["label"] for x in json.loads((FX / "distance-labels-v1.json").read_text())["labels"]}
CASES = {c["id"]: c for c in CORPUS["cases"]}
RAW_SHA256 = "7be8e9b4b0ee8860f393728152c3564dc7f99776f424a327cfc498f83430ac52"


def test_fixture_is_frozen_and_fully_labeled():
    assert hashlib.sha256(CORPUS_BYTES).hexdigest() == RAW_SHA256
    assert set(LABELS) == set(CASES) and len(CASES) == 28
    assert set(LABELS.values()) == {"yes", "no"}  # no unsure labels


def _echo_status(case):
    results = ca.check_echo(case["b"], {"directive": case["a"]})
    return ca.worst(results)


def _dup_status(case):
    return ca.worst(ca.check_duplicate("CAND", case["b"], {"SIB": case["a"]}))


@pytest.mark.parametrize("cid", [c for c in CASES if CASES[c]["task"] == "echo"])
def test_echo_never_rejects_a_labeled_no_and_never_passes_an_obvious_yes(cid):
    status = _echo_status(CASES[cid])
    if LABELS[cid] == "no":
        assert status != ca.REJECT, "deterministic layer must not auto-reject a founder NO"
    else:
        # echo YES: reject, or gray (goes to Jev/founder). A pass would be a silent miss.
        assert status in (ca.REJECT, ca.GRAY), cid


def test_echo_extremes_and_gray_band_shape():
    s = {cid: _echo_status(CASES[cid]) for cid in CASES if CASES[cid]["task"] == "echo"}
    for exact_or_variant in ("C-01", "C-02", "C-03", "C-04", "C-05", "C-06", "C-08"):
        assert s[exact_or_variant] == ca.REJECT
    assert s["C-07"] == ca.GRAY   # quotation inside useful text: not auto-decided
    assert s["C-12"] == ca.GRAY   # partial directive: founder YES, no deterministic cut
    for far in ("C-09", "C-10", "C-11", "C-13"):
        assert s[far] == ca.PASS


@pytest.mark.parametrize("cid", [c for c in CASES if CASES[c]["task"] == "duplicate"])
def test_duplicate_separates_the_labels(cid):
    status = _dup_status(CASES[cid])
    if LABELS[cid] == "no":
        assert status == ca.PASS, cid
    elif cid == "C-25":
        assert status == ca.GRAY   # headings only: nothing substantive, review trigger
    else:
        assert status == ca.REJECT, cid


def test_holdout_split_agrees_with_selection_split():
    """Thresholds were read off all 28 labels, so also report each deterministic
    half separately: a threshold that only works on one half is not trustworthy."""
    dup = [c for c in CASES if CASES[c]["task"] == "duplicate" and c != "C-25"]
    halves = {0: [], 1: []}
    for c in dup:
        halves[int(hashlib.sha256(c.encode()).hexdigest(), 16) % 2].append(c)
    for part in halves.values():
        assert part, "split must not be empty"
        for c in part:
            assert (_dup_status(CASES[c]) == ca.REJECT) == (LABELS[c] == "yes")


def test_calibration_scores_reproduce():
    """Whole-body echo distances match the calibration review's table."""
    for cid, want in {"C-04": 0.0208, "C-05": 0.1078, "C-06": 0.1193, "C-07": 0.3191,
                      "C-12": 0.3646, "C-09": 0.6354, "C-13": 0.6458}.items():
        c = CASES[cid]
        got = ca.normalized_distance(ca.normalize(c["a"]), ca.normalize(c["b"]))
        assert round(got, 4) == want, cid


def test_normalization_preserves_ids_numbers_and_domain_nouns():
    assert ca.normalize("## **R-12** needs 3 NOTES\r\n  per  `exercise`") == "r-12 needs 3 notes per exercise"
    assert ca.normalize("ｐｒｏｇｒｅｓｓ") == "progress"


def test_empty_string_behavior_is_explicit():
    assert ca.normalized_distance("", "") == 0.0
    assert ca.normalized_distance("", "x") == 1.0
    assert ca.shingle_jaccard("", "") == 1.0
    assert ca.check_echo("") == []   # empty body is a schema failure elsewhere, not an echo
    assert ca.substantive_length("## Interface\n## Errors") == 0


def test_over_limit_work_is_not_checked_never_a_pass(monkeypatch):
    monkeypatch.setattr(ca, "MAX_COMPARE_CELLS", 50)
    body = "word " * 40
    assert ca.worst(ca.check_duplicate("C", body, {"S": body + " different tail text here"})) == ca.NOT_CHECKED
    assert ca.worst(ca.check_echo(body, {"d": "some directive that is long enough to blow the cap " * 2})) == ca.NOT_CHECKED


def test_short_valid_clauses_are_a_review_trigger_not_an_auto_reject():
    # 19 vs 20 char boundary: non-identical short bodies never auto-reject on distance
    a, b = "Return a numeric score.", "Return a numeric scores."
    assert ca.substantive_length(a) >= ca.MIN_SUBSTANTIVE_CHARS
    short_a, short_b = "Reject bad input.", "Reject bad inputs."
    assert ca.substantive_length(short_a) < ca.MIN_SUBSTANTIVE_CHARS
    assert ca.worst(ca.check_duplicate("C", short_b, {"S": short_a})) == ca.GRAY
    assert ca.worst(ca.check_duplicate("C", short_a, {"S": short_a})) == ca.REJECT  # exact always fails


def test_shared_headings_and_pipeline_titles_are_excluded_but_other_headings_count():
    a = "# SPEC-1: Piano\n## Interface\nplayTone(noteId, time) produces a tone from the synthesizer."
    b = "# SPEC-2: Score\n## Interface\nplayTone(noteId, time) produces a tone from the synthesizer."
    assert ca.worst(ca.check_duplicate("SPEC-2", b, {"SPEC-1": a})) == ca.REJECT
    c = "## Custom heading\nplayTone(noteId, time) produces a tone from the synthesizer."
    assert ca.substantive_text(c).startswith("## Custom heading")


def test_shingle_catches_reordered_paragraphs_levenshtein_punishes():
    p1 = "Store each note event locally with its onset time and pitch value."
    p2 = "Compute the score from timing deviations and missing or extra notes."
    p3 = "Show the grade immediately after the last exercise finishes."
    a, b = "\n\n".join([p1, p2, p3]), "\n\n".join([p3, p1, p2])
    assert ca.normalized_distance(ca.normalize(a), ca.normalize(b)) > ca.DUP_LEV_MAX
    assert ca.shingle_jaccard(ca.normalize(a), ca.normalize(b)) >= ca.DUP_SHINGLE_MIN


def test_result_records_evidence_for_the_ledger():
    r = ca.check_echo("## Piano\n\n" + ca.DIRECTIVE_CATALOGUE["watchdog-retry"])[0].to_dict()
    assert r["catalogue_id"] == "watchdog-retry" and r["status"] == ca.REJECT
    assert r["candidate_sha256"] and r["other_sha256"] and r["policy_version"] == ca.POLICY_VERSION
    assert r["threshold"]["reject_max"] == ca.ECHO_REJECT_MAX
