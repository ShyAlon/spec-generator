
from spec_generator import graph, provenance
from spec_generator.findings_ledger import Ledger


def req(rid, title, statement, kind="functional", needs=None, acc=None):
    return {"requirement_id": rid, "title": title, "statement": statement, "kind": kind, "priority": "must",
            "need_ids": needs or [f"N-{rid[2:]}"], "intent_fields": ["stakeholder_needs"], "source_turns": [],
            "acceptance": acc or [{"criterion_id": f"AC-{rid[2:]}", "statement": f"{rid} works"}]}


R16 = req("R-16", "Bundled Lesson Content and Free Backing Tunes",
          "The app shall ship with all lesson content and all backing tunes already included in the delivered app, "
          "so that every lesson is fully usable without retrieving any content or audio from outside the app.")
R17 = req("R-17", "Bundled Lesson Content and Audio",
          "The app shall make all lesson content and all audio assets available to the child without retrieving "
          "any of them from an external source at runtime.")
OTHER = req("R-9", "Parent Report", "The app shall show a parent a weekly summary of completed lessons.")


def test_review_band_pair_becomes_open_finding_and_reruns_replace_it():
    ledger = Ledger()
    data = {"functional": [R16, R17, OTHER], "non_functional": []}
    out = graph._dedupe_requirements(data, ledger)
    assert [r["requirement_id"] for r in out["functional"]] == ["R-16", "R-17", "R-9"]   # no auto-merge
    open_ = ledger.open_findings()
    assert [f["rule_id"] for f in open_] == ["srs.duplicate_requirement"]
    assert open_[0]["targets"] == ["R-16", "R-17"] and ledger.gate()["blocked"]
    graph._dedupe_requirements({"functional": [R16, OTHER], "non_functional": []}, ledger)  # the pair is gone
    assert ledger.open_findings() == []


def test_identical_requirements_remain_separate_with_all_traceability():
    import copy
    twin = {**R17, "requirement_id": "R-18", "need_ids": ["N-18"],
            "acceptance": [{"criterion_id": "AC-18", "statement": "Other check"}]}
    nf = req("R-30", "Speed", "The app shall respond quickly.", kind="non_functional")
    original = {"functional": [R17, twin, OTHER], "non_functional": [nf]}
    before = copy.deepcopy(original)
    ledger = Ledger()
    out = graph._dedupe_requirements(original, ledger)
    assert out == before and original == before
    assert any(f["targets"] == ["R-17", "R-18"] for f in ledger.open_findings())
    assert ledger.gate()["blocked"]
    assert not any(h["event"] == "auto_merged_requirements" for h in ledger.history)


@__import__("pytest").mark.parametrize("first,second", [
    ("The system shall permit guest access.", "The system shall not permit guest access."),
    ("The system shall retain records for 7 days.", "The system shall retain records for 30 days."),
])
def test_opposite_or_different_numeric_limits_are_findings_not_merges(first, second):
    data = {"functional": [req("R-1", "", first), req("R-2", "", second)], "non_functional": []}
    ledger = Ledger()
    assert graph._dedupe_requirements(data, ledger) == data
    assert len(data["functional"]) == 2
    finding = ledger.open_findings()[0]
    assert finding["targets"] == ["R-1", "R-2"] and finding["detail"]["score"] == 1.0
    assert ledger.gate()["blocked"]


def test_requirement_without_text_marks_the_check_incomplete():
    ledger = Ledger()
    graph._dedupe_requirements({"functional": [R17, req("R-5", "", "")], "non_functional": []}, ledger)
    assert ledger.incomplete and ledger.incomplete[0]["requirement_ids"] == ["R-5"] and ledger.gate()["blocked"]
    graph._dedupe_requirements({"functional": [R17], "non_functional": []}, ledger)
    assert ledger.incomplete == []


INTENT = {"reference_names": [{"name": "Duolingo 16K", "kind": "working_title", "source_turns": [1]}]}


def test_provenance_lint_flags_turn_tags_and_reference_names_per_target():
    srs = {"functional": [req("R-1", "Lessons", "The app shall offer lessons (turn 2).")],
           "non_functional": [req("R-2", "Speed", "Like Duolingo 16K, it shall be fast.", kind="non_functional")]}
    items = [{"spec_id": "SPEC-1", "body_markdown": "Clean body.", "requirement_ids": ["R-1"]}]
    ledger = Ledger()
    n = provenance.raise_findings(ledger, {"srs": srs, "items": items}, INTENT)
    rules = {(f["rule_id"], tuple(f["targets"])) for f in ledger.open_findings()}
    assert n == 2 and rules == {("provenance.turn_tag", ("R-1",)), ("provenance.reference_name", ("R-2",))}
    srs["functional"][0]["statement"] = "The app shall offer lessons."
    srs["non_functional"][0]["statement"] = "It shall be fast."
    provenance.raise_findings(ledger, {"srs": srs, "items": items}, INTENT)
    assert ledger.open_findings() == []     # re-run retires findings that no longer hit


def test_no_declared_names_records_check_not_run_without_blocking():
    ledger = Ledger()
    provenance.raise_findings(ledger, {"srs": {"functional": [R17], "non_functional": []}}, {})
    assert ledger.history[-1]["event"] == "check_not_run" and not ledger.gate()["blocked"]
    provenance.raise_findings(ledger, {"srs": {"functional": [R17], "non_functional": []}}, {})
    assert len([h for h in ledger.history if h["event"] == "check_not_run"]) == 1


def test_spec_target_revision_is_the_body_revision():
    from spec_generator.findings_ledger import revision_of
    items = [{"spec_id": "SPEC-1", "body_markdown": "Built like Duolingo 16K does."}]
    ledger = Ledger()
    provenance.raise_findings(ledger, {"items": items}, INTENT)
    f = ledger.open_findings()[0]
    assert f["artifact_revision"] == revision_of(items[0]["body_markdown"]) and f["targets"] == ["SPEC-1"]
