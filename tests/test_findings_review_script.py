"""scripts/findings-review.py: summary, list, dry-run dispose, write (validated on a copy), apply (print only)."""
import json
import subprocess
import sys
from pathlib import Path

from spec_generator.findings_ledger import Ledger, revision_of

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "findings-review.py"


def run(*args, ok=True):
    p = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)
    assert (p.returncode == 0) == ok, p.stdout + p.stderr
    return p.stdout + p.stderr


def make(tmp_path):
    led = Ledger()
    rev = revision_of("SPEC-1 body")
    ids = [led.add(rule_id="review.llm", rule_version="r1", source="review:round-1", kind="semantic", targets=["SPEC-1"],
                   artifact_revision=rev, summary=f"Remove GitHub Pages {i}",
                   evidence=[{"id": f"e{i}", "sha256": str(i)}]) for i in range(3)]
    other = led.add(rule_id="review.llm", rule_version="r1", source="review:round-1", kind="semantic", targets=["SPEC-2"],
                    artifact_revision=revision_of("b"), summary="Add error handling", evidence=[{"id": "x", "sha256": "x"}])
    struct = led.add(rule_id="structural", rule_version="1", source="rule", kind="structural", targets=["SPEC-2"],
                     artifact_revision=revision_of("b"), summary="broken id", evidence=[{"id": "s", "sha256": "s"}])
    led.mark_incomplete("grounding:SPEC-1", "grounding unverified: x")
    (tmp_path / "findings-ledger.json").write_text(json.dumps(led.to_dict()))
    return ids, other, struct


def test_summary_and_list_and_empty_match_and_structural_refusal(tmp_path):
    ids, other, struct = make(tmp_path)
    out = run(tmp_path, "summary")
    assert "5 findings, 5 OPEN" in out and "review.llm" in out and "grounding:SPEC-1" in out and "BLOCKED" in out
    assert "3 open finding(s) match" in run(tmp_path, "list", "--target", "SPEC-1", "--text", "github")
    assert "refused: no open finding matches" in run(tmp_path, "dispose", "--text", "nothing-like-this",
                                                      "--subtype", "false_positive", "--rationale", "x", ok=False)
    assert "structural" in run(tmp_path, "dispose", "--ids", struct, "--subtype", "false_positive", "--rationale", "x", ok=False)


def test_dispose_is_a_dry_run_until_yes_then_write_validates_and_apply_only_prints(tmp_path):
    ids, other, _ = make(tmp_path)
    args = ["--target", "SPEC-1", "--rule", "review.llm", "--subtype", "false_positive", "--rationale", "decided stack"]
    dry = run(tmp_path, "dispose", *args)
    assert "DRY RUN" in dry and all(i in dry for i in ids) and not (tmp_path / "findings-dispositions.staged.json").exists()
    run(tmp_path, "dispose", *args, "--yes")
    assert len(json.loads((tmp_path / "findings-dispositions.staged.json").read_text())) == 3
    assert "0 already" not in run(tmp_path, "dispose", *args)       # second run reports them as already staged
    w = run(tmp_path, "write")
    script = json.loads((tmp_path / "founder-script.json").read_text())
    assert script["authored_by"] == "founder" and {d["selector"]["finding_id"] for d in script["dispositions"]} == set(ids)
    assert "2 open finding(s)" in w and "1 incomplete" in w
    # the real ledger file is untouched: write validates on a copy and apply only prints
    before = (tmp_path / "findings-ledger.json").read_text()
    out = run(tmp_path, "apply")
    assert "--dispose" in out and (tmp_path / "findings-ledger.json").read_text() == before
    # and the script really applies against the pipeline ledger
    led = Ledger.from_dict(json.loads(before))
    assert sorted(led.apply_script(script)) == sorted(ids)


def test_write_refuses_a_stale_staged_entry(tmp_path):
    ids, other, _ = make(tmp_path)
    run(tmp_path, "dispose", "--ids", ids[0], "--subtype", "false_positive", "--rationale", "x", "--yes")
    staged = json.loads((tmp_path / "findings-dispositions.staged.json").read_text())
    staged[0]["expected_artifact_sha256"] = revision_of("older")
    (tmp_path / "findings-dispositions.staged.json").write_text(json.dumps(staged))
    assert "stale" in run(tmp_path, "write", ok=False) and not (tmp_path / "founder-script.json").exists()
