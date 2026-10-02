#!/usr/bin/env python3
"""List and dispose findings in a spec-generator output dir (the findings ledger).

  findings-review.py DIR summary
  findings-review.py DIR list    [FILTERS] [--limit N]
  findings-review.py DIR dispose [FILTERS | --ids F-..,F-..] --subtype S --rationale TEXT [--yes]
  findings-review.py DIR staged
  findings-review.py DIR unstage [FINDING_ID ...]          (no ids = clear everything staged)
  findings-review.py DIR write   [OUT.json]
  findings-review.py DIR apply   [OUT.json]

FILTERS: --rule R  --source S  --kind K  --scope S  --target SPEC-3  --text SUBSTRING  (all must match;
only OPEN findings are listed or disposed). --subtype is one of founder_disposition, false_positive,
accepted_risk, resolved_by_reviewed_replacement.

dispose is a DRY RUN: it prints the count and exact finding ids it would stage and changes nothing until
you add --yes. It refuses an empty match and structural findings. write turns the staged entries into a
founder script (selector = exact finding_id, bound to the finding's artifact revision) and validates it
with the pipeline's own Ledger.apply_script on a COPY. apply only PRINTS the --dispose command to run;
nothing here changes the checkpoint.

DIR is the output dir holding findings-ledger.json (or spec/findings-ledger.json). Run with the
spec-generator virtualenv:  .venv/bin/python scripts/findings-review.py DIR ..."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from spec_generator.findings_ledger import DISPOSITION_SUBTYPES, OPEN, STRUCTURAL, Ledger, LedgerError

STAGED = "findings-dispositions.staged.json"


def locate(d: Path) -> Path:
    for cand in (d / "findings-ledger.json", d / "spec" / "findings-ledger.json"):
        if cand.is_file():
            return cand
    sys.exit(f"no findings-ledger.json in {d} or {d}/spec")


def load(d: Path) -> Ledger:
    return Ledger.from_dict(json.loads(locate(d).read_text()))


def matches(f: dict, a) -> bool:
    if a.rule and f["rule_id"] != a.rule:
        return False
    if a.source and a.source not in (f.get("sources") or []):
        return False
    if a.kind and f["kind"] != a.kind:
        return False
    if a.scope and f.get("scope") != a.scope:
        return False
    if a.target and a.target not in f["targets"]:
        return False
    if a.text and a.text.casefold() not in (f.get("summary") or "").casefold():
        return False
    return True


def open_selected(led: Ledger, a) -> list[dict]:
    wanted = {x.strip() for x in (a.ids or "").split(",") if x.strip()}
    out = [f for f in led.open_findings() if (not wanted or f["finding_id"] in wanted) and matches(f, a)]
    if wanted:
        missing = wanted - {f["finding_id"] for f in out}
        if missing:
            sys.exit(f"not open (or filtered out): {', '.join(sorted(missing))}")
    return sorted(out, key=lambda f: (f["targets"], f["finding_id"]))


def staged_path(d: Path) -> Path:
    return d / STAGED


def read_staged(d: Path) -> list[dict]:
    p = staged_path(d)
    return json.loads(p.read_text()) if p.is_file() else []


def short(f: dict, n: int = 110) -> str:
    s = " ".join((f.get("summary") or "").split())
    return s if len(s) <= n else s[: n - 3] + "..."


def cmd_summary(d, led, a):
    fs = list(led.findings.values())
    opens = [f for f in fs if f["state"] == OPEN]
    print(f"{len(fs)} findings, {len(opens)} OPEN")
    for title, key in (("state", lambda f: f["state"]), ("rule", lambda f: f["rule_id"]),
                       ("source", lambda f: ",".join(f.get("sources") or [])), ("kind", lambda f: f["kind"]),
                       ("scope", lambda f: f.get("scope") or "")):
        print(f"\nOPEN by {title}:")
        for k, n in Counter(key(f) for f in opens).most_common():
            print(f"  {n:4d}  {k}")
    print("\nOPEN by target:")
    for k, n in sorted(Counter(t for f in opens for t in f["targets"]).items()):
        print(f"  {n:4d}  {k}")
    inc = led.incomplete
    print(f"\n{len(inc)} incomplete required check(s)  (these cannot be disposed; re-run the check):")
    for m in inc:
        print(f"  {m['scope']}: {m['reason']}")
    print(f"\ngate: {'BLOCKED' if led.gate()['blocked'] else 'clear'}  staged dispositions: {len(read_staged(d))}")


def cmd_list(d, led, a):
    sel = open_selected(led, a)
    for f in sel[: a.limit]:
        print(f"{f['finding_id']}  {f['rule_id']:<12} {','.join(f['targets']):<14} rev {f['artifact_revision'][:8]}  {short(f)}")
    print(f"-- {len(sel)} open finding(s) match" + (f" (showing {a.limit})" if len(sel) > a.limit else ""))


def cmd_dispose(d, led, a):
    if a.subtype not in DISPOSITION_SUBTYPES:
        sys.exit(f"--subtype must be one of: {', '.join(sorted(DISPOSITION_SUBTYPES))}")
    if not (a.rationale or "").strip():
        sys.exit("--rationale is required")
    sel = open_selected(led, a)
    if not sel:
        sys.exit("refused: no open finding matches (nothing staged)")
    structural = [f["finding_id"] for f in sel if f["kind"] == STRUCTURAL]
    if structural:
        sys.exit(f"refused: structural findings cannot be waived: {', '.join(structural)}")
    already = {e["selector"]["finding_id"] for e in read_staged(d)}
    new = [f for f in sel if f["finding_id"] not in already]
    print(f"{'STAGING' if a.yes else 'DRY RUN - would stage'} {len(new)} disposition(s) as {a.subtype} "
          f"({len(sel) - len(new)} already staged):")
    for f in new:
        print(f"  {f['finding_id']}  {','.join(f['targets'])}  {short(f, 80)}")
    if not a.yes:
        print("nothing changed; add --yes to stage these")
        return
    staged = read_staged(d) + [{"selector": {"finding_id": f["finding_id"]},
                                "expected_artifact_sha256": f["artifact_revision"], "action": "reject",
                                "rationale": a.rationale.strip(), "reason_subtype": a.subtype} for f in new]
    staged_path(d).write_text(json.dumps(staged, indent=2))
    print(f"staged: {len(staged)} total in {staged_path(d)}")


def cmd_staged(d, led, a):
    st = read_staged(d)
    for e in st:
        f = led.findings.get(e["selector"]["finding_id"])
        print(f"{e['selector']['finding_id']}  {e['reason_subtype']:<20} {short(f, 70) if f else '(unknown finding)'}")
    print(f"-- {len(st)} staged")


def cmd_unstage(d, led, a):
    st = read_staged(d)
    drop = set(a.rest)
    keep = [] if not drop else [e for e in st if e["selector"]["finding_id"] not in drop]
    staged_path(d).write_text(json.dumps(keep, indent=2))
    print(f"unstaged {len(st) - len(keep)}; {len(keep)} staged")


def build_script(d, led):
    st = read_staged(d)
    if not st:
        sys.exit("nothing staged (use dispose --yes first)")
    script = {"version": 1, "authored_by": "founder", "dispositions": st}
    try:
        copy = Ledger.from_dict(led.to_dict())          # the pipeline's own validation, on a copy
        copy.apply_script(script, None)
    except LedgerError as exc:
        sys.exit(f"refused by the ledger (nothing written): {exc}")
    return script, copy


def cmd_write(d, led, a):
    script, copy = build_script(d, led)
    out = Path(a.rest[0]) if a.rest else d / "founder-script.json"
    out.write_text(json.dumps(script, indent=2))
    left = copy.gate()
    print(f"wrote {out}: {len(script['dispositions'])} disposition(s), validated on a copy of the ledger.")
    print(f"after applying: {len(left['open'])} open finding(s), {len(left['incomplete'])} incomplete check(s) remain")


def cmd_apply(d, led, a):
    out = Path(a.rest[0]) if a.rest else d / "founder-script.json"
    if not out.is_file():
        sys.exit(f"{out} does not exist; run write first")
    json.loads(out.read_text())
    print("Nothing was changed. To apply (uses the checkpoint, hash-bound, all-or-nothing), run your usual\n"
          "spec-generator command for this session with these extra flags:\n\n"
          f"  --dispose {out}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("cmd", choices=["summary", "list", "dispose", "staged", "unstage", "write", "apply"])
    ap.add_argument("rest", nargs="*")
    for f in ("rule", "source", "kind", "scope", "target", "text", "ids", "subtype", "rationale"):
        ap.add_argument("--" + f)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    led = load(a.dir)
    globals()["cmd_" + a.cmd](a.dir, led, a)


if __name__ == "__main__":
    main()
