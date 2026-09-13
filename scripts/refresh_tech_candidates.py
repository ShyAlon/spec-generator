"""Refresh the dated tech-candidate MATRIX from REAL data sources (founder
decision 2026-09-13: NOT the local model - he explicitly rejected
model-regenerated candidates because the model's knowledge cutoff would
just become the matrix's cutoff). Two data planes:

- registry rows (a ``package`` mapping): live numbers from the npm and PyPI
  APIs - weekly/monthly downloads (his primary sort key), version, publish
  date. The refresh rewrites ONLY the metrics block; names, kinds,
  platforms, aliases, notes, capabilities stay curated.
- service rows (a ``service`` mapping): majority vote across ranking
  sources (StackShare stacks, DB-Engines rank, GitHub stars). Graceful
  degradation: a source that is unreachable or whose markup changed KEEPS
  the row's previous values and the output says so - a bad scrape day must
  never destroy the artifact.
- fully curated rows (neither mapping: HealthKit, SwiftUI, JWT, AES...):
  preserved byte-for-byte; the artifact's as-of date marks when a human
  last reviewed them.

He runs this BY HAND and reviews the output - that review is the human
gate. So the script is built for logs and checks: verbose per-source
progress, a per-row change log, a final summary, and anomaly guards that
REFUSE to write (schema invalid, a source answering for fewer than half
its mapped rows = probable markup change/block, one source flipping >30%
of its mapped votes in a run, every registry fetch failing) - each refusal
comes with the likely cause and the exact next action.

Default is a DRY RUN: fetch, validate, print the review surface. --write
replaces the artifact atomically and bumps as_of. --allow-mass-change
overrides ONLY the vote-flip guard (for a real industry reshuffle you have
already eyeballed in the dry run).

Usage:
    python scripts/refresh_tech_candidates.py [--write] [--allow-mass-change]
        [--no-discovery] [--no-services] [--skip-source=stackshare[,db_engines,github]]
"""
from __future__ import annotations

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spec_generator.errors import UserFacingError            # noqa: E402
from spec_generator.refresh_registry import (                # noqa: E402
    discover_candidates, metrics_diff_lines, refresh_artifact)
from spec_generator.service_sources import (                 # noqa: E402
    refresh_services, services_diff_lines)
from spec_generator.tech_candidates import (                 # noqa: E402
    TECH_CANDIDATES_PATH, validate_artifact)


def main() -> int:
    write = "--write" in sys.argv[1:]
    allow_mass = "--allow-mass-change" in sys.argv[1:]
    discover = "--no-discovery" not in sys.argv[1:]
    services = "--no-services" not in sys.argv[1:]
    # --skip-source stackshare[,db_engines]: permanently-blocked or broken
    # sources are excluded explicitly (logged in the summary), instead of
    # tripping the coverage guard forever.
    skip_sources: set[str] = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--skip-source="):
            skip_sources.update(arg.split("=", 1)[1].split(","))
    today = datetime.date.today()

    if not TECH_CANDIDATES_PATH.exists():
        raise UserFacingError(
            f"no committed artifact at {TECH_CANDIDATES_PATH}.",
            "the matrix is committed to git - a partial checkout, or it was deleted.",
            "restore it with: git checkout -- src/spec_generator/tech_candidates.json")
    data = json.loads(TECH_CANDIDATES_PATH.read_text())

    registry_rows = [r for r in data.get("technologies", [])
                     if isinstance(r, dict) and r.get("package")]
    service_rows = [r for r in data.get("technologies", [])
                    if isinstance(r, dict) and r.get("service")]
    curated_rows = [r for r in data.get("technologies", [])
                    if isinstance(r, dict) and not r.get("package") and not r.get("service")]
    print(f"Matrix: {len(registry_rows)} registry-backed rows, {len(service_rows)} "
          f"service-ranked rows, {len(curated_rows)} fully curated rows (preserved).")

    print("\n== Phase 1/3: package registries (npm, PyPI) ==")
    changes, failures = refresh_artifact(data, today=today)
    print("\nMetric changes (per row):")
    for line in metrics_diff_lines(changes, failures, today=today):
        print(f"  {line}")

    svc_changes, svc_failures, svc_stats = [], [], {}
    if services and service_rows:
        print("\n== Phase 2/3: ranking sources (StackShare, DB-Engines, GitHub) ==")
        if skip_sources:
            print(f"  (explicitly skipped sources: {', '.join(sorted(skip_sources))})")
        svc_changes, svc_failures, svc_stats = refresh_services(
            data, today=today, log=lambda m: print(m, flush=True),
            exclude=frozenset(skip_sources))
        print("\nService-metric changes (per row, with contributors):")
        for line in services_diff_lines(svc_changes, svc_failures):
            print(f"  {line}")

    if discover:
        print("\n== Phase 3/3: discovery (advisory - nothing written) ==")
        for line in discover_candidates(data, today=today):
            print(f"  {line}")

    # ---- checks (refuse to write on any anomaly) ----
    print("\n== Checks ==")
    anomalies = []
    problems = validate_artifact(data)
    if problems:
        anomalies.append(("the refreshed matrix is invalid: " + "; ".join(problems[:5]),
                          "a source returned data in a shape the schema refuses.",
                          "report it or hand-fix the row; the committed artifact was NOT modified."))
    else:
        print("  schema: ok (rows, kinds, aliases, capabilities, metrics, service votes)")
    if write and registry_rows and len(failures) == len(registry_rows):
        anomalies.append(("EVERY registry fetch failed.",
                          "no network, or every registry API is down - writing would only bump dates.",
                          "retry when the network is back. The committed artifact was NOT modified."))
    for src, st in sorted(svc_stats.items()):
        if st["mapped"] >= 4 and st["failed"] * 2 > st["mapped"]:
            anomalies.append((f"source {src} answered for only "
                              f"{st['ok']}/{st['mapped']} mapped rows.",
                              "the site is blocking this run or its markup changed - writing would "
                              "freeze every row it covers at stale values.",
                              "dry-run again later, or fix the parser against the new markup; "
                              "the committed artifact was NOT modified."))
        flip_ratio = st["flipped"] / st["mapped"] if st["mapped"] else 0
        if st["mapped"] >= 4 and flip_ratio > 0.3 and not allow_mass:
            anomalies.append((f"source {src} flipped the strong/weak vote on "
                              f"{st['flipped']}/{st['mapped']} mapped rows ({flip_ratio:.0%}) in one run.",
                              "either a real industry reshuffle or a parsing change - a silent "
                              "write would bake it in unreviewed.",
                              "if the dry-run changes look RIGHT, rerun with: "
                              "python scripts/refresh_tech_candidates.py --write --allow-mass-change"))
    if not anomalies:
        print("  anomaly guards: ok (source coverage, vote stability, registry liveness)")

    # ---- summary ----
    print("\n== Summary ==")
    print(f"  registry: {len(changes)} rows updated, {len(failures)} kept previous (failures)")
    for src, st in sorted(svc_stats.items()):
        print(f"  {src}: {st['ok']}/{st['mapped']} rows answered, "
              f"{st['failed']} kept previous, {st['flipped']} vote flips")
    print(f"  service rows changed: {len(svc_changes)}; fully curated rows untouched: {len(curated_rows)}")

    if anomalies and write:
        worst = anomalies[0]
        raise UserFacingError(
            f"refusing to write ({len(anomalies)} anomaly/anomalies): {worst[0]}",
            worst[1],
            worst[2])
    if not write:
        if anomalies:
            print(f"\n{len(anomalies)} anomaly/anomalies WOULD refuse a --write right now:")
            for a in anomalies:
                print(f"  ! {a[0]}\n    likely cause: {a[1]}\n    next: {a[2]}")
        print("\nDry run only - review above, then pass --write to replace the artifact.")
        return 0
    data["_generated_by"] = "scripts/refresh_tech_candidates.py"
    data["as_of"] = today.isoformat()
    tmp = TECH_CANDIDATES_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    tmp.replace(TECH_CANDIDATES_PATH)
    print(f"\nWrote {TECH_CANDIDATES_PATH} (as_of {today}).")
    print("Review with: git diff src/spec_generator/tech_candidates.json")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except UserFacingError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
