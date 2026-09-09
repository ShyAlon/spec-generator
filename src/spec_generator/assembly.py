"""Deterministic assembly: everything here is plain code on purpose.

Numbering, the traceability matrix, the manifest, the AGENTS.md render, the
build plan ordering, and bundle validation are STRUCTURE - an LLM adds cost
and drift, not quality (Cost Addendum: deterministic owns structure).
"""
from __future__ import annotations

from vmodel_contracts import (
    AcceptanceTest,
    ArchitectureOverview,
    BuildMilestone,
    ItemSpec,
    PunchListItem,
    RequirementSet,
    SpecBundle,
    SpecManifest,
    StakeholderNeedsSet,
    TraceabilityEntry,
)

MANIFEST_FILENAME = "manifest.json"


class AssemblyError(ValueError):
    """The bundle fails its own invariant checks. Never ship an orphan."""


def validate_traceability(
    needs: StakeholderNeedsSet,
    requirements: RequirementSet,
    items: list[ItemSpec],
    tests: list[AcceptanceTest],
) -> list[str]:
    """V-model invariants. Returns a list of violations (empty = clean)."""
    violations: list[str] = []
    need_ids = {n.need_id for n in needs.needs}
    req_ids = requirements.requirement_ids()
    spec_ids = {i.spec_id for i in items}
    test_ids = {t.test_id for t in tests}

    for r in requirements.requirements:
        missing = set(r.need_ids) - need_ids
        if missing:
            violations.append(f"{r.requirement_id} traces to unknown needs {sorted(missing)}")
        if not r.need_ids:
            violations.append(f"{r.requirement_id} traces to no stakeholder need")
    covered_reqs = {rid for i in items for rid in i.requirement_ids}
    for rid in req_ids - covered_reqs:
        violations.append(f"{rid} is covered by no item spec")
    for i in items:
        unknown = set(i.requirement_ids) - req_ids
        if unknown:
            violations.append(f"{i.spec_id} claims unknown requirements {sorted(unknown)}")
        unknown_deps = set(i.depends_on) - spec_ids
        if unknown_deps:
            violations.append(f"{i.spec_id} depends on unknown specs {sorted(unknown_deps)}")
    for t in tests:
        if t.spec_id and t.spec_id not in spec_ids:
            violations.append(f"{t.test_id} targets unknown spec {t.spec_id}")
        if t.criterion_id and all(
            t.criterion_id != ac.criterion_id
            for r in requirements.requirements for ac in r.acceptance
        ):
            violations.append(f"{t.test_id} realizes unknown criterion {t.criterion_id}")
    # The demo's promise: every requirement has at least one test.
    tested = {t.criterion_id for t in tests if t.criterion_id}
    for r in requirements.requirements:
        for ac in r.acceptance:
            if ac.criterion_id not in tested:
                violations.append(f"{ac.criterion_id} ({r.requirement_id}) has no test")
    return violations


def build_traceability(requirements: RequirementSet, items: list[ItemSpec], tests: list[AcceptanceTest]) -> list[TraceabilityEntry]:
    entries = []
    for r in requirements.requirements:
        ac_ids = [ac.criterion_id for ac in r.acceptance]
        entries.append(TraceabilityEntry(
            requirement_id=r.requirement_id,
            spec_ids=[i.spec_id for i in items if r.requirement_id in i.requirement_ids],
            acceptance_criterion_ids=ac_ids,
            test_ids=[t.test_id for t in tests if t.criterion_id in ac_ids or t.spec_id in
                      [i.spec_id for i in items if r.requirement_id in i.requirement_ids]],
        ))
    return entries


def order_milestones(items: list[ItemSpec], tests: list[AcceptanceTest]) -> list[BuildMilestone]:
    """Topological order over depends_on, stable by spec_id. Kind order breaks
    ties: data before API before UI before rules before integration."""
    kind_rank = {"data_model": 0, "api_surface": 1, "ui_flow": 2, "business_rule": 3, "integration": 4, "non_functional": 5}
    done: set[str] = set()
    remaining = {i.spec_id: i for i in items}
    milestones: list[BuildMilestone] = []
    while remaining:
        ready = [i for i in remaining.values() if set(i.depends_on) <= done]
        if not ready:  # cycle: break deterministically, flag loudly in the punch list upstream
            ready = [min(remaining.values(), key=lambda i: i.spec_id)]
        ready.sort(key=lambda i: (kind_rank.get(i.kind.value, 9), i.spec_id))
        batch_ids = [i.spec_id for i in ready]
        milestones.append(BuildMilestone(
            order=len(milestones) + 1,
            title=" + ".join(i.title for i in ready[:3]) + (" ..." if len(ready) > 3 else ""),
            spec_ids=batch_ids,
            test_ids=[t.test_id for t in tests if t.spec_id in batch_ids],
            done_when="All milestone tests pass in the founder's environment.",
        ))
        done.update(batch_ids)
        for bid in batch_ids:
            remaining.pop(bid)
    return milestones
