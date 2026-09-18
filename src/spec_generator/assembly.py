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
    """Producer adapter over the canonical cross-repository graph validator."""
    from vmodel_contracts import validate_traceability_graph
    return [f"{d.path}: {d.message}" for d in validate_traceability_graph(
        needs, requirements, items, tests)]


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
