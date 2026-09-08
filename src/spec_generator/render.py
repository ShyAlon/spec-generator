"""Deterministic renderers: bundle artifacts as markdown/text. The AGENTS.md
bundle is the deliverable a coding agent executes against - milestone build
order, conventions, guardrails."""
from __future__ import annotations

from vmodel_contracts import (
    AcceptanceTest,
    ArchitectureOverview,
    ItemSpec,
    PunchListItem,
    RequirementSet,
    SpecManifest,
    StakeholderNeedsSet,
)


def render_agents_md(
    arch: ArchitectureOverview,
    items: list[ItemSpec],
    manifest: SpecManifest,
    punch_list: list[PunchListItem],
) -> str:
    L = [
        "# AGENTS.md",
        "",
        "You are building against a systems-engineering specification. Do not guess.",
        "If a behavior is not specified, stop and ask - an unspecified behavior is a gap, not an invitation.",
        "",
        "## Architecture (decided - do not re-litigate)",
        "",
    ]
    for k, v in arch.stack.items():
        L.append(f"- **{k}**: {v}")
    L += ["", "### Decisions and why", ""]
    for d in arch.decisions:
        L.append(f"**{d.decision_id} {d.title}.** {d.decision} Rationale: {d.rationale}")
        for r in d.rejected:
            L.append(f"- Rejected: {r}")
    L += ["", "## Build order", ""]
    for m in manifest.milestones:
        L.append(f"{m}. {m.title} ({', '.join(m.spec_ids)})")
        L.append(f"   Done when: {m.done_when}")
    L += ["", "## Specs", ""]
    for i in items:
        L.append(f"- `{i.spec_id}` ({i.kind.value}) {i.title} -> spec file `{i.spec_id}.md`")
    if punch_list:
        L += ["", "## Known gaps (punch list - resolve before assuming)", ""]
        for p in punch_list:
            L.append(f"- [{p.severity}] {p.item_id}: {p.description} (blocks {', '.join(p.blocks)})")
    L += [
        "",
        "## Acceptance",
        "",
        "Every requirement ships with machine-checkable acceptance criteria.",
        "Run the acceptance suite after each milestone; a red test is a divergence, not a suggestion.",
    ]
    return "\n".join(L)


def render_spec_file(item: ItemSpec) -> str:
    return f"# {item.spec_id}: {item.title}\n\n_kind: {item.kind.value}_\n_requirements: {', '.join(item.requirement_ids) or 'none'}_\n\n{item.body_markdown}\n"


def render_needs(needs: StakeholderNeedsSet) -> str:
    L = ["# Stakeholder needs", ""]
    for n in needs.needs:
        L.append(f"**{n.need_id}** ({n.stakeholder}): {n.statement}")
    return "\n".join(L)


def render_requirements(reqs: RequirementSet) -> str:
    L = ["# System requirements", ""]
    for r in reqs.requirements:
        L.append(f"## {r.requirement_id} [{r.priority.value}] {r.title}")
        L.append(r.statement)
        L.append(f"_Needs: {', '.join(r.need_ids)}_")
        L.append("")
        for ac in r.acceptance:
            hint = f" (check: {ac.executable_hint})" if ac.executable_hint else ""
            L.append(f"- **{ac.criterion_id}** {ac.statement}{hint}")
        L.append("")
    return "\n".join(L)
