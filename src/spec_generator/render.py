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
    SRSDocument,
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
    for e in arch.stack_entries:
        L.append(f"- **{e.key}**: {e.value}")
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


def render_srs(srs: SRSDocument) -> str:
    """The standalone SRS: WHAT must be verifiably true, never HOW. A human
    reads it to sign off; downstream artifacts reference R-ids from it."""

    def _render_req(r):
        L = [f"## {r.requirement_id} [{r.priority.value}] {r.title}", r.statement,
             f"_Needs: {', '.join(r.need_ids)}_", ""]
        for ac in r.acceptance:
            L.append(f"- **{ac.criterion_id}** {ac.statement}")
        L.append("")
        return L

    L = [f"# Software Requirements Specification - {srs.session_id}", "",
         "## 1. Purpose", srs.purpose, "",
         "## 2. Scope", srs.scope_summary, ""]
    if srs.definitions:
        L += ["## 3. Definitions", *[f"- **{t}** - {m}" for t, m in srs.definitions.items()], ""]
    L.append("## 4. Functional requirements")
    L.append("")
    for r in srs.functional:
        L += _render_req(r)
    L.append("## 5. Non-functional requirements")
    L.append("")
    for r in srs.non_functional:
        L += _render_req(r)
    if srs.constraints:
        L += ["## 6. Constraints (design-space limits)", *[f"- {c}" for c in srs.constraints], ""]
    if srs.assumptions:
        L += ["## 7. Assumptions (validated later, not built)",
              *[f"- {a.statement}" for a in srs.assumptions], ""]
    if srs.open_questions:
        L += ["## 8. Open questions (never silently resolved)", *[f"- {q}" for q in srs.open_questions], ""]
    L += ["---",
          "_Solution-agnostic by design: observable outcomes only. Executable checks "
          "(endpoints, selectors, probes) live in the test specifications._"]
    return "\n".join(L)


def render_architecture(arch: ArchitectureOverview) -> str:
    """Human-readable architecture for the founder review gate (ADR-0019) and
    the bundle's architecture.md."""
    lines = ["# Architecture overview\n"]
    if arch.stack_entries:
        lines.append("## Stack")
        for e in arch.stack_entries:
            refs = ", ".join(e.baseline_refs)
            if refs:
                trace = f" (baseline: {refs})"
            elif e.migration_adr_id:
                trace = f" (migration: {e.migration_adr_id})"
            else:
                trace = " (migration: see ADRs)"  # healed pre-0.13.0 artifact - linked by ADR text
            lines.append(f"- **{e.key}**: {e.value}{trace}")
    if arch.components:
        lines.append("\n## Components")
        for c in arch.components:
            refs = f" (realizes {', '.join(c.requirement_ids)})" if c.requirement_ids else ""
            baseline = f" [baseline: {', '.join(c.baseline_refs)}]" if c.baseline_refs else ""
            migration = f" [migration: {c.migration_adr_title}]" if c.migration_adr_title else ""
            lines.append(f"- **{c.name}**{refs}: {c.responsibility}{baseline}{migration}")
    if arch.decisions:
        lines.append("\n## Decisions (ADRs)")
        for d in arch.decisions:
            lines.append(f"- **{d.decision_id} {d.title}** - {d.decision}")
            # ADR-0021: drivers/reasons are local-model judgment until retrieval
            # grounding lands (ADR-0019 Part 3) - labeled so the gate can veto.
            if d.driver is not None:
                lines.append(f"  Driver: {d.driver.value.replace('_', ' ')} "
                             f"(model judgment - unverified; challenge here)")
            else:
                lines.append("  Driver: (not recorded - pre-0.8.0 artifact)")
            lines.append(f"  Why: {d.rationale}")
            if d.baseline_refs:
                lines.append(f"  Baseline: {', '.join(d.baseline_refs)}")
            if d.migration:
                lines.append(f"  Migration: yes; affected repos: {', '.join(d.affected_repositories)}")
                lines.append(f"  Evidence: {'; '.join(d.evidence)}")
            if d.options_considered:
                considered = "; ".join(f"{o.option} - rejected: {o.reason}" for o in d.options_considered)
                lines.append(f"  Considered: {considered}")
            elif d.rejected:  # deprecated 0.7.x field, legacy artifacts only
                lines.append(f"  Rejected: {'; '.join(d.rejected)}")
    if arch.non_functionals:
        lines.append("\n## Non-functional positions")
        lines += [f"- **{k}**: {v}" for k, v in arch.non_functionals.items()]
    return "\n".join(lines) + "\n"

