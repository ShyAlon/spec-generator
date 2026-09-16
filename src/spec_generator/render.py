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


def _adr_title(decision_id: str, title: str) -> str:
    """Return title prose only; the renderer owns the ADR id prefix."""
    import re
    return re.sub(rf"^\s*{re.escape(decision_id)}\s*[:.\-]?\s*", "", title,
                  flags=re.IGNORECASE).strip()


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
    for e in arch.stack_entry_dispositions:
        if e.status.value == "research_required":
            L.append(f"- **{e.key}**: RESEARCH REQUIRED - {e.research_brief}")
        else:
            L.append(f"- **{e.key}**: NOT APPLICABLE - {e.reason}")
    L += ["", "### Decisions and why", ""]
    for d in arch.decisions:
        L.append(f"**{d.decision_id}: {_adr_title(d.decision_id, d.title)}.** {d.decision} Rationale: {d.rationale}")
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
    from .skeleton import is_future_commitment
    L = ["# Stakeholder needs", ""]
    for n in needs.needs:
        marker = " _(deferred - future phase, drives no current requirements)_" \
            if is_future_commitment(n.statement) else ""
        L.append(f"**{n.need_id}** ({n.stakeholder}): {n.statement}{marker}")
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


def architecture_challenges(arch: ArchitectureOverview) -> list[str]:
    """The deterministic OPTIMIZATION LAYER (founder decision 2026-09-13, his
    words: 'I prefer it to be deterministic code'). Computed from the
    candidate matrix's capability tags - flags land in the review output as
    NAMED CHALLENGES, never silent changes:
    - capability overlap: two chosen technologies provide the same capability
      (his example: Django alongside Firebase Auth + Firestore + FCM);
    - tooling spread: every runtime is its own toolchain, CI and dependency
      surface (his minimal-dependencies preference).
    Unknown (novel) technologies contribute nothing - a matrix miss is never
    a verdict."""
    from .tech_candidates import primary_row_in_text, tech_row
    resolved_rows = [(e, tech_row(e.value) or primary_row_in_text(e.value))
                     for e in arch.stack_entries]
    rows = [(e, r) for e, r in resolved_rows if r]
    flags: list[str] = []

    # Stack-entry hygiene (2026-09-13 greenfield defects, deterministic):
    # (a) a vendor/technology answering a STRATEGY slot
    #     ('resource_utilization_strategy: Firebase' - the question asked for
    #     an approach, not a product);
    # Do this from the unfiltered entries, independently of later grouping.
    # That keeps the flag alive even when another check normalizes/merges the
    # same vendor family (the overnight Firebase shape exposed that coupling).
    for e, r in resolved_rows:
        if r and any(m in e.key.lower() for m in ("strategy", "approach", "pattern", "policy")):
            flags.append(f"strategy slot answered with a vendor: {e.key} = {r['name']} - "
                         "a strategy slot wants an approach (cache-first, serverless, "
                         "in-house...); the vendor belongs in a technology slot")
    # (b) cross-key duplication: the SAME matrix row answering two different
    #     keys (database_system and data_persistence_service both PostgreSQL).
    by_row: dict[str, list[str]] = {}
    for e, r in rows:
        by_row.setdefault(r["name"], []).append(e.key)
    for name, keys in sorted(by_row.items()):
        if len(keys) > 1:
            flags.append(f"same technology under two keys: {name} answers "
                         + " and ".join(sorted(keys))
                         + " - merge the slots or let each ADR justify why both exist")
    # (c) vendor concentration (the Django+Firebase defect mutated: Firebase
    #     picked 3x - FCM, Auth, resource strategy). Same product family
    #     across 3+ slots must be justified per-slot or consolidated.
    by_family: dict[str, list[str]] = {}
    for e, r in rows:
        family = r["name"].lower().split()[0]
        by_family.setdefault(family, []).append(f"{r['name']} ({e.key})")
    for family, holders in sorted(by_family.items()):
        if len(holders) >= 3:
            flags.append(f"vendor concentration: the {family} family holds {len(holders)} slots ("
                         + ", ".join(sorted(holders))
                         + ") - each slot's ADR must justify the same-vendor choice, "
                         "or consolidate")

    caps: dict[str, list[tuple[str, str]]] = {}
    for e, r in rows:
        for c in r.get("capabilities", []):
            caps.setdefault(c, []).append((e.key, r["name"]))

    def _family(a: str, b: str) -> bool:
        """Composition, not redundancy: a product family covering its own
        sub-service (Firebase + Firebase Cloud Messaging) or a framework +
        its own ORM (Django + Django ORM) is ONE choice, not two."""
        a, b = a.lower(), b.lower()
        return a.startswith(b) or b.startswith(a)

    for c, holders in sorted(caps.items()):
        holders = [h for h in holders
                   if not any(h is not o and _family(h[1], o[1]) for o in holders)]
        if len(holders) > 1:
            flags.append(f"capability overlap: {c} is provided by "
                         + " + ".join(f"{name} ({key})" for key, name in holders)
                         + " - one owner per capability keeps the build small; if both "
                         "stay, an ADR should say why")
    runtimes = sorted({r["runtime"] for _, r in rows if r.get("runtime")})
    if len(runtimes) > 2:
        flags.append(f"tooling spread: {len(runtimes)} runtimes ({', '.join(runtimes)}) - "
                     "each is its own toolchain, CI and dependency surface "
                     "(minimal-dependencies preference)")
    return flags



def _matrix_evidence(row: dict | None) -> str | None:
    """Compact observed matrix signal for one selected technology."""
    if not row:
        return None
    sm = row.get("service_metrics") or {}
    contributors = sm.get("contributors") or []
    score = sm.get("consensus")
    if contributors and score is not None:
        n = len(contributors)
        if n >= 2:
            verdict = "consensus strong" if sm.get("strong") else "consensus mixed/weak"
        else:
            verdict = "strong one-source signal" if score >= 2 else "weak one-source signal"
        shown = int(score) if float(score).is_integer() else score
        return f"matrix: {n} source{'s' if n != 1 else ''}, score {shown}, {verdict}"
    metrics = row.get("metrics") or {}
    if metrics.get("weekly_downloads") is not None:
        return f"matrix: registry weekly downloads {metrics['weekly_downloads']:,}"
    return None


def _driver_evidence(decision) -> str:
    """Show observed evidence without laundering it into proof of a driver.

    Adoption/maturity drivers may be grounded by adoption metrics. For other
    drivers, retain the unverified label but still show the selected
    technology's matrix signal. This makes the retrieval visible at review
    while saying exactly what it cannot prove (for example, popularity does
    not prove simplicity).
    """
    from .tech_candidates import primary_row_in_text
    row = primary_row_in_text(" ".join((decision.title, decision.decision)))
    matrix = _matrix_evidence(row)
    driver = decision.driver.value if decision.driver is not None else None
    if driver in {"ecosystem_maturity", "community_adoption"} and matrix:
        return matrix
    if matrix:
        return f"model judgment - unverified; {matrix} (adoption signal only, not proof of {driver.replace('_', ' ')})"
    return "model judgment - unverified; challenge here"


def render_architecture(arch: ArchitectureOverview, greenfield: bool = False,
                        conops_present: bool = True) -> str:
    """Human-readable architecture for the founder review gate (ADR-0019) and
    the bundle's architecture.md. greenfield=True labels ungrounded entries as
    what they ARE - new design, not a migration (2026-09-13: his greenfield
    bundle's architecture.md said 'migration: see ADRs' for all 10 stack
    entries, contradicting the greenfield no-migration semantics fixed the
    same morning)."""
    lines = ["# Architecture overview\n"]
    if not conops_present:
        lines += ["## Concept of Operations grounding",
                  "- UNVERIFIED: no Concept of Operations artifact was supplied (legacy session). "
                  "Any product-concept decision in this architecture must be challenged before approval.", ""]
    if arch.stack_entries:
        lines.append("## Stack")
        for e in arch.stack_entries:
            refs = ", ".join(e.baseline_refs)
            if refs:
                trace = f" (baseline: {refs})"
            elif e.migration_adr_id:
                trace = f" (migration: {e.migration_adr_id})"
            elif greenfield:
                trace = " (new design)"
            else:
                trace = " (migration: see ADRs)"  # healed pre-0.13.0 artifact - linked by ADR text
            lines.append(f"- **{e.key}**: {e.value}{trace}")
    if arch.stack_entry_dispositions:
        lines.append("\n## Considered stack slots without a technology selection")
        for e in arch.stack_entry_dispositions:
            if e.status.value == "research_required":
                lines.append(f"- **{e.key}** - RESEARCH REQUIRED: {e.research_brief}")
            else:
                lines.append(f"- **{e.key}** - NOT APPLICABLE: {e.reason}")
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
            lines.append(f"- **{d.decision_id}: {_adr_title(d.decision_id, d.title)}** - {d.decision}")
            if d.driver is not None:
                driver = d.driver.value.replace("_", " ")
                lines.append(f"  Driver: {driver} "
                             f"({_driver_evidence(d)})")
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
    flags = architecture_challenges(arch)
    if flags:
        lines.append("\n## Deterministic challenges (review these before approving)")
        lines += [f"- {f}" for f in flags]
    return "\n".join(lines) + "\n"

