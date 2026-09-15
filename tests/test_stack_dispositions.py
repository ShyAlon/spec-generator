from vmodel_contracts import ArchitectureOverview
from spec_generator.graph import _assemble_architecture
from spec_generator.render import render_architecture, render_agents_md


def test_dispositions_are_not_projected_or_verification_targets():
    archd = _assemble_architecture(session_id="s", baseline_id="b", entries={
        "language":{"key":"language","value":"Python","technology_id":"tech:python","baseline_refs":[]},
        "live_api":{"key":"live_api","status":"research_required","research_brief":"Check current API coverage and terms."},
        "hardware":{"key":"hardware","status":"not_applicable","reason":"The browser-only concept controls no hardware."},
    }, comp_slots=[], comp_bodies={}, adr_slots=[{"decision_id":"ADR-1","authorizes":["stack:language","stack:live_api","stack:hardware"],"migration":False}],
    adr_bodies={"ADR-1":{"title":"Runtime","decision":"Python","rationale":"fit","driver":"simplicity","options_considered":[],"executes_concept_ids":[]}}, nfr={})
    arch=ArchitectureOverview.model_validate(archd)
    assert [e.key for e in arch.stack_entries] == ["language"]
    assert [x.slot_id for x in arch.decisions[0].technology_selections] == ["slot:language"]
    assert {e.status.value for e in arch.stack_entry_dispositions} == {"research_required","not_applicable"}
    md=render_architecture(arch)
    assert "RESEARCH REQUIRED" in md and "NOT APPLICABLE" in md
