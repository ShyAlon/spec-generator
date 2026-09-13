"""2026-09-13 batch: future-marker handling, deterministic optimization
layer (capability challenges), P2 ADR dedupe, registry-backed refresh,
and ranking-source majority vote for service rows."""
import json
from datetime import date

import pytest

from spec_generator import refresh_registry as rr
from spec_generator import service_sources as ss
from spec_generator.graph import _dedupe_adr_slots, _same_decision_tech
from spec_generator.render import architecture_challenges
from spec_generator.skeleton import (build_srs_skeleton, deferred_needs,
                                     is_future_commitment)
from spec_generator.tech_candidates import (TECH_CANDIDATES, TECH_ROWS,
                                            validate_artifact)


# ---------- 1. future markers ----------

def test_future_markers_detected():
    assert is_future_commitment("In the future the app will have a community feature")
    assert is_future_commitment("Later we might add social sharing")
    assert not is_future_commitment("The app must track habits daily")


def test_deferred_needs_and_skeleton_skip():
    needs = {"needs": [
        {"need_id": "SN-1", "statement": "Track habits daily", "source_turns": [1]},
        {"need_id": "SN-2", "statement": "In the future, a community feature", "source_turns": [2]},
        {"need_id": "N-3", "statement": "Reminders via push", "source_turns": [3]}]}
    assert [n["need_id"] for n in deferred_needs(needs)] == ["SN-2"]
    slots = build_srs_skeleton(needs)
    ids = [s["requirement_id"] for s in slots]
    assert len(slots) == 2
    assert ids == [f"R-{i}" for i in range(1, 3)]  # renumbered, gapless
    assert all("community" not in str(s).lower() for s in slots)


def test_deferred_scope_block_idempotent():
    from spec_generator.graph import srs_document_data
    intent = {"session_id": "s", "problem_statement": "p", "target_users": ["u"],
              "core_value_proposition": "v", "tech_preferences": [],
              "stakeholder_needs": [{"need_id": "SN-1", "statement": "Track habits daily",
                                     "kind": "functional", "source_turns": [1]},
                                    {"need_id": "SN-2", "statement": "In the future, community",
                                     "kind": "functional", "source_turns": [2]}],
              "in_scope": ["a"], "out_of_scope": ["b"], "external_integrations": [],
              "constraints": [], "open_questions": [], "source_turns": {}}
    body = {"scope_summary": "Habit tracking only.\n\nDeferred to a future phase (old):\n- x"}
    d1 = srs_document_data(intent, dict(body))
    d2 = srs_document_data(intent, dict(d1))  # re-validation stays idempotent
    assert "Deferred to a future phase" in d1["scope_summary"]
    assert "community" in d1["scope_summary"] and "Track habits" not in d1["scope_summary"]
    assert d1["scope_summary"].count("Deferred to a future phase") == 1
    assert d1["scope_summary"] == d2["scope_summary"]


# ---------- 2. optimization layer ----------

def _arch_with_stack(stack):
    from test_decision_drivers import _arch
    return _arch({"session_id": "s",
                  "stack_entries": [{"key": k, "value": v} for k, v in stack],
                  "components": [{"name": "web", "responsibility": "UI",
                                  "requirement_ids": ["R-1"]}],
                  "decisions": [{"title": "t", "decision": "d", "rationale": "r",
                                 "driver": "simplicity",
                                 "options_considered": []}],
                  "non_functionals": {}})


def test_capability_overlap_flagged_once():
    arch = _arch_with_stack([("language", "Python"), ("api_framework", "Django"),
                             ("auth", "Firebase Authentication")])
    flags = architecture_challenges(arch)
    auth_flags = [f for f in flags if "auth" in f.lower()]
    assert len(auth_flags) == 1
    assert "Django" in auth_flags[0] and "Firebase" in auth_flags[0]


def test_family_composition_suppressed():
    arch = _arch_with_stack([("language", "JavaScript"), ("backend", "Firebase"),
                             ("push", "Firebase Cloud Messaging")])
    flags = architecture_challenges(arch)
    assert not [f for f in flags if "overlap" in f.lower()]


def test_tooling_spread_flagged():
    arch = _arch_with_stack([("language", "Python"), ("frontend", "React"),
                             ("mobile", "SwiftUI"), ("api_framework", "Django")])
    flags = architecture_challenges(arch)
    assert any("runtime" in f.lower() or "spread" in f.lower() for f in flags)


def test_matrix_rows_carry_capabilities():
    firebase = next(r for r in TECH_ROWS if r["name"] == "Firebase")
    assert set(firebase["capabilities"]) >= {"auth", "database"}
    react = next(r for r in TECH_ROWS if r["name"] == "React")
    assert react["capabilities"] == [] or isinstance(react["capabilities"], list)


# ---------- 3. ADR dedupe ----------

def _slot(i, slug, migration=False):
    return {"decision_id": f"ADR-{i}", "slug": slug, "migration": migration, "authorizes": []}


def _body(title, decision, driver="simplicity", options=None, rationale="r"):
    return {"title": title, "decision": decision, "rationale": rationale,
            "driver": driver, "options_considered": options or [], "baseline_refs": []}


def test_dedupe_merges_same_matrix_row():
    slots = [_slot(1, "use-fastapi"), _slot(2, "adopt-postgres"), _slot(3, "fastapi-api")]
    bodies = {"ADR-1": _body("Adopt FastAPI", "Use FastAPI", options=[{"option": "Flask", "reason": "x"}]),
              "ADR-2": _body("Adopt PostgreSQL", "Use PostgreSQL"),
              "ADR-3": _body("FastAPI for the API", "FastAPI serves the API",
                             driver="founder_constraint", options=[{"option": "Django", "reason": "y"}],
                             rationale="async")}
    notes = _dedupe_adr_slots(slots, bodies)
    assert len(notes) == 1
    assert [s["decision_id"] for s in slots] == ["ADR-1", "ADR-2"]  # gapless
    merged = bodies["ADR-1"]
    assert merged["driver"] == "founder_constraint"
    assert {o["option"] for o in merged["options_considered"]} == {"Flask", "Django"}
    assert "async" in merged["rationale"]
    assert _dedupe_adr_slots(slots, bodies) == []  # idempotent


def test_dedupe_react_vs_react_native_never_merge():
    assert _same_decision_tech(_body("Adopt React", "Use React"),
                               _body("Adopt React Native", "Use React Native")) is None


def test_dedupe_unknown_tech_containment_and_distance():
    assert _same_decision_tech(_body("Adopt Redis", "x"), _body("Adopt Redis for caching", "y"))
    # Redis IS matrix-known; both resolve to the same row. Unknown pairs:
    assert _same_decision_tech(_body("Choose ZyphoraDB", "x"),
                               _body("Adopt ZyphoraDB for data", "y"))
    assert _same_decision_tech(_body("Choose AcmeSearch", "x"),
                               _body("Choose ZyphoraDB", "y")) is None


def test_dedupe_never_merges_migration_slots():
    slots = [_slot(1, "a", migration=True), _slot(2, "b", migration=True)]
    bodies = {"ADR-1": _body("Adopt FastAPI", "Use FastAPI"),
              "ADR-2": _body("FastAPI for the API", "FastAPI serves the API")}
    assert _dedupe_adr_slots(slots, bodies) == []


# ---------- 4. registry refresh (mocked) ----------

class _Resp:
    def __init__(self, payload): self._b = json.dumps(payload).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return self._b


def _fake_opener(mapping):
    def opener(req, timeout=0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        for key, payload in mapping.items():
            if key in url:
                return _Resp(payload)
        raise rr.RefreshError(f"no fake for {url}")
    return opener


NPM_FAKES = {
    "downloads/point/last-week/express": {"downloads": 97_000_000},
    "downloads/point/last-month/express": {"downloads": 400_000_000},
    "registry.npmjs.org/express": {"dist-tags": {"latest": "5.1.0"},
                                   "time": {"5.1.0": "2026-09-01T00:00:00.000Z"}},
}


def test_fetch_npm_metrics_happy():
    m = rr.fetch_npm_metrics("express", opener=_fake_opener(NPM_FAKES),
                             today=date(2026, 9, 13))
    assert m["weekly_downloads"] == 97_000_000
    assert m["version"] == "5.1.0" and m["published"] == "2026-09-01"
    assert m["fetched"] == "2026-09-13"


def test_invalid_registry_data_refused():
    bad = {**NPM_FAKES, "downloads/point/last-week/express": {"downloads": "lots"}}
    with pytest.raises(rr.RefreshError):
        rr.fetch_npm_metrics("express", opener=_fake_opener(bad), today=date(2026, 9, 13))


def test_refresh_failure_keeps_old_metrics():
    data = {"technologies": [
        {"name": "Express", "package": {"registry": "npm", "name": "express"},
         "metrics": {"weekly_downloads": 1, "monthly_downloads": 4, "version": "1.0",
                     "published": "2026-01-01", "fetched": "2026-01-01"}}]}
    def boom(row, today=None): raise rr.RefreshError("offline")
    changes, failures = rr.refresh_artifact(data, fetcher=boom, today=date(2026, 9, 13))
    assert changes == [] and len(failures) == 1
    assert data["technologies"][0]["metrics"]["version"] == "1.0"  # untouched


def test_curated_rows_preserved_by_refresh():
    data = {"technologies": [
        {"name": "Supabase", "note": "curated"},
        {"name": "Express", "package": {"registry": "npm", "name": "express"}}]}
    changes, failures = rr.refresh_artifact(
        data, fetcher=lambda row, today=None: {"weekly_downloads": 5, "monthly_downloads": 20,
                                               "version": "2", "published": "2026-09-01",
                                               "fetched": "2026-09-13"},
        today=date(2026, 9, 13))
    assert len(changes) == 1 and failures == []
    assert data["technologies"][0] == {"name": "Supabase", "note": "curated"}


def test_metrics_diff_movers_and_stale():
    changes = [{"name": "X", "registry": "npm",
                "old": {"weekly_downloads": 100, "version": "1", "published": "2026-08-01",
                        "fetched": "2026-08-01"},
                "new": {"weekly_downloads": 200, "version": "2", "published": "2020-01-01",
                        "fetched": "2026-09-13"}}]
    lines = rr.metrics_diff_lines(changes, [], today=date(2026, 9, 13))
    assert any("100 -> 200" in l and "up" in l for l in lines)
    assert any("version 1 -> 2" in l for l in lines)
    assert any("stale" in l for l in lines)


def test_download_sort_in_prompt_view():
    fw = TECH_CANDIDATES["framework"]
    ranked = [n for n, _ in fw]
    rows = {r["name"]: r for r in TECH_ROWS}
    with_metrics = [n for n in ranked if rows[n].get("metrics")]
    downloads = [rows[n]["metrics"]["weekly_downloads"] for n in with_metrics]
    assert downloads == sorted(downloads, reverse=True)
    # curated rows rank after registry rows
    kinds = [0 if rows[n].get("metrics") else (1 if rows[n].get("service_metrics") else 2)
             for n in ranked]
    assert kinds == sorted(kinds)


# ---------- 5. service ranking sources (fixtures) ----------

STACKSHARE_HTML = "<html><body><span>17.2K</span> <span>stacks</span></body></html>"
DB_ENGINES_HTML = """<table>
<tr><td>1.</td><td></td><td><a>Oracle</a></td></tr>
<tr><td>2.</td><td></td><td><a>PostgreSQL</a></td></tr>
<tr><td>7.</td><td></td><td><a>Redis</a></td></tr>
</table>"""


def test_parse_stackshare_fixture():
    assert ss.parse_stackshare(STACKSHARE_HTML, "firebase") == 17_200
    assert ss.parse_stackshare("<html>nothing</html>", "firebase") is None


def test_parse_db_engines_fixture():
    ranks = ss.parse_db_engines(DB_ENGINES_HTML)
    assert ranks == {"oracle": 1, "postgresql": 2, "redis": 7}
    assert ss.parse_db_engines("<html>nothing</html>") is None


def test_parse_github_fixture():
    assert ss.parse_github({"stargazers_count": 72_000}) == 72_000
    assert ss.parse_github({"message": "Not Found"}) is None


def test_majority_vote_logic():
    data = {"technologies": [
        {"name": "Supabase",
         "service": {"stackshare": "supabase", "github": "supabase/supabase",
                     "db_engines": "Supabase"}}]}
    def opener(req, timeout=0):
        url = req.full_url
        if "stackshare" in url:
            return _Resp_html('<div>30K</div> stacks')
        if "api.github.com" in url:
            return _Resp({"stargazers_count": 80_000})
        if "db-engines" in url:
            return _Resp_html(DB_ENGINES_HTML)
        raise rr.RefreshError("no fake")
    changes, failures, stats = ss.refresh_services(
        data, opener=opener, today=date(2026, 9, 13), _sleep=lambda s: None)
    sm = data["technologies"][0]["service_metrics"]
    # github 2 + stackshare 2 strong, db_engines absent weak -> majority of 3 = strong
    assert sm["strong"] is True and sm["consensus"] > 1
    assert set(sm["contributors"]) == {"github", "stackshare", "db_engines"}
    assert failures == []


class _Resp_html(_Resp):
    def __init__(self, text): self._b = text.encode()


def test_one_source_down_keeps_previous():
    data = {"technologies": [
        {"name": "Redis",
         "service": {"github": "redis/redis"},
         "service_metrics": {"sources": {"github": {"value": 60_000, "score": 2,
                                                    "vote": "strong",
                                                    "fetched": "2026-06-01"}},
                             "contributors": ["github"], "consensus": 2.0,
                             "strong": False, "fetched": "2026-06-01"}}]}
    def boom(req, timeout=0): raise rr.RefreshError("connection refused")
    changes, failures, stats = ss.refresh_services(
        data, opener=boom, today=date(2026, 9, 13), _sleep=lambda s: None)
    assert failures and stats["github"]["failed"] == 1
    gh = data["technologies"][0]["service_metrics"]["sources"]["github"]
    assert gh["value"] == 60_000 and gh["fetched"] == "2026-06-01"  # previous kept


# ---------- 6. schema ----------

def test_committed_artifact_valid():
    data = json.loads(open("src/spec_generator/tech_candidates.json").read())
    assert validate_artifact(data) == []


def test_validate_rejects_bad_metrics_and_dual_source():
    base = json.loads(open("src/spec_generator/tech_candidates.json").read())
    bad = json.loads(json.dumps(base))
    row = next(r for r in bad["technologies"] if r.get("package"))
    row["metrics"]["weekly_downloads"] = -5
    assert any("weekly_downloads" in p for p in validate_artifact(bad))
    dual = json.loads(json.dumps(base))
    row = next(r for r in dual["technologies"] if r.get("package"))
    row["service"] = {"github": "x/y"}
    assert any("never both" in p for p in validate_artifact(dual))
