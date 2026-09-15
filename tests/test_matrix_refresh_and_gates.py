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


def test_dedupe_is_advisory_and_mutates_nothing():
    slots = [_slot(1, "use-fastapi"), _slot(2, "adopt-postgres"), _slot(3, "fastapi-api")]
    bodies = {"ADR-1": _body("Adopt FastAPI", "Use FastAPI"),
              "ADR-2": _body("Adopt PostgreSQL", "Use PostgreSQL"),
              "ADR-3": _body("FastAPI for the API", "FastAPI serves the API")}
    before=(json.loads(json.dumps(slots)),json.loads(json.dumps(bodies)))
    notes = _dedupe_adr_slots(slots, bodies)
    assert len(notes)==1 and notes[0]["kind"]=="possible_duplicate_adr"
    assert (slots,bodies)==before

def test_dedupe_react_vs_react_native_never_merge():
    assert _same_decision_tech(_body("Adopt React", "Use React"),
                               _body("Adopt React Native", "Use React Native")) is None


def test_dedupe_unknown_tech_containment_and_distance():
    assert _same_decision_tech(_body("Adopt Redis", "x"), _body("Adopt Redis for caching", "y"))
    # Redis IS matrix-known; both resolve to the same row. Unknown pairs:
    assert _same_decision_tech(_body("Choose ZyphoraDB data store", "x"),
                               _body("Adopt ZyphoraDB data store", "y"))
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
# mirrors the real 2026-09-13 db-engines markup (unquoted attrs, unclosed <td>s)
DB_ENGINES_HTML = """<table class="dbi">
<tr><td>1.<td class=small>1.<td class="small pad-r">1.<th class="pad-l"><a href="https://db-engines.com/en/system/Oracle">Oracle</a><th class="small pad-r">Relational
<tr><td>2.<td class=small>2.<td class="small pad-r">2.<th class="pad-l"><a href="https://db-engines.com/en/system/PostgreSQL">PostgreSQL</a><th class="small pad-r">Relational
<tr><td>7.<td class=small>7.<td class="small pad-r">7.<th class="pad-l"><a href="https://db-engines.com/en/system/Redis">Redis</a><th class="small pad-r">Key-value
</table>"""


def test_parse_stackshare_fixture():
    assert ss.parse_stackshare(STACKSHARE_HTML, "firebase") == (17_200, "stacks")
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
        data, opener=opener, today=date(2026, 9, 13), _sleep=lambda s: None,
        _browser_factory=lambda log: None)
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


# ---------- 7. fetch robustness (his 2026-09-13 dry run) ----------

def test_explain_fetch_error_classifies():
    import ssl as _ssl
    from urllib.error import HTTPError
    from spec_generator.refresh_registry import explain_fetch_error
    cert = explain_fetch_error(_ssl.SSLError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] x"))
    assert "certificate" in cert and "certifi" in cert and "Certificates.command" in cert
    assert "rate-limited" in explain_fetch_error(HTTPError("u", 429, "x", {}, None))
    assert "blocked" in explain_fetch_error(HTTPError("u", 403, "x", {}, None))
    assert "timed out" in explain_fetch_error(TimeoutError("timed out"))


class _Flaky429:
    """429s N times, then serves the page."""
    def __init__(self, n_fail, html):
        self.left = n_fail; self.html = html; self.calls = 0
    def __call__(self, req, timeout=0, **kw):
        from urllib.error import HTTPError
        self.calls += 1
        if self.left:
            self.left -= 1
            raise HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
        return _Resp_html(self.html)


def test_fetch_page_retries_429_once_then_succeeds():
    opener = _Flaky429(1, "<div>30K</div> stacks")
    html = ss._fetch_page("https://stackshare.io/x", opener, _sleep=lambda s: None)
    assert ss.parse_stackshare(html, "x") == (30_000, "stacks")
    assert opener.calls == 2


def test_fetch_page_persistent_429_classified():
    opener = _Flaky429(5, "never")
    with pytest.raises(rr.RefreshError, match="rate-limited"):
        ss._fetch_page("https://stackshare.io/x", opener, _sleep=lambda s: None)
    assert opener.calls == 2  # one long-backoff retry, not a hammer


def test_stackshare_down_keeps_previous_and_reports():
    data = {"technologies": [
        {"name": "Firebase", "service": {"stackshare": "firebase"},
         "service_metrics": {"sources": {"stackshare": {"value": 50_000, "score": 2,
                                                        "vote": "strong",
                                                        "fetched": "2026-06-01"}},
                             "contributors": ["stackshare"], "consensus": 2.0,
                             "strong": False, "fetched": "2026-06-01"}}]}
    opener = _Flaky429(99, "never")
    changes, failures, stats = ss.refresh_services(
        data, opener=opener, today=date(2026, 9, 13), _sleep=lambda s: None,
        _browser_factory=lambda log: None)
    assert stats["stackshare"]["failed"] == 1
    assert "rate-limited" in failures[0] and "kept previous" in failures[0]
    row = data["technologies"][0]["service_metrics"]["sources"]["stackshare"]
    assert row["value"] == 50_000 and row["fetched"] == "2026-06-01"  # previous kept


def test_cert_error_during_services_is_classified_not_misread():
    import ssl as _ssl
    data = {"technologies": [
        {"name": "PostgreSQL", "service": {"db_engines": "PostgreSQL"}}]}
    def cert_fail(req, timeout=0, **kw):
        raise _ssl.SSLError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate")
    changes, failures, stats = ss.refresh_services(
        data, opener=cert_fail, today=date(2026, 9, 13), _sleep=lambda s: None)
    assert stats["db_engines"]["failed"] == 1
    assert "certificate" in failures[0] and "Certificates.command" in failures[0]


def test_shared_package_fetched_once():
    # Django + Django ORM map to the same pypi package: one fetch, both rows
    data = {"technologies": [
        {"name": "Django", "package": {"registry": "pypi", "name": "django"}},
        {"name": "Django ORM", "package": {"registry": "pypi", "name": "django"}}]}
    calls = []
    def fetcher(row, today=None):
        calls.append(row["name"])
        return {"weekly_downloads": 9, "monthly_downloads": 9, "version": "6",
                "published": "2026-09-01", "fetched": "2026-09-13"}
    changes, failures = rr.refresh_artifact(data, fetcher=fetcher, today=date(2026, 9, 13))
    assert calls == ["Django"] and failures == [] and len(changes) == 2
    assert data["technologies"][1]["metrics"]["version"] == "6"


# ---------- 8. StackShare headless browser path (his 2026-09-13 call) ----------

class _FakeBrowser:
    def __init__(self, data=None, exc=None):
        self.data = data or {}; self.exc = exc; self.fetched = []; self.closed = False
    def fetch(self, slug, name=""):
        self.fetched.append(slug)
        if self.exc:
            raise self.exc
        return self.data
    def close(self):
        self.closed = True


def test_browser_path_used_when_available():
    data = {"technologies": [
        {"name": "Firebase", "service": {"stackshare": "firebase"}}]}
    fake = _FakeBrowser(data={"html": "<div><span>12,400</span> companies reportedly use "
                                      "<b>Firebase</b> in their tech stacks</div>"})
    changes, failures, stats = ss.refresh_services(
        data, today=date(2026, 9, 13), _sleep=lambda s: None,
        _browser_factory=lambda log: fake)
    assert fake.fetched == ["firebase"] and fake.closed
    sm = data["technologies"][0]["service_metrics"]["sources"]["stackshare"]
    assert sm["value"] == 12_400 and sm["score"] == 3 and sm["metric"] == "companies"
    assert failures == []


def test_browser_failure_falls_to_keep_previous_not_crash():
    data = {"technologies": [
        {"name": "Firebase", "service": {"stackshare": "firebase"},
         "service_metrics": {"sources": {"stackshare": {"value": 50_000, "score": 2,
                                                        "vote": "strong",
                                                        "fetched": "2026-06-01"}},
                             "contributors": ["stackshare"], "consensus": 2.0,
                             "strong": False, "fetched": "2026-06-01"}}]}
    fake = _FakeBrowser(exc=rr.RefreshError("bot checkpoint rejected the automated browser"))
    changes, failures, stats = ss.refresh_services(
        data, today=date(2026, 9, 13), _sleep=lambda s: None,
        _browser_factory=lambda log: fake)
    assert stats["stackshare"]["failed"] == 1
    assert data["technologies"][0]["service_metrics"]["sources"]["stackshare"]["value"] == 50_000


def test_no_browser_falls_back_to_headers_with_install_hint():
    data = {"technologies": [
        {"name": "Firebase", "service": {"stackshare": "firebase"}}]}
    logs = []
    def factory(log):
        log("  source stackshare: headless browser not installed - plain HTTP fallback.")
        log("    to enable the browser path: pip install playwright")
        log("    then fetch the browser once: playwright install chromium")
        return None
    opener = _Flaky429(99, "never")
    changes, failures, stats = ss.refresh_services(
        data, opener=opener, today=date(2026, 9, 13), _sleep=lambda s: None,
        log=logs.append, _browser_factory=factory)
    assert any("pip install playwright" in l for l in logs)
    assert any("playwright install chromium" in l for l in logs)
    assert stats["stackshare"]["failed"] == 1  # headers fallback still 429s


def test_vercel_checkpoint_html_classified_not_markup():
    with pytest.raises(rr.RefreshError, match="Vercel bot checkpoint"):
        ss._stackshare_result("<title>Vercel Security Checkpoint</title>", "firebase")


def test_try_start_without_playwright_logs_install_commands(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    logs = []
    result = ss._StackShareBrowser.try_start(logs.append)
    assert result is None
    assert any("pip install playwright" in l for l in logs)
    assert any("playwright install chromium" in l for l in logs)


def test_parse_stackshare_current_phrasings():
    # his 2026-09-13 dry run: pages fetched fine, old 'stacks' phrasing gone
    companies = '<div><span>3226</span> companies reportedly use <b>Firebase</b> in their tech stacks</div>'
    developers = '<div><span>37,412</span> developers on StackShare have stated that they use Firebase</div>'
    assert ss.parse_stackshare(companies, "firebase") == (3226, "companies")
    assert ss.parse_stackshare(developers, "firebase") == (37412, "developers")
    assert ss._stackshare_result(companies, "firebase")["score"] == 2
    assert ss._stackshare_result(developers, "firebase")["score"] == 3


# ---------- 9. StackShare structured-data path (his HAR, 2026-09-13) ----------

HAR_SPARKLINE = ('https://stackshare.io/api/trpc/a,tools.getToolAdoptionSparkline?batch=1&input=x',
                 '[{"result":{"data":{"json":{"pros":[]}}}},'
                 '{"result":{"data":{"json":{"data":[17,23],"total":184,"weeks":52}}}}]')
HAR_TRPC_REQ = ("https://stackshare.io/api/trpc/tools.getToolAdoptionSparkline?batch=1"
                "&input=%7B%220%22%3A%7B%22json%22%3A%7B%22toolId%22%3A116%7D%7D%7D")
HAR_ALGOLIA_REQ = ("https://km8652f2eg-dsn.algolia.net/1/indexes/*/queries"
                   "?x-algolia-api-key=KEY123&x-algolia-application-id=APP9")


def test_extract_tool_id_from_percent_encoded_trpc_url():
    assert ss._extract_tool_id([HAR_TRPC_REQ]) == "116"
    assert ss._extract_tool_id(["https://stackshare.io/firebase"]) is None


def test_extract_sparkline_total_from_har_shape():
    assert ss._extract_sparkline_total([HAR_SPARKLINE]) == 184
    assert ss._extract_sparkline_total([("x/other", "not json")]) is None


def test_extract_algolia_creds_from_page_traffic():
    assert ss._extract_algolia_creds([HAR_ALGOLIA_REQ]) == ("APP9", "KEY123")
    assert ss._extract_algolia_creds(["https://stackshare.io/api/trpc/x"]) is None


def test_result_data_priority_popularity_over_sparkline_over_markup():
    r = ss._stackshare_result_data({"popularity": 45678.9, "stacks_52wk": 184,
                                    "html": "no counts here"}, "firebase")
    assert (r["value"], r["metric"], r["score"]) == (45679, "algolia_popularity", 3)
    r = ss._stackshare_result_data({"stacks_52wk": 184, "html": "x"}, "firebase")
    assert (r["value"], r["metric"], r["score"]) == (184, "stacks_52wk", 1)
    r = ss._stackshare_result_data(
        {"html": "<span>3226</span> companies reportedly use <b>Firebase</b> in their tech stacks"},
        "firebase")
    assert (r["value"], r["metric"]) == (3226, "companies")
    with pytest.raises(rr.RefreshError, match="no adoption data"):
        ss._stackshare_result_data({"html": "<div>bare shell</div>"}, "firebase")


def test_browser_popularity_flows_to_metrics():
    data = {"technologies": [
        {"name": "Firebase", "service": {"stackshare": "firebase"}}]}
    fake = _FakeBrowser(data={"html": "shell", "popularity": 45678.9, "tool_id": "116"})
    changes, failures, stats = ss.refresh_services(
        data, today=date(2026, 9, 13), _sleep=lambda s: None,
        _browser_factory=lambda log: fake)
    sm = data["technologies"][0]["service_metrics"]["sources"]["stackshare"]
    assert sm["value"] == 45679 and sm["metric"] == "algolia_popularity"
    assert failures == []


# ---------- 10. Persisted reasons + calibrated bands (his 2026-09-13 ask) ----------

def test_every_source_persists_metric_and_reason():
    r = ss._github_result({"stargazers_count": 22086}, "postgres/postgres")
    assert r["metric"] == "stars"
    assert r["reason"] == "github stars 22086, band >= 5000 -> score 1 weak"
    r = ss._db_engines_result({"postgresql": 4}, "PostgreSQL")
    assert r["metric"] == "rank"
    assert r["reason"] == "db_engines rank 4, band <= 5 -> score 3 strong"
    r = ss._db_engines_result({"postgresql": 4}, "NotADB")
    assert r["found"] is False and r["value"] is None and r["metric"] == "rank"
    assert "absent from the ranking" in r["reason"]
    r = ss._stackshare_result_data({"popularity": 895.4, "html": "x"}, "supabase")
    assert r["reason"] == ("stackshare algolia_popularity 895, "
                           "band >= 500 -> score 1 weak")


def test_algolia_popularity_bands_calibrated_to_first_interception_run():
    # His 2026-09-13 run, verbatim values:
    expect = [("firebase", 100816, 3), ("jenkins", 140767, 3), ("rails", 45193, 3),
              ("flutter", 43744, 3), ("spring", 11114, 2), ("onesignal", 1075, 1),
              ("supabase", 895, 1), ("fcm", 846, 1), ("buildkite", 581, 1),
              ("clerk", 95, 0)]
    for slug, popularity, score in expect:
        r = ss._stackshare_result_data({"popularity": float(popularity), "html": ""}, slug)
        assert (r["score"], r["metric"]) == (score, "algolia_popularity"), slug


def test_console_line_names_the_metric():
    data = {"technologies": [
        {"name": "Firebase", "service": {"stackshare": "firebase"}}]}
    fake = _FakeBrowser(data={"html": "shell", "popularity": 895.0})
    lines = []
    ss.refresh_services(data, today=date(2026, 9, 13), _sleep=lambda s: None,
                        log=lines.append, _browser_factory=lambda log: fake)
    row_line = next(l for l in lines if l.strip().startswith("stackshare:"))
    assert "algolia_popularity 895" in row_line


def test_github_headers_use_token_when_present(monkeypatch):
    from spec_generator import model_config
    monkeypatch.setattr(model_config, "_DOTENV", {})  # isolate from real .env
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert ss._github_headers() == {}
    monkeypatch.setenv("GITHUB_TOKEN", "tok123")
    assert ss._github_headers() == {"Authorization": "Bearer tok123"}
    monkeypatch.delenv("GITHUB_TOKEN")
    monkeypatch.setattr(model_config, "_DOTENV", {"GITHUB_TOKEN": "from-dotenv"})
    assert ss._github_headers() == {"Authorization": "Bearer from-dotenv"}


def test_403_with_exhausted_quota_is_called_a_rate_limit():
    from urllib.error import HTTPError
    from spec_generator.refresh_registry import explain_fetch_error
    exc = HTTPError("https://api.github.com/repos/x/y", 403, "Forbidden",
                    {"X-RateLimit-Remaining": "0"}, None)
    msg = explain_fetch_error(exc)
    assert "rate-limited" in msg and "GITHUB_TOKEN" in msg
    exc2 = HTTPError("https://stackshare.io/x", 403, "Forbidden", {}, None)
    assert "bot protection" in explain_fetch_error(exc2)
