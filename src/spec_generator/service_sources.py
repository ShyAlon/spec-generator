"""Ranking-source refresh for CURATED service rows (2026-09-13 founder
decision): services exist in no package registry, so their freshness comes
from majority vote across 2-3 independent ranking sources. Registry data
carries package rows; this module carries service rows.

Sources:
- stackshare  : the service's StackShare page, "N stacks" adoption count
- db_engines  : the DB-Engines popularity ranking (databases only)
- github      : the GitHub REST API stargazers count, for services with an
                open-source core repo (Supabase, Keycloak, Redis...)

Design rules he accepted:
- GRACEFUL DEGRADATION: an unreachable source or changed markup NEVER
  destroys the artifact - the row keeps its previous values for that
  source and the diff output says so. A fetch error is a failure; a
  fetched page where the service is simply absent is a weak vote.
- MAJORITY VOTE: a source with a strong signal votes "strong"; the row is
  community_strong when a majority of contributing sources (>= 2) agree.
- POLITE FETCHING: one bulk page per ranking source, small delays between
  per-service pages, robots-honoring manual cadence.
- VERBOSE LOGGING: he runs this by hand and reviews the output - every
  source reports fetched/ok/failed and every row reports its contributors.
"""
from __future__ import annotations

import json
import random
import re
import time
from datetime import date
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request

from .refresh_registry import (RefreshError, _get_json, default_opener,
                               explain_fetch_error)

SOURCES = ("stackshare", "db_engines", "github")
STACKSHARE_PAGE = "https://stackshare.io/{slug}"
DB_ENGINES_RANKING = "https://db-engines.com/en/ranking"
GITHUB_REPO = "https://api.github.com/repos/{repo}"

# score bands: (threshold, score) checked high to low -> score in 0..3.
# StackShare's page copy changed (his 2026-09-13 dry run: every row fetched
# fine via the browser but 'no stacks count found'): the adoption signal is
# now 'N companies reportedly use X in their tech stacks' / 'N developers
# on StackShare stated that they use X'. Bands per metric, and the
# contribution records WHICH metric it measured.
_STACKSHARE_BANDS = {"stacks": ((100_000, 3), (10_000, 2), (1_000, 1)),
                     "companies": ((5_000, 3), (1_000, 2), (100, 1)),
                     "developers": ((20_000, 3), (5_000, 2), (500, 1)),
                     # PROVISIONAL (2026-09-13): no live spread to calibrate
                     # against yet - his next dry run prints raw values; tune
                     # bands once the real distribution is visible.
                     "algolia_popularity": ((100_000, 3), (20_000, 2), (2_000, 1)),
                     "stacks_52wk": ((2_000, 3), (400, 2), (40, 1))}
_DB_ENGINES_BANDS = ((5, 3), (20, 2), (50, 1))  # rank position, lower is better
_GITHUB_BANDS = ((100_000, 3), (30_000, 2), (5_000, 1))
STRONG_SCORE = 2  # score >= this votes "strong"


def _band(value: float, bands) -> int:
    for threshold, score in bands:
        if value >= threshold:
            return score
    return 0


# ---------- parsers (pure functions over fetched text; fixture-tested) ----------

def parse_stackshare(html: str, slug: str) -> tuple[int, str] | None:
    """Adoption signal from a service page + WHICH metric was found:
    ('3226', 'companies') / ('37412', 'developers') / ('17200', 'stacks'
    - the legacy phrasing). None = markup changed (the refresh treats that
    as a source failure and keeps the previous value)."""
    for pattern, metric in (
            (r'([\d,]+)(?:\s*<[^>]+>)*\s+companies reportedly use', "companies"),
            (r'([\d,]+)(?:\s*<[^>]+>)*\s+developers on StackShare', "developers"),
            (r'([\d,.]+)\s*(K|M)?(?:\s*<[^>]+>)*\s*stacks\b', "stacks")):
        m = re.search(pattern, html, re.I)
        if m:
            value = float(m.group(1).replace(",", ""))
            if metric == "stacks":
                mult = {"K": 1_000, "M": 1_000_000}.get((m.group(2) or "").upper(), 1)
                value *= mult
            return int(value), metric
    return None


def parse_db_engines(html: str) -> dict[str, int] | None:
    """name (lowercased) -> rank from the ranking table. None = markup
    changed. A name ABSENT from a well-formed table is genuinely unranked."""
    # Real markup (verified live 2026-09-13): '<tr><td>4.<td class=small>4.
    # <td class="small pad-r">4.<th class="pad-l"><a href=".../system/PostgreSQL">
    # PostgreSQL</a>' - unquoted attrs, unclosed <td>s, the engine name only
    # in the /en/system/ link.
    ranks: dict[str, int] = {}
    for m in re.finditer(
            r'<tr[^>]*><td[^>]*>\s*(\d+)\..{0,500}?/en/system/[^"]+">([^<]+)</a>',
            html, re.S):
        ranks.setdefault(m.group(2).strip().lower(), int(m.group(1)))
    return ranks or None


def parse_github(payload) -> int | None:
    """Stargazers from the repo API payload. None = unexpected shape."""
    if isinstance(payload, dict) and isinstance(payload.get("stargazers_count"), int):
        return payload["stargazers_count"]
    return None


# ---------- StackShare structured-data extraction (his HAR, 2026-09-13) ----------
# The /firebase document is a bare Next.js shell (zero counts, zero mentions);
# the SEO copy ('3226 companies reportedly use...') is crawler-served only.
# Real data flows through tRPC endpoints + Algolia inside the browser:
#   tools.getToolAdoptionSparkline (toolId) -> {'data': [...], 'total': N, 'weeks': 52}
#   Algolia Search_production popularity score (public search creds captured
#   from the page's own requests).

def _extract_tool_id(request_urls: list[str]) -> str | None:
    from urllib.parse import unquote
    for u in request_urls:
        m = re.search(r'"toolId"\s*:\s*(\d+)', unquote(u))
        if m:
            return m.group(1)
    return None


def _extract_sparkline_total(responses: list[tuple[str, str]]) -> int | None:
    for u, body in responses:
        if "getToolAdoptionSparkline" not in u:
            continue
        try:
            payload = json.loads(body)
        except Exception:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            j = ((item or {}).get("result") or {}).get("data", {}).get("json")
            if isinstance(j, dict) and isinstance(j.get("total"), int):
                return j["total"]
    return None


def _extract_algolia_creds(request_urls: list[str]) -> tuple[str, str] | None:
    from urllib.parse import parse_qs, urlparse
    for u in request_urls:
        if "algolia" not in u:
            continue
        q = parse_qs(urlparse(u).query)
        if q.get("x-algolia-api-key") and q.get("x-algolia-application-id"):
            return (q["x-algolia-application-id"][0], q["x-algolia-api-key"][0])
    return None


def _stackshare_result_data(data: dict, slug: str) -> dict:
    """Pick the best available metric: Algolia popularity > 52-week new-stack
    total > markup copy (companies/developers/legacy stacks)."""
    html = data.get("html", "")
    if ("Vercel Security Checkpoint" in html or "Failed to verify your browser" in html
            or "cf-chl" in html or "Just a moment" in html):
        raise RefreshError("StackShare's bot checkpoint rejected the fetch")
    if data.get("popularity") is not None:
        value, metric = int(round(data["popularity"])), "algolia_popularity"
    elif data.get("stacks_52wk") is not None:
        value, metric = data["stacks_52wk"], "stacks_52wk"
    else:
        parsed = parse_stackshare(html, slug)
        if parsed is None:
            raise RefreshError("page fetched but no adoption data found "
                               "(shell-only page, no tRPC/Algolia capture)")
        value, metric = parsed
    score = _band(value, _STACKSHARE_BANDS[metric])
    return {"found": True, "value": value, "metric": metric, "score": score,
            "vote": "strong" if score >= STRONG_SCORE else "weak"}


# ---------- source fetchers ----------

def _stackshare_result(html_text: str, slug: str) -> dict:
    if "Vercel Security Checkpoint" in html_text or "Failed to verify your browser" in html_text:
        raise RefreshError("StackShare's Vercel bot checkpoint rejected the fetch")
    parsed = parse_stackshare(html_text, slug)
    if parsed is None:
        raise RefreshError("page fetched but no adoption count found - markup changed?")
    value, metric = parsed
    score = _band(value, _STACKSHARE_BANDS[metric])
    return {"found": True, "value": value, "metric": metric, "score": score,
            "vote": "strong" if score >= STRONG_SCORE else "weak"}


def _github_result(payload, repo: str) -> dict:
    stars = parse_github(payload)
    if stars is None:
        raise RefreshError("API payload without stargazers_count - shape changed?")
    score = _band(stars, _GITHUB_BANDS)
    return {"found": True, "value": stars, "score": score,
            "vote": "strong" if score >= STRONG_SCORE else "weak"}


def _db_engines_result(ranks: dict[str, int], name: str) -> dict:
    rank = ranks.get(name.lower())
    if rank is None:
        # well-formed ranking, service absent: genuinely unranked, not a failure
        return {"found": False, "value": None, "score": 0, "vote": "weak"}
    # rank position: lower is better, so the generic >= band helper does not apply
    score = next((sc for hi, sc in _DB_ENGINES_BANDS if rank <= hi), 0)
    return {"found": True, "value": rank, "score": score,
            "vote": "strong" if score >= STRONG_SCORE else "weak"}


# StackShare throttles hard (his 2026-09-13 dry run: 429 on every row even
# from a residential IP): fetch like a browser, pace at seconds - not
# fractions - with jitter, and give each page one long-backoff retry.
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_page(url: str, opener, _sleep, retries_429: int = 1) -> str:
    """One page fetch with a long-backoff retry on 429. Anything else fails
    immediately with a classified cause."""
    for attempt in range(1 + retries_429):
        try:
            req = Request(url, headers=_BROWSER_HEADERS)
            with opener(req, timeout=30) as resp:
                return resp.read().decode("utf-8", "replace")
        except HTTPError as exc:
            if exc.code == 429 and attempt < retries_429:
                _sleep(30 + random.uniform(0, 15))  # long backoff, then ONE retry
                continue
            raise RefreshError(explain_fetch_error(exc)) from exc
        except RefreshError:
            raise
        except Exception as exc:
            raise RefreshError(explain_fetch_error(exc)) from exc
    raise RefreshError("unreachable")  # pragma: no cover


class _StackShareBrowser:
    """Headless Chromium for StackShare ONLY (his 2026-09-13 call: an
    automated Chrome is fine - he runs the refresh manually from his own
    Mac on a residential IP). npm/PyPI/GitHub/DB-Engines stay on plain
    HTTP - they work, no browser cost. One browser session serves every
    stackshare row, with the same polite pacing as the headers path.
    Unavailable (Playwright or the Chromium binary missing) = fall back to
    the headers path; the install commands print ONCE per run."""

    def __init__(self, pw, browser, context):
        self._pw, self._browser, self._ctx = pw, browser, context

    @classmethod
    def try_start(cls, log):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            log("  source stackshare: headless browser not installed - plain HTTP fallback.")
            log("    to enable the browser path: pip install playwright")
            log("    then fetch the browser once: playwright install chromium")
            return None
        pw = None
        try:
            pw = sync_playwright().start()
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(user_agent=_BROWSER_HEADERS["User-Agent"],
                                          locale="en-US")
            log("  source stackshare: headless Chromium session started")
            return cls(pw, browser, context)
        except Exception as exc:
            log(f"  source stackshare: browser failed to start ({exc}) - plain HTTP fallback.")
            log("    if the browser binary is missing: playwright install chromium")
            try:
                if pw:
                    pw.stop()
            except Exception:
                pass
            return None

    def fetch(self, slug: str, name: str) -> dict:
        """Load the tool page, INTERCEPTING its own structured traffic (his
        HAR, 2026-09-13: the document is a bare shell; data flows via tRPC +
        Algolia). Returns {'html', 'tool_id', 'stacks_52wk', 'popularity'}
        with whatever was captured; _stackshare_result_data picks the best
        metric, markup copy as last resort."""
        captured_requests: list[str] = []
        captured_responses: list[tuple[str, str]] = []
        page = self._ctx.new_page()
        page.on("request", lambda req: captured_requests.append(req.url))

        def _on_response(resp):
            u = resp.url
            if "/api/trpc/" in u or "algolia" in u:
                try:
                    captured_responses.append((u, resp.text()[:200_000]))
                except Exception:
                    pass
        page.on("response", _on_response)
        try:
            for attempt in (1, 2):  # one retry: his first run saw stray goto timeouts
                try:
                    page.goto(STACKSHARE_PAGE.format(slug=quote(slug)),
                              wait_until="domcontentloaded", timeout=45000)
                    break
                except Exception as exc:
                    if attempt == 2 or "Timeout" not in type(exc).__name__ + str(exc):
                        raise
                    page.wait_for_timeout(4000)
            page.wait_for_timeout(2000 + random.uniform(0, 1000))  # let tRPC fire
            data = {"html": page.content()}
        except Exception as exc:
            raise RefreshError(f"browser fetch failed: {exc}") from exc
        finally:
            page.close()
        data["tool_id"] = _extract_tool_id(captured_requests)
        total = _extract_sparkline_total(captured_responses)
        if total is not None:
            data["stacks_52wk"] = total
        creds = _extract_algolia_creds(captured_requests)
        if creds:
            pop = self._algolia_popularity(creds, slug, name)
            if pop is not None:
                data["popularity"] = pop
        if "popularity" not in data and "stacks_52wk" not in data and data["tool_id"]:
            total = self._sparkline_total(data["tool_id"])
            if total is not None:
                data["stacks_52wk"] = total
        return data

    def _algolia_popularity(self, creds: tuple[str, str], slug: str, name: str) -> float | None:
        """Query the site's own search index from INSIDE the browser context
        (same-origin fetch carries the session; the search-only creds are the
        page's own, captured from its traffic - never hardcoded)."""
        app, key = creds
        script = """async ([app, key, slug, name]) => {
            const body = {requests: [{indexName: 'Search_production', query: name,
                hitsPerPage: 8, filters: 'type:Service',
                attributesToRetrieve: ['name', 'canonical_url', 'popularity']}]};
            const resp = await fetch(`https://${app}-dsn.algolia.net/1/indexes/*/queries`, {
                method: 'POST',
                headers: {'content-type': 'application/json',
                          'x-algolia-api-key': key, 'x-algolia-application-id': app},
                body: JSON.stringify(body)});
            if (!resp.ok) return null;
            const data = await resp.json();
            const hits = (data.results && data.results[0] && data.results[0].hits) || [];
            const exact = hits.find(h => h.canonical_url === '/' + slug) || hits[0];
            return exact && typeof exact.popularity === 'number' ? exact.popularity : null;
        }"""
        page = self._ctx.new_page()
        try:
            page.goto(STACKSHARE_PAGE.format(slug=quote(slug)),
                      wait_until="domcontentloaded", timeout=45000)
            return page.evaluate(script, [app, key, slug, name])
        except Exception:
            return None
        finally:
            page.close()

    def _sparkline_total(self, tool_id: str) -> int | None:
        page = self._ctx.new_page()
        try:
            page.goto("https://stackshare.io/", wait_until="domcontentloaded", timeout=45000)
            return page.evaluate("""async (toolId) => {
                const input = encodeURIComponent(JSON.stringify(
                    {'0': {json: {toolId: Number(toolId), weeks: 52}}}));
                const resp = await fetch(
                    `/api/trpc/tools.getToolAdoptionSparkline?batch=1&input=${input}`);
                if (!resp.ok) return null;
                const data = await resp.json();
                const j = data[0] && data[0].result && data[0].result.data &&
                          data[0].result.data.json;
                return j && typeof j.total === 'number' ? j.total : null;
            }""", tool_id)
        except Exception:
            return None
        finally:
            page.close()

    def close(self):
        for closer in (self._ctx.close, self._browser.close, self._pw.stop):
            try:
                closer()
            except Exception:
                pass


def refresh_services(data: dict, opener=default_opener, today: date | None = None,
                     log=lambda msg: None, _sleep=time.sleep,
                     exclude: frozenset[str] = frozenset(),
                     _browser_factory=_StackShareBrowser.try_start):
    """Refresh service_metrics on rows carrying a ``service`` mapping.
    Returns (changes, failures). Per source per row: a fresh result REPLACES
    that source's contribution; a failure KEEPS the previous contribution
    and is reported. Consensus = mean score of contributing sources; the row
    is community-strong when a majority of >=2 contributing sources vote
    strong."""
    today = today or date.today()
    changes, failures = [], []
    rows = [r for r in data.get("technologies", [])
            if isinstance(r, dict) and r.get("service")
            and set(r["service"]) - set(exclude)]
    # per-source accounting feeds the script's anomaly guards and summary
    stats: dict[str, dict] = {}
    for row in rows:
        for src in row["service"]:
            if src in exclude:
                continue
            st = stats.setdefault(src, {"mapped": 0, "ok": 0, "failed": 0, "flipped": 0})
            st["mapped"] += 1

    # bulk sources first: one polite fetch serves every mapped row
    db_rows = ([r for r in rows if "db_engines" in r["service"]]
               if "db_engines" not in exclude else [])
    db_ranks: dict[str, int] | None = None
    if db_rows:
        log(f"  source db_engines: fetching ranking page ({len(db_rows)} rows mapped)...")
        try:
            db_ranks = parse_db_engines(_fetch_page(DB_ENGINES_RANKING, opener, _sleep))
            if db_ranks is None:
                raise RefreshError("ranking page fetched but no table found - markup changed?")
            log(f"  source db_engines: ok ({len(db_ranks)} entries parsed)")
        except Exception as exc:
            stats["db_engines"]["failed"] += len(db_rows)
            failures.append(f"db_engines: source failed ({exc}); "
                            f"kept previous values for {len(db_rows)} rows")
            log(f"  source db_engines: FAILED - kept previous values ({exc})")
        _sleep(1.0)

    browser = None
    if any("stackshare" in r["service"] and "stackshare" not in exclude for r in rows):
        browser = _browser_factory(log)
    try:
        _refresh_service_rows(rows, exclude, db_ranks, stats, changes, failures,
                              today, log, _sleep, opener, browser)
    finally:
        if browser:
            browser.close()
    return changes, failures, stats


def _refresh_service_rows(rows, exclude, db_ranks, stats, changes, failures,
                          today, log, _sleep, opener, browser):
    for i, row in enumerate(rows):
        name = row["name"]
        mapping = row["service"]
        sm = row.setdefault("service_metrics", {"sources": {}})
        sources = sm.setdefault("sources", {})
        contributors_changed = []
        for source, ident in mapping.items():
            if source in exclude:
                continue
            log(f"  {name}: {source} ({ident})...")
            try:
                if source == "stackshare":
                    if browser is not None:
                        result = _stackshare_result_data(browser.fetch(ident, name), ident)
                    else:
                        result = _stackshare_result(
                            _fetch_page(STACKSHARE_PAGE.format(slug=quote(ident)),
                                        opener, _sleep), ident)
                    _sleep(random.uniform(3.0, 6.0))  # seconds, jittered: their 429 wall
                elif source == "github":
                    result = _github_result(
                        _get_json(GITHUB_REPO.format(repo=quote(ident)), opener), ident)
                    _sleep(0.5)
                elif source == "db_engines":
                    if db_ranks is None:
                        continue  # source-level failure already reported
                    result = _db_engines_result(db_ranks, ident)
                else:
                    continue
            except Exception as exc:
                stats[source]["failed"] += 1
                cause = explain_fetch_error(exc) if not isinstance(exc, RefreshError) else str(exc)
                failures.append(f"{name}: {source} failed ({cause}); kept previous value")
                log(f"    {source}: FAILED - kept previous ({cause})")
                continue
            stats[source]["ok"] += 1
            old = sources.get(source)
            result = {**result, "fetched": today.isoformat()}
            if old and old.get("vote") != result["vote"]:
                stats[source]["flipped"] += 1
            sources[source] = result
            # the review surface shows MEANINGFUL change: first fetch, score
            # move, or vote flip - a 28555 -> 28556 star drift is stored
            # silently, not printed.
            if old is None or old.get("score") != result["score"] or old.get("vote") != result["vote"]:
                old_desc = (f"{old.get('value')} (score {old.get('score')})"
                            if old else "none")
                contributors_changed.append(
                    f"{source}: {old_desc} -> {result['value']} (score {result['score']})")
            log(f"    {source}: {result['value']} score {result['score']} vote {result['vote']}")
        contributing = [s for s, r in sources.items() if r.get("fetched")]
        strong_votes = sum(1 for s in contributing if sources[s].get("vote") == "strong")
        n = len(contributing)
        sm["contributors"] = sorted(contributing)
        sm["consensus"] = round(sum(sources[s].get("score", 0) for s in contributing) / n, 2) if n else None
        sm["strong"] = n >= 2 and strong_votes * 2 > n
        sm["fetched"] = today.isoformat()
        if contributors_changed:
            changes.append({"name": name, "changes": contributors_changed,
                            "consensus": sm["consensus"], "strong": sm["strong"],
                            "contributors": sm["contributors"]})
        if i and i % 8 == 7:
            _sleep(2.0)  # pacing breather between batches of services


def services_diff_lines(changes: list[dict], failures: list[str]) -> list[str]:
    """Per-row change log: contributors, consensus moves, strong flips."""
    lines = []
    for c in changes:
        verdict = "STRONG" if c["strong"] else "not-strong"
        lines.append(f"~ {c['name']}: consensus {c['consensus']} ({verdict}, "
                     f"{len(c['contributors'])} sources: {', '.join(c['contributors'])})")
        for detail in c["changes"]:
            lines.append(f"    {detail}")
    for f in failures:
        lines.append(f"x {f}")
    return lines or ["(no service-metric changes)"]
