"""Registry-backed metrics refresh for the tech candidate matrix
(2026-09-13 founder decision): numbers come from REAL registry APIs -
npm and PyPI - never from a model, so the matrix cannot go stale at the
model's knowledge cutoff.

Two-sourced matrix:
- rows with a ``package`` mapping are registry-backed: the refresh script
  rewrites ONLY their ``metrics`` block from live API data;
- rows without one are curated services (Supabase, Firebase, HealthKit,
  SwiftUI, DynamoDB...): the refresh preserves them byte-for-byte, and the
  artifact's ``as_of`` date marks when a human last reviewed them.

Download counts are his primary sort key; publish date gives freshness.
"""
from __future__ import annotations

import json
import ssl
import time
from pathlib import Path
from datetime import date
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def _ssl_context():
    """His 2026-09-13 dry run hit CERTIFICATE_VERIFY_FAILED on db-engines
    from his Mac - TWO stacked causes handled here:
    1. python.org macOS Python ships without root certs: prefer certifi's
       CA bundle (present in any pip-built toolchain), fall back to the
       system store.
    2. db-engines serves an INCOMPLETE chain: its leaf is signed by Let's
       Encrypt's 2025-generation YR1/YE* intermediates and the server does
       not send them (browsers mask this via AIA fetching; Python does
       not). letsencrypt_y_roots.pem ships those intermediates + Root YR,
       fetched from LE's own AIA endpoints (i.lencr.org).
    If verification STILL fails, explain_fetch_error says how to fix the
    LOCAL store - a local-cert failure must never read as 'site blocked'."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    extra = Path(__file__).with_name("letsencrypt_y_roots.pem")
    if extra.exists():
        ctx.load_verify_locations(cafile=str(extra))
    return ctx


_CONTEXT = _ssl_context()


def default_opener(req, timeout=20):
    """The production opener: real CA bundle, so macOS python.org installs
    verify without system surgery."""
    return urlopen(req, timeout=timeout, context=_CONTEXT)


def explain_fetch_error(exc: Exception) -> str:
    """Classify a fetch failure into the cause + next action he should see.
    'Local cert store broken' and 'site is blocking us' are DIFFERENT
    failures with different fixes - the log must never blur them."""
    text = f"{exc}"
    if isinstance(exc, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in text:
        return ("local certificate store can't verify HTTPS "
                "(python.org macOS Python ships without root certs) - run once: "
                "'/Applications/Python 3.x/Install Certificates.command', or: "
                "python3 -m pip install certifi")
    if isinstance(exc, HTTPError):
        if exc.code == 429:
            return "rate-limited by the site (HTTP 429) - it is throttling automated fetches"
        if exc.code == 403:
            # APIs that send rate-limit headers (GitHub): a 403 with the
            # quota exhausted is a RATE LIMIT, not bot protection - his
            # 2026-09-13 run hit this after repeated anonymous refreshes
            # (60/hour). The fix is different, so the message must be too.
            remaining = (exc.headers or {}).get("X-RateLimit-Remaining")
            if remaining == "0":
                return ("rate-limited by the API (HTTP 403, hourly quota exhausted) - "
                        "set GITHUB_TOKEN in .env for 5000/hour, or wait for the reset")
            return "blocked by the site (HTTP 403) - bot protection"
        if exc.code == 404:
            return "not found (HTTP 404) - the identifier or URL shape is wrong"
        return f"HTTP {exc.code} from the site"
    if isinstance(exc, TimeoutError) or "timed out" in text:
        return "timed out - network or the site is slow right now"
    return text or type(exc).__name__

STALE_DAYS = 18 * 30  # ~18 months without a publish reads as stale

NPM_DOWNLOADS = "https://api.npmjs.org/downloads/point/{period}/{pkg}"
NPM_PACKUMENT = "https://registry.npmjs.org/{pkg}"
NPM_SEARCH = "https://api.npms.io/v2/search?q={q}&size=10"
PYPI_JSON = "https://pypi.org/pypi/{pkg}/json"
PYPISTATS = "https://pypistats.org/api/packages/{pkg}/recent"

# npm keyword searches used to surface leading packages the matrix does not
# know yet (discovery output is advisory - a human still adds the row).
DISCOVERY_QUERIES = {
    "framework": "keywords:web-framework",
    "mobile_framework": "keywords:mobile-framework",
    "visualization": "keywords:chart",
    "orm": "keywords:orm",
    "queue": "keywords:job-queue",
}


class RefreshError(Exception):
    """One package's fetch or payload was unusable; the row keeps its old
    metrics and the failure is reported, never silently zeroed."""


def _get_json(url: str, opener, headers: dict | None = None, _sleep=time.sleep) -> dict:
    req = Request(url, headers={"User-Agent": "vmodel-spec-generator", **(headers or {})})
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with opener(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RefreshError("registry returned a non-object payload")
            return payload
        except HTTPError as exc:
            last_exc = exc
            if exc.code in (429, 500, 502, 503) and attempt < 2:
                _sleep(2 + 3 * attempt)  # rate limit / transient: back off and retry
                continue
            break
        except RefreshError:
            raise
        except Exception as exc:  # network, JSON - all land here
            last_exc = exc
            break
    raise RefreshError(explain_fetch_error(last_exc)) from last_exc


def _require_int(value, label: str) -> int:
    # bool is an int subclass - refuse it explicitly (True == 1 downloads?)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RefreshError(f"{label} is not a non-negative integer: {value!r}")
    return value


def _require_iso(value, label: str) -> str:
    text = str(value or "")[:10]
    try:
        date.fromisoformat(text)
    except ValueError:
        raise RefreshError(f"{label} is not an ISO timestamp: {value!r}") from None
    return text


def fetch_npm_metrics(package: str, opener=default_opener, today: date | None = None) -> dict:
    today = today or date.today()
    week = _get_json(NPM_DOWNLOADS.format(period="last-week", pkg=quote(package)), opener)
    month = _get_json(NPM_DOWNLOADS.format(period="last-month", pkg=quote(package)), opener)
    # Full packument (the abbreviated install-v1 format drops the time map):
    # time[latest] is the publish date of the CURRENT release - the real
    # freshness signal; time.modified (any publish at all) is the fallback.
    meta = _get_json(NPM_PACKUMENT.format(pkg=quote(package)), opener)
    latest = str((meta.get("dist-tags") or {}).get("latest") or "").strip()
    if not latest:
        raise RefreshError(f"npm packument for {package!r} has no dist-tags.latest")
    return {
        "weekly_downloads": _require_int(week.get("downloads"), "npm weekly downloads"),
        "monthly_downloads": _require_int(month.get("downloads"), "npm monthly downloads"),
        "version": latest,
        "published": _require_iso((meta.get("time") or {}).get(latest)
                                  or (meta.get("time") or {}).get("modified"),
                                  "npm publish time"),
        "fetched": today.isoformat(),
    }


def fetch_pypi_metrics(package: str, opener=default_opener, today: date | None = None) -> dict:
    today = today or date.today()
    meta = _get_json(PYPI_JSON.format(pkg=quote(package)), opener)
    stats = _get_json(PYPISTATS.format(pkg=quote(package)), opener)
    info = meta.get("info") or {}
    version = str(info.get("version") or "").strip()
    if not version:
        raise RefreshError(f"PyPI payload for {package!r} has no info.version")
    releases = meta.get("releases") or {}
    files = releases.get(version) or []
    if not files or not isinstance(files, list):
        raise RefreshError(f"PyPI payload for {package!r} has no files for {version}")
    data = stats.get("data") or {}
    return {
        "weekly_downloads": _require_int(data.get("last_week"), "pypistats last_week"),
        "monthly_downloads": _require_int(data.get("last_month"), "pypistats last_month"),
        "version": version,
        "published": _require_iso(files[0].get("upload_time"), "PyPI upload_time"),
        "fetched": today.isoformat(),
    }


def fetch_metrics(row: dict, opener=default_opener, today: date | None = None) -> dict:
    pkg = row.get("package") or {}
    registry = pkg.get("registry")
    if registry == "npm":
        return fetch_npm_metrics(pkg["name"], opener, today)
    if registry == "pypi":
        return fetch_pypi_metrics(pkg["name"], opener, today)
    raise RefreshError(f"{row.get('name')}: no registry mapping (curated row)")


def refresh_artifact(data: dict, fetcher=fetch_metrics, today: date | None = None):
    """Rewrite metrics for registry-backed rows in place. Curated rows are
    never touched. A failed fetch keeps the row's OLD metrics (transient
    outages must not zero the matrix) and is reported. Returns
    (changes, failures)."""
    changes, failures = [], []
    rows = [r for r in data.get("technologies", [])
            if isinstance(r, dict) and r.get("package")]
    cache: dict = {}  # Django + Django ORM map to the SAME pypi package: one fetch,
    # both rows - pypistats 429s the duplicate hit (seen live 2026-09-13)
    for i, row in enumerate(rows):
        pkg_key = (row["package"]["registry"], row["package"]["name"])
        if i and pkg_key not in cache:
            time.sleep(0.3)  # registry politeness: ~44 rows spread over ~15s
        try:
            if pkg_key not in cache:
                cache[pkg_key] = fetcher(row, today=today)
            new_metrics = dict(cache[pkg_key])
        except Exception as exc:
            failures.append(f"{row.get('name')}: {exc}")
            continue
        old_metrics = row.get("metrics")
        row["metrics"] = new_metrics
        changes.append({"name": row["name"], "registry": row["package"]["registry"],
                        "old": old_metrics, "new": new_metrics})
    return changes, failures


def metrics_diff_lines(changes: list[dict], failures: list[str],
                       today: date | None = None) -> list[str]:
    """The review surface: download movers (>=20% weekly), version bumps,
    stale packages, unrefreshed rows, failures."""
    today = today or date.today()
    lines = []
    for c in changes:
        n = c["new"]
        tag = f"{c['name']} ({c['registry']})"
        o = c["old"]
        if o is None:
            lines.append(f"~ {tag}: first refresh - {n['weekly_downloads']:,}/wk, "
                         f"v{n['version']}, published {n['published']}")
        else:
            ow, nw = o.get("weekly_downloads", 0), n["weekly_downloads"]
            if ow and abs(nw - ow) * 5 >= ow:  # >=20% swing
                lines.append(f"~ {tag}: downloads {ow:,} -> {nw:,}/wk "
                             f"({'up' if nw > ow else 'down'} {abs(nw - ow) * 100 // ow}%)")
            if o.get("version") != n["version"]:
                lines.append(f"~ {tag}: version {o.get('version')} -> {n['version']}")
        try:
            if (today - date.fromisoformat(n["published"])).days > STALE_DAYS:
                lines.append(f"! {tag}: stale - last publish {n['published']} "
                             f"(>{STALE_DAYS // 30} months ago)")
        except ValueError:
            lines.append(f"! {tag}: unreadable publish date {n['published']!r}")
    for f in failures:
        lines.append(f"x {f}")
    return lines or ["(no metric changes)"]


def discover_candidates(data: dict, opener=default_opener, today: date | None = None,
                        fetcher=fetch_npm_metrics) -> list[str]:
    """Advisory discovery: top npm packages per searched kind that the
    matrix does not know, with real download counts. Never writes - the
    output tells him which curated row to add by hand (or delegate)."""
    today = today or date.today()
    known_packages = {row["package"]["name"].lower()
                      for row in data.get("technologies", [])
                      if isinstance(row, dict) and row.get("package")}
    known_aliases = {a for row in data.get("technologies", [])
                     if isinstance(row, dict) for a in row.get("aliases", [])}
    lines = []
    for kind, query in DISCOVERY_QUERIES.items():
        try:
            result = _get_json(NPM_SEARCH.format(q=quote(query)), opener)
        except RefreshError as exc:
            lines.append(f"x discovery {kind}: {exc}")
            continue
        found = []
        for hit in result.get("results") or []:
            name = str((hit.get("package") or {}).get("name") or "").strip()
            if not name or name.lower() in known_packages or name.lower() in known_aliases:
                continue
            found.append(name)
            if len(found) >= 3:
                break
        for name in found:
            try:
                m = fetcher(name, opener, today)
                lines.append(f"+? {kind}: {name} - {m['weekly_downloads']:,}/wk, "
                             f"v{m['version']}, published {m['published']} "
                             f"(not in matrix; add a curated row to adopt)")
            except RefreshError:
                lines.append(f"+? {kind}: {name} (not in matrix; downloads lookup failed)")
    return lines
