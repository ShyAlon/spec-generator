"""JEV-D10 content acceptance: deterministic checks on a candidate spec body.

Two rules, one boundary (the item-acceptance validator):

* rule 1, directive echo: the body repeats a pipeline control string (retry or
  format-control text the generator itself emitted).
* rule 2, duplicate body: the body repeats an already accepted sibling body.

Metric: normalized Levenshtein (distance / max normalized length). Duplicates
also use token-shingle Jaccard, which earned a place on the founder's labels
(it ignores paragraph order). Nothing here judges meaning. Echo has no clean
deterministic separator on the founder's labels (quotation versus paraphrase),
so only the extremes are decided here; the middle band is returned as GRAY and
the caller turns it into a finding for Jev or the founder.

Thresholds are PROVISIONAL (28 designed labels). They are named constants with
a version so a change is visible in every stored result. Over-limit work is never a
silent pass and never a stop: duplicate pairs decide on shingle overlap, everything else
becomes a GRAY finding for review. Raw text and hashes are preserved; normalized copies
are comparison-only.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass, field

NORMALIZATION_VERSION = "norm-1"
POLICY_VERSION = "content-acceptance-2"

# --- provisional thresholds (founder-labeled fixture, see tests/fixtures) ----
ECHO_REJECT_MAX = 0.12      # whole-body distance at or below: deterministic echo
ECHO_PASS_MIN = 0.55        # whole-body distance at or above (and no span match): clearly not an echo
ECHO_SPAN_GRAY_MAX = 0.12   # a span this close to a directive is gray, not auto-reject (C-07)
DUP_LEV_MAX = 0.02          # distance at or below: duplicate (labels: yes <= .0050, no >= .0299)
DUP_SHINGLE_MIN = 0.82      # jaccard at or above: duplicate (labels: yes >= .9391, no <= .8182)
SHINGLE_SIZE = 3
MIN_SUBSTANTIVE_CHARS = 20  # review trigger only; never an automatic pass or reject by itself
# --- work limits (placeholders; over-limit: duplicate decides on shingles, other checks go GRAY) --------
MAX_COMPARE_CELLS = 4_000_000   # len(a) * len(b) per Levenshtein call
MAX_SPANS = 400                 # sentence/paragraph spans checked per body
MAX_SIBLINGS = 200              # accepted siblings compared per candidate

PASS, REJECT, GRAY, NOT_CHECKED = "pass", "reject", "gray", "not_checked"

# Versioned catalogue of strings the generator itself emits into model calls.
DIRECTIVE_CATALOGUE_VERSION = 1
DIRECTIVE_CATALOGUE: dict[str, str] = {
    "watchdog-retry": "The prior call exceeded its runtime budget before completing. Return the requested artifact now.",
    "final-response-contract": "FINAL RESPONSE CONTRACT: Return only the JSON value requested above; no prose or markdown fences.",
    "founder-feedback-header": "FOUNDER FEEDBACK ON YOUR PREVIOUS DRAFT (apply it):",
    "review-fixes-header": "REVIEW FIXES - apply EVERY one:",
    "scoped-review-fixes-header": "SCOPED REVIEW FIXES:",
}

# Narrow, versioned list of section headings that legitimately repeat.
SHARED_HEADINGS_VERSION = 1
SHARED_HEADINGS = frozenset({
    "interface", "data model", "behavior", "behaviors", "behaviour", "error cases", "errors",
    "non-functionals", "non-functional requirements", "overview", "purpose", "scope",
})

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_PIPELINE_TITLE = re.compile(r"^\s{0,3}#{1,6}\s+.*\bSPEC-\d+\b", re.I)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_markdown(text: str) -> str:
    """Remove wrapper markup only: fences, heading/quote/list markers, emphasis,
    inline-code ticks. Words, numbers and IDs are never touched."""
    out = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = line.strip()
        if s.startswith("```") or re.fullmatch(r"[-=*_]{3,}", s) or re.fullmatch(r"\|?[\s:|-]+\|?", s) and "-" in s:
            continue
        s = re.sub(r"^\s{0,3}#{1,6}\s+", "", s)
        s = re.sub(r"^\s*>+\s?", "", s)
        s = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", s)
        s = s.replace("|", " ")
        s = re.sub(r"(\*\*|__|\*|`)", "", s)
        out.append(s)
    return "\n".join(out)


def normalize(text: str) -> str:
    """NFKC + case-fold + markdown wrapper removal + collapsed whitespace."""
    t = unicodedata.normalize("NFKC", text or "")
    t = strip_markdown(t)
    return re.sub(r"\s+", " ", t.casefold()).strip()


def levenshtein(a: str, b: str, max_cells: int | None = None) -> int | None:
    """Edit distance, or None when len(a)*len(b) exceeds the work limit."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) * len(b) > (MAX_COMPARE_CELLS if max_cells is None else max_cells):
        return None
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_distance(a: str, b: str, max_cells: int | None = None) -> float | None:
    """distance / max(len). Both empty -> 0.0 (equal); one empty -> 1.0.
    None = over the work limit (not checked)."""
    if not a and not b:
        return 0.0
    d = levenshtein(a, b, max_cells)
    return None if d is None else d / max(len(a), len(b))


def shingles(text: str, n: int = SHINGLE_SIZE) -> set[tuple[str, ...]]:
    toks = text.split()
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def shingle_jaccard(a: str, b: str, n: int = SHINGLE_SIZE) -> float:
    sa, sb = shingles(a, n), shingles(b, n)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def substantive_text(md: str) -> str:
    """Body prose a duplicate check compares: drops the pipeline-owned
    title/ID header and the narrow shared-heading list. Other headings stay."""
    kept = []
    for line in (md or "").replace("\r\n", "\n").split("\n"):
        m = _HEADING.match(line)
        if m:
            if _PIPELINE_TITLE.match(line) or normalize(m.group(1)) in SHARED_HEADINGS:
                continue
        kept.append(line)
    return "\n".join(kept)


def substantive_length(md: str) -> int:
    """Non-heading normalized characters (headings-only bodies count zero)."""
    body = "\n".join(l for l in (md or "").replace("\r\n", "\n").split("\n") if not _HEADING.match(l))
    return len(normalize(body))


def spans(text: str) -> list[str]:
    """Bounded paragraph and sentence spans of the RAW text, normalized."""
    out, seen = [], set()
    for para in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        cands = [para] + re.split(r"(?<=[.!?])\s+", para.strip())
        for c in cands:
            n = normalize(c)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
    return out


@dataclass
class CheckResult:
    rule: str                     # "echo" | "duplicate"
    status: str                   # pass | reject | gray | not_checked
    reason: str
    score: float | None = None    # best (smallest) distance
    shingle: float | None = None
    threshold: dict = field(default_factory=dict)
    catalogue_id: str | None = None
    sibling_id: str | None = None
    matched_span: str | None = None
    matched_offsets: list[int] | None = None   # raw offsets of the matched span, when found
    candidate_sha256: str = ""
    other_sha256: str = ""
    policy_version: str = POLICY_VERSION
    normalization_version: str = NORMALIZATION_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _raw_offsets(raw: str, span_norm: str) -> list[int] | None:
    """Locate the raw text whose normalization equals span_norm (best effort)."""
    for m in re.finditer(r"[^\n]+", raw):
        if normalize(m.group(0)) == span_norm:
            return [m.start(), m.end()]
    return None


def check_echo(body: str, catalogue: dict[str, str] | None = None) -> list[CheckResult]:
    """Rule 1. One result per catalogue entry that is not a clean pass."""
    cat = catalogue or DIRECTIVE_CATALOGUE
    nbody = normalize(body)
    body_hash = sha256(body)
    body_spans = spans(body)
    results: list[CheckResult] = []
    for cid, directive in cat.items():
        nd = normalize(directive)
        base = dict(rule="echo", catalogue_id=cid, candidate_sha256=body_hash,
                    other_sha256=sha256(directive),
                    threshold={"reject_max": ECHO_REJECT_MAX, "pass_min": ECHO_PASS_MIN,
                               "span_gray_max": ECHO_SPAN_GRAY_MAX})
        if nbody == nd:
            results.append(CheckResult(status=REJECT, reason="body equals a pipeline directive after normalization",
                                       score=0.0, matched_span=nd, matched_offsets=[0, len(body)], **base))
            continue
        if len(body_spans) > MAX_SPANS:
            results.append(CheckResult(status=GRAY, reason=f"more than {MAX_SPANS} spans; not compared, needs review", **base))
            continue
        whole = normalized_distance(nd, nbody)
        best, best_span = None, None
        for s in body_spans:
            d = normalized_distance(nd, s)
            if d is None:
                results.append(CheckResult(status=GRAY, reason="span comparison over the work limit; not compared, needs review", **base))
                best = None
                break
            if best is None or d < best:
                best, best_span = d, s
        else:
            pass
        if whole is None:
            results.append(CheckResult(status=GRAY, reason="whole-body comparison over the work limit; not compared, needs review", **base))
            continue
        if whole <= ECHO_REJECT_MAX:
            results.append(CheckResult(status=REJECT, score=whole, matched_span=nbody[:400],
                                       reason=f"whole body is within {ECHO_REJECT_MAX} of a pipeline directive",
                                       matched_offsets=[0, len(body)], **base))
        elif best is not None and best <= ECHO_SPAN_GRAY_MAX:
            results.append(CheckResult(status=GRAY, score=best, matched_span=best_span,
                                       matched_offsets=_raw_offsets(body, best_span),
                                       reason="a span of the body matches a pipeline directive; the body may quote it or may be it",
                                       **base))
        elif whole < ECHO_PASS_MIN:
            results.append(CheckResult(status=GRAY, score=whole, matched_span=best_span,
                                       reason="body is in the unresolved distance band from a pipeline directive", **base))
    return results


def check_duplicate(candidate_id: str, body: str, siblings: dict[str, str]) -> list[CheckResult]:
    """Rule 2. Compare substantive prose against accepted sibling bodies."""
    results: list[CheckResult] = []
    cand = normalize(substantive_text(body))
    chash = sha256(body)
    if len(siblings) > MAX_SIBLINGS:
        return [CheckResult(rule="duplicate", status=GRAY, candidate_sha256=chash,
                            reason=f"more than {MAX_SIBLINGS} siblings; not compared, needs review")]
    short = substantive_length(body) < MIN_SUBSTANTIVE_CHARS
    for sid, other_body in siblings.items():
        if sid == candidate_id:
            continue
        other = normalize(substantive_text(other_body))
        base = dict(rule="duplicate", sibling_id=sid, candidate_sha256=chash, other_sha256=sha256(other_body),
                    threshold={"lev_max": DUP_LEV_MAX, "shingle_min": DUP_SHINGLE_MIN,
                               "min_substantive_chars": MIN_SUBSTANTIVE_CHARS})
        if not cand and not other:
            # nothing substantive on either side: headings only, handled as a short body
            results.append(CheckResult(status=GRAY, reason="no substantive prose to compare (headings only)", **base))
            continue
        if cand == other:
            results.append(CheckResult(status=REJECT, score=0.0, shingle=1.0,
                                       reason="substantive body equals sibling after normalization", **base))
            continue
        if short:
            results.append(CheckResult(
                status=GRAY, reason=f"substantive text under {MIN_SUBSTANTIVE_CHARS} characters; distance not decisive",
                **base))
            continue
        d = normalized_distance(cand, other)
        j = shingle_jaccard(cand, other)
        if d is None:
            # Over the Levenshtein work limit: decide on shingle overlap alone
            # (linear time, calibrated). Never an unchecked reject.
            if j >= DUP_SHINGLE_MIN:
                results.append(CheckResult(status=REJECT, score=None, shingle=j,
                                           reason="near-duplicate of sibling (shingle overlap past threshold; "
                                                  "distance over the work limit)", **base))
            continue
        if d <= DUP_LEV_MAX or j >= DUP_SHINGLE_MIN:
            results.append(CheckResult(status=REJECT, score=d, shingle=j,
                                       reason="near-duplicate of sibling (distance or shingle overlap past threshold)",
                                       **base))
    return results


def worst(results: list[CheckResult]) -> str:
    order = {REJECT: 3, NOT_CHECKED: 2, GRAY: 1, PASS: 0}
    return max((r.status for r in results), key=lambda s: order[s], default=PASS)
