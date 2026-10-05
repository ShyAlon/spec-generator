"""Review-noise classes: deterministic phrase patterns for spec/test/systemic review findings the
founder rejected as noise (measured on the 2026-10-02 ledgers). Shared by the review node and
Ledger.refresh so a re-raised finding does not come back open. Requirement-scope findings are never
touched. A finding that states a contradiction, an absence or an orphan is rescued and stays open,
except class G, which is checked before the rescue."""
from __future__ import annotations

import re as _re

# Rescue: contradiction / absence / orphan wording keeps a finding open. ("restated requirement" no longer
# rescues: class G matches those first and the phrase was the reason the noise survived.)
_NOISE_RESCUE = _re.compile(
    r"contradict|conflict|inconsisten|mismatch|does not match|doesn't match|\bbut the (architecture|srs|requirement)|"
    r"\binstead of\b|\b(missing|omits?|omitted|absent|lacks?|no mention|not mentioned|never (states|mentions))\b|"
    r"\borphan|untraced|unreferenced", _re.I)

# Class G: "references X without restating X". Checked BEFORE the rescue. spec/test/systemic.
_CLASS_G = ("G:reference-without-restating", _re.compile(
    r"\b(without|avoid(ing)?|to avoid|do not|don't|not) (re)?stat\w+\b.{0,40}\b(requirement|criteri\w+|AC-\d+|R-\d+|its text|verbatim)|"
    r"\breferences?\b.{0,40}\b(requirement|AC-\d+|R-\d+|SPEC-\d+)\b.{0,30}\b(by id|instead)\b|"
    r"\bremove\b.{0,120}\bto avoid restating\b|\bremove restated requirement text", _re.I))
_G_SCOPES = ("spec", "test", "systemic")

# Class H: systemic "ensure all <artifact>..." intake with NO artifact id anywhere in the text.
_CLASS_H = ("H:systemic-universal", _re.compile(
    r"^\s*ensure (that )?all (the )?(acceptance criteria|ACs|tests?|specs?|requirements)\b", _re.I))
CLASS_H_ENABLED = False
_ID_RE = _re.compile(r"\b(?:R|AC|SPEC|T)-\d+\b")

_NOISE_CLASSES = (
    ("A:anti-technology-doctrine", _re.compile(
        r"solution[- ]agnostic|technology[- ]agnostic|"
        r"(do not|don't|not|avoid|without)\s+(\w+\s+){0,2}(imply\w*|prescrib\w*|specif\w*|nam\w+|referenc\w*|mention\w*)"
        r"\s+(\w+\s+){0,3}(specific |particular )?(technolog|implementation|design solution|frameworks?\b)", _re.I)),
    ("E:solution-detail-in-spec-or-test", _re.compile(
        r"(does not|do not|not)\s+specify\b.{0,80}(UI elements|UI interactions|selectors?|solution details?|"
        r"performance metrics|GitHub|HTTP methods?|status codes?|CLI commands?)|"
        r"(remove|avoid)\b.{0,60}\b(solution[- ]specific|solution detail|solution language|implies? (a )?(solution|implementation)|"
        r"prescribes? a design)|"
        r"\bsolution[- ]specific\b|"
        r"\buse javascript to\b|\bjavascript[- ]based\b|\b(implemented|managed|handled) (in|using|with) javascript\b|"
        r"\b(add|change|use|specify)\b.{0,40}\b(DOM )?selectors?\b|\b(dom|ui) selectors?\b", _re.I)),
    ("F:cites-decided-architecture", _re.compile(
        r"(as per|aligning with|consistent with|per) the decided (architecture|storage|programming language|stack)", _re.I)),
)


def noise_class(fix: dict) -> str | None:
    scope = fix.get("scope")
    text = fix.get("fix") or ""
    if scope in _G_SCOPES and _CLASS_G[1].search(text):
        return _CLASS_G[0]
    if scope == "systemic":
        # Class H (systemic-universal intake) is defined above but NOT active: on the 171-ledger it matches
        # 3 systemic-vague findings plus 1 storage-real finding that the founder KEPT, so enabling it
        # fails the 0-false-positive gate. Held for a decision.
        if CLASS_H_ENABLED and _CLASS_H[1].search(text) and not _ID_RE.search(text) \
                and not _NOISE_RESCUE.search(text):
            return _CLASS_H[0]
        return None
    if scope not in ("spec", "test"):
        return None
    if _NOISE_RESCUE.search(text):
        return None
    return next((name for name, pat in _NOISE_CLASSES if pat.search(text)), None)


def filter_review_noise(fixes: list[dict]) -> tuple[list[dict], list[dict]]:
    """Archive (never open) findings in the measured noise classes. Auditable: the caller records them."""
    kept, filtered = [], []
    for fix in fixes:
        cls = noise_class(fix)
        if cls:
            filtered.append({**fix, "filtered": f"filtered: noise-class {cls}", "noise_class": cls})
        else:
            kept.append(fix)
    return kept, filtered
