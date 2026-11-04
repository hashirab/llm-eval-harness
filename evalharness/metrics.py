"""Scoring functions.

All metrics are deterministic and explainable: every score comes with the evidence behind it
(which facts were found, which entities were unsupported), so a result can be checked by hand.

- key_fact_recall: share of the reference key facts that appear in the model output.
- hallucination: share of checkable entities in the output (IPs, CVEs, hashes, domains,
  emails, times, numbers) that do NOT appear anywhere in the source text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "from", "by", "and", "or", "with",
    "was", "were", "is", "are", "be", "been", "via", "as", "that", "this", "it",
}

# Order matters: more specific patterns first, so their spans are removed before
# the generic number pattern runs.
ENTITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("cve", re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)),
    ("hash", re.compile(r"\b(?:[a-f0-9]{64}|[a-f0-9]{40}|[a-f0-9]{32})\b", re.IGNORECASE)),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("domain", re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,24}\b", re.IGNORECASE)),
    ("time", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")),
    ("number", re.compile(r"\b\d{2,}(?:,\d{3})*\b")),
]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _content_tokens(text: str) -> list[str]:
    tokens = (t.rstrip(".:-/@") for t in re.findall(r"[a-z0-9][a-z0-9.\-:@/]*", normalise(text)))
    return [t for t in tokens if t and t not in _STOPWORDS]


@dataclass
class FactResult:
    fact: str
    found: bool
    coverage: float


@dataclass
class RecallResult:
    score: float
    facts: list[FactResult] = field(default_factory=list)


def fact_found(fact: str, output: str, min_coverage: float = 0.8) -> FactResult:
    """A fact counts as found if it appears verbatim, or if at least `min_coverage`
    of its content tokens appear in the output (handles light rephrasing)."""
    out_norm = normalise(output)
    if normalise(fact) in out_norm:
        return FactResult(fact, True, 1.0)
    tokens = _content_tokens(fact)
    if not tokens:
        return FactResult(fact, False, 0.0)
    out_tokens = set(_content_tokens(output))
    coverage = sum(t in out_tokens for t in tokens) / len(tokens)
    return FactResult(fact, coverage >= min_coverage, round(coverage, 3))


def key_fact_recall(output: str, key_facts: list[str], min_coverage: float = 0.8) -> RecallResult:
    if not key_facts:
        return RecallResult(score=1.0)
    results = [fact_found(f, output, min_coverage) for f in key_facts]
    return RecallResult(score=sum(r.found for r in results) / len(results), facts=results)


@dataclass
class Entity:
    kind: str
    value: str


def extract_entities(text: str) -> list[Entity]:
    """Pull out checkable, factual tokens. Matched spans are blanked out so an IP address
    is not also counted as four numbers."""
    remaining = text
    found: list[Entity] = []
    for kind, pattern in ENTITY_PATTERNS:
        for match in pattern.finditer(remaining):
            value = match.group(0)
            if kind == "domain" and value.replace(".", "").isdigit():
                continue
            found.append(Entity(kind, value))
        remaining = pattern.sub(lambda m: " " * len(m.group(0)), remaining)
    return found


def _canonical(value: str) -> str:
    return value.lower().replace(",", "")


@dataclass
class HallucinationResult:
    rate: float
    total_entities: int
    unsupported: list[Entity] = field(default_factory=list)


def hallucination(output: str, source: str) -> HallucinationResult:
    """Share of entities in the output that cannot be found in the source.

    This is a precision-style check on hard facts. It will not catch a wrong claim made
    in plain words, but it reliably catches invented IPs, hosts, CVEs, counts and times,
    which are the errors that matter most in incident reports.
    """
    entities = extract_entities(output)
    if not entities:
        return HallucinationResult(rate=0.0, total_entities=0)
    source_canon = _canonical(source)
    unsupported = [e for e in entities if _canonical(e.value) not in source_canon]
    return HallucinationResult(
        rate=len(unsupported) / len(entities),
        total_entities=len(entities),
        unsupported=unsupported,
    )
