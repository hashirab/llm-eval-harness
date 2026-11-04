"""Compare a candidate run against a baseline run and flag regressions.

Designed to be used as a CI gate: `evalharness compare baseline.json candidate.json`
exits with status 1 if any model x prompt got meaningfully worse.
"""

from __future__ import annotations

from dataclasses import dataclass

from .report import Summary, summarise


@dataclass
class Thresholds:
    max_recall_drop: float = 0.05          # absolute, e.g. 0.05 = 5 percentage points
    max_hallucination_rise: float = 0.03   # absolute
    max_latency_rise: float = 0.25         # relative, on p95 latency


@dataclass
class Delta:
    model: str
    prompt: str
    recall: float
    hallucination: float
    latency_p95_rel: float
    problems: list[str]


def compare(baseline: dict, candidate: dict, thresholds: Thresholds | None = None) -> list[Delta]:
    t = thresholds or Thresholds()
    base = {s.key: s for s in summarise(baseline)}
    cand = {s.key: s for s in summarise(candidate)}
    deltas = []
    for key in sorted(base.keys() & cand.keys()):
        b, c = base[key], cand[key]
        if b.failed or c.failed:
            problems = ["candidate failed on every request"] if c.failed else []
            deltas.append(Delta(key[0], key[1], 0.0, 0.0, 0.0, problems))
            continue
        d_recall = c.recall_mean - b.recall_mean
        d_halluc = c.hallucination_mean - b.hallucination_mean
        d_latency = (c.latency_p95 - b.latency_p95) / b.latency_p95 if b.latency_p95 else 0.0
        problems = []
        if d_recall < -t.max_recall_drop:
            problems.append(f"recall fell {abs(d_recall):.1%}")
        if d_halluc > t.max_hallucination_rise:
            problems.append(f"hallucination rose {d_halluc:.1%}")
        if d_latency > t.max_latency_rise:
            problems.append(f"p95 latency rose {d_latency:.0%}")
        if c.errors > b.errors:
            problems.append(f"errors rose {b.errors} -> {c.errors}")
        deltas.append(Delta(key[0], key[1], d_recall, d_halluc, d_latency, problems))
    return deltas


def to_markdown(deltas: list[Delta]) -> str:
    lines = [
        "| Model | Prompt | Δ recall | Δ hallucination | Δ p95 latency | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for d in deltas:
        status = "REGRESSION: " + "; ".join(d.problems) if d.problems else "ok"
        lines.append(
            f"| {d.model} | {d.prompt} | {d.recall:+.1%} | {d.hallucination:+.1%} | "
            f"{d.latency_p95_rel:+.0%} | {status} |"
        )
    return "\n".join(lines) + "\n"


def has_regression(deltas: list[Delta]) -> bool:
    return any(d.problems for d in deltas)


__all__ = ["Thresholds", "Delta", "compare", "to_markdown", "has_regression", "Summary"]
