"""Aggregate a run into one row per model x prompt, and render it as Markdown."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Summary:
    model: str
    prompt: str
    n: int
    errors: int
    recall_mean: float
    recall_std: float
    hallucination_mean: float
    instability: float
    latency_p50: float
    latency_p95: float

    @property
    def key(self) -> tuple[str, str]:
        return (self.model, self.prompt)

    @property
    def failed(self) -> bool:
        """True when every request errored, so the scores are not meaningful."""
        return self.n > 0 and self.errors == self.n


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _instability(rows: list[dict]) -> float:
    """Average, over test cases, of the spread in recall across repeats of that case.
    0 means the model gives equally good answers every time; higher means it varies."""
    by_case: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r["recall"])
    spreads = [statistics.pstdev(v) for v in by_case.values() if len(v) > 1]
    return statistics.mean(spreads) if spreads else 0.0


def summarise(run_data: dict) -> list[Summary]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in run_data["results"]:
        groups[(r["model"], r["prompt"])].append(r)
    summaries = []
    for (model, prompt), rows in groups.items():
        ok = [r for r in rows if r["error"] is None]
        recalls = [r["recall"] for r in ok]
        latencies = [r["latency_ms"] for r in ok]
        summaries.append(Summary(
            model=model,
            prompt=prompt,
            n=len(rows),
            errors=len(rows) - len(ok),
            recall_mean=statistics.mean(recalls) if recalls else 0.0,
            recall_std=statistics.pstdev(recalls) if len(recalls) > 1 else 0.0,
            hallucination_mean=statistics.mean(r["hallucination_rate"] for r in ok) if ok else 0.0,
            instability=_instability(ok),
            latency_p50=_percentile(latencies, 0.5),
            latency_p95=_percentile(latencies, 0.95),
        ))
    return sorted(summaries, key=lambda s: (s.failed, -s.recall_mean, s.hallucination_mean))


def to_markdown(run_data: dict) -> str:
    rows = summarise(run_data)
    lines = [
        f"## {run_data['name']} ({run_data['created']})",
        "",
        f"{run_data['cases']} cases x {run_data['repeats']} repeats per model and prompt.",
        "",
        "| Model | Prompt | Key-fact recall | Hallucination rate | Instability | p50 latency (ms) | p95 latency (ms) | Errors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in rows:
        if s.failed:
            lines.append(f"| {s.model} | {s.prompt} | — | — | — | — | — | {s.errors}/{s.n} (all failed) |")
            continue
        lines.append(
            f"| {s.model} | {s.prompt} | {s.recall_mean:.1%} ± {s.recall_std:.1%} | "
            f"{s.hallucination_mean:.1%} | {s.instability:.3f} | {s.latency_p50:,.0f} | "
            f"{s.latency_p95:,.0f} | {s.errors}/{s.n} |"
        )
    worst = _worst_hallucinations(run_data)
    if worst:
        lines += ["", "### Unsupported entities (most frequent)", ""]
        lines += [f"- `{value}` ({count}x)" for value, count in worst]
    failed = [s for s in rows if s.failed]
    if failed:
        first_error = next(r["error"] for r in run_data["results"] if r["error"])
        lines += ["", f"**{len(failed)} model/prompt pair(s) failed on every request.** First error: `{first_error[:200]}`"]
    return "\n".join(lines) + "\n"


def _worst_hallucinations(run_data: dict, top: int = 10) -> list[tuple[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for r in run_data["results"]:
        for value in r.get("unsupported", []):
            counts[value] += 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:top]
