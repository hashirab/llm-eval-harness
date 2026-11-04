"""Run every model x prompt x test case, repeated N times, and save the raw results."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .metrics import hallucination, key_fact_recall
from .providers import Provider, ProviderError, build_provider


@dataclass
class Case:
    id: str
    source: str
    key_facts: list[str]


@dataclass
class Result:
    case_id: str
    model: str
    prompt: str
    repeat: int
    output: str = ""
    latency_ms: float | None = None
    recall: float | None = None
    missed_facts: list[str] = field(default_factory=list)
    hallucination_rate: float | None = None
    entities: int = 0
    unsupported: list[str] = field(default_factory=list)
    error: str | None = None


def load_dataset(path: str | Path) -> list[Case]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"id", "source", "key_facts"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no} is missing {sorted(missing)}")
            cases.append(Case(row["id"], row["source"], list(row["key_facts"])))
    if not cases:
        raise ValueError(f"No test cases found in {path}")
    return cases


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base = path.parent
    config["dataset"] = str((base / config["dataset"]).resolve())
    config["prompts"] = {name: (base / p).read_text(encoding="utf-8") for name, p in config["prompts"].items()}
    for name, template in config["prompts"].items():
        if "{incident}" not in template:
            raise ValueError(f"Prompt '{name}' must contain the placeholder {{incident}}")
    config.setdefault("repeats", 1)
    config.setdefault("output_dir", "runs")
    config["output_dir"] = str((base / config["output_dir"]).resolve())
    return config


def evaluate_one(provider: Provider, prompt_name: str, template: str, case: Case, repeat: int) -> Result:
    result = Result(case_id=case.id, model=provider.name, prompt=prompt_name, repeat=repeat)
    try:
        completion = provider.complete(template.replace("{incident}", case.source))
    except ProviderError as exc:
        result.error = str(exc)
        return result
    recall = key_fact_recall(completion.text, case.key_facts)
    halluc = hallucination(completion.text, case.source)
    result.output = completion.text
    result.latency_ms = round(completion.latency_ms, 1)
    result.recall = round(recall.score, 4)
    result.missed_facts = [f.fact for f in recall.facts if not f.found]
    result.hallucination_rate = round(halluc.rate, 4)
    result.entities = halluc.total_entities
    result.unsupported = [f"{e.kind}:{e.value}" for e in halluc.unsupported]
    return result


def run(config: dict, providers: list[Provider] | None = None, limit: int | None = None,
        progress: bool = True) -> dict:
    cases = load_dataset(config["dataset"])[:limit]
    providers = providers or [build_provider(spec) for spec in config["models"]]
    results: list[Result] = []
    total = len(providers) * len(config["prompts"]) * len(cases) * config["repeats"]
    done = 0
    started = time.perf_counter()
    for provider in providers:
        for prompt_name, template in config["prompts"].items():
            for case in cases:
                for repeat in range(config["repeats"]):
                    results.append(evaluate_one(provider, prompt_name, template, case, repeat))
                    done += 1
                    if progress:
                        print(f"\r[{done}/{total}] {provider.name} | {prompt_name} | {case.id}", end="", flush=True)
    if progress:
        print()
    return {
        "name": config.get("name", "eval"),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - started, 1),
        "repeats": config["repeats"],
        "cases": len(cases),
        "results": [asdict(r) for r in results],
    }


def save_run(run_data: dict, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = run_data["created"].replace(":", "").replace("-", "").replace("+0000", "Z")
    path = output_dir / f"{run_data['name']}-{stamp}.json"
    path.write_text(json.dumps(run_data, indent=2), encoding="utf-8")
    return path
