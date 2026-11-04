"""Command-line interface.

    evalharness run configs/example.yaml
    evalharness report runs/<run>.json
    evalharness compare runs/baseline.json runs/candidate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import compare as compare_mod
from . import report
from .runner import load_config, run, save_run


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.repeats:
        config["repeats"] = args.repeats
    if args.models:
        wanted = [m.strip().lower() for m in args.models.split(",") if m.strip()]
        config["models"] = [
            spec for spec in config["models"]
            if any(w in f"{spec['provider']}:{spec.get('model', '')}".lower() for w in wanted)
        ]
        if not config["models"]:
            print(f"No models in {args.config} match --models {args.models}")
            return 2
    if args.model_override:
        for spec in config["models"]:
            spec["model"] = args.model_override
    data = run(config, limit=args.limit)
    path = save_run(data, config["output_dir"])
    print(report.to_markdown(data))
    print(f"Saved raw results to {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    md = report.to_markdown(_load(args.run))
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(md)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    thresholds = compare_mod.Thresholds(
        max_recall_drop=args.max_recall_drop,
        max_hallucination_rise=args.max_hallucination_rise,
        max_latency_rise=args.max_latency_rise,
    )
    deltas = compare_mod.compare(_load(args.baseline), _load(args.candidate), thresholds)
    if not deltas:
        print("No model/prompt pairs in common between the two runs.")
        return 2
    print(compare_mod.to_markdown(deltas))
    if compare_mod.has_regression(deltas):
        print("Regression detected.")
        return 1
    print("No regressions.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalharness", description="Evaluate LLMs on accuracy, hallucination and latency.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an evaluation from a YAML config")
    p_run.add_argument("config")
    p_run.add_argument("--limit", type=int, help="only use the first N test cases")
    p_run.add_argument("--repeats", type=int, help="override the number of repeats per case")
    p_run.add_argument("--models", help="only run models whose 'provider:model' contains one of these comma-separated words, e.g. gemini")
    p_run.add_argument("--model-override", help="use this model name for every selected model, e.g. gemini-3.8-flash-lite")
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="summarise a saved run as Markdown")
    p_report.add_argument("run")
    p_report.add_argument("--out", help="write the report to a file")
    p_report.set_defaults(func=cmd_report)

    p_cmp = sub.add_parser("compare", help="compare two runs and fail on regression")
    p_cmp.add_argument("baseline")
    p_cmp.add_argument("candidate")
    p_cmp.add_argument("--max-recall-drop", type=float, default=0.05)
    p_cmp.add_argument("--max-hallucination-rise", type=float, default=0.03)
    p_cmp.add_argument("--max-latency-rise", type=float, default=0.25)
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
