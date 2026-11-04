import json
from pathlib import Path

import pytest

from evalharness import cli
from evalharness.compare import Thresholds, compare, has_regression
from evalharness.providers import MockProvider, ProviderError, build_provider
from evalharness.report import summarise, to_markdown
from evalharness.runner import load_config, load_dataset, run

ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIG = ROOT / "configs" / "mock.yaml"


def test_dataset_loads_and_is_valid():
    cases = load_dataset(ROOT / "datasets" / "security_incidents.jsonl")
    assert len(cases) >= 5
    assert all(c.key_facts for c in cases)


def test_key_facts_are_supported_by_their_source():
    """Every key fact's entities must exist in its own source, or the dataset is wrong."""
    from evalharness.metrics import hallucination

    for case in load_dataset(ROOT / "datasets" / "security_incidents.jsonl"):
        for fact in case.key_facts:
            assert hallucination(fact, case.source).rate == 0.0, (case.id, fact)


def test_config_resolves_paths_and_checks_placeholder(tmp_path):
    config = load_config(MOCK_CONFIG)
    assert Path(config["dataset"]).exists()
    assert "{incident}" in config["prompts"]["baseline"]

    (tmp_path / "bad.txt").write_text("no placeholder here")
    (tmp_path / "data.jsonl").write_text('{"id": "x", "source": "s", "key_facts": []}\n')
    (tmp_path / "c.yaml").write_text(
        "dataset: data.jsonl\nprompts:\n  p: bad.txt\nmodels:\n  - provider: mock\n"
    )
    with pytest.raises(ValueError, match="placeholder"):
        load_config(tmp_path / "c.yaml")


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        build_provider({"provider": "nope", "model": "x"})


def test_run_with_perfect_and_hallucinating_models():
    config = load_config(MOCK_CONFIG)
    cases = {c.id: c for c in load_dataset(config["dataset"])}

    def perfect(prompt):
        case = next(c for c in cases.values() if c.source in prompt)
        return " ".join(case.key_facts)

    def liar(prompt):
        return "Traffic from 10.66.66.66 to totally-fake.example.io, 99,999 files affected."

    providers = [MockProvider("perfect", responder=perfect), MockProvider("liar", responder=liar)]
    data = run(config, providers=providers, progress=False)
    rows = {s.model: s for s in summarise(data)}

    assert rows["mock:perfect"].recall_mean == 1.0
    assert rows["mock:perfect"].hallucination_mean == 0.0
    assert rows["mock:liar"].recall_mean < 0.2
    assert rows["mock:liar"].hallucination_mean == 1.0
    assert "mock:liar" in to_markdown(data)


def test_provider_errors_are_recorded_not_raised():
    def broken(prompt):
        raise ProviderError("model offline")

    config = load_config(MOCK_CONFIG)
    data = run(config, providers=[MockProvider("broken", responder=broken)], limit=2, progress=False)
    assert all(r["error"] == "model offline" for r in data["results"])
    assert summarise(data)[0].errors == len(data["results"])


def test_compare_flags_regression_and_passes_identical_runs():
    config = load_config(MOCK_CONFIG)
    cases = load_dataset(config["dataset"])

    def good(prompt):
        case = next(c for c in cases if c.source in prompt)
        return " ".join(case.key_facts)

    def worse(prompt):
        case = next(c for c in cases if c.source in prompt)
        return case.key_facts[0]

    base = run(config, providers=[MockProvider("m", responder=good)], progress=False)
    cand = run(config, providers=[MockProvider("m", responder=worse)], progress=False)

    assert not has_regression(compare(base, base))
    deltas = compare(base, cand, Thresholds())
    assert has_regression(deltas)
    assert any("recall fell" in p for p in deltas[0].problems)


def test_cli_run_report_and_compare(tmp_path, monkeypatch, capsys):
    config_text = MOCK_CONFIG.read_text().replace("../runs", str(tmp_path / "runs"))
    config_text = config_text.replace("../datasets", str(ROOT / "datasets"))
    config_text = config_text.replace("../prompts", str(ROOT / "prompts"))
    cfg = tmp_path / "mock.yaml"
    cfg.write_text(config_text)

    assert cli.main(["run", str(cfg), "--limit", "3"]) == 0
    run_file = next((tmp_path / "runs").glob("*.json"))
    assert json.loads(run_file.read_text())["cases"] == 3

    report_file = tmp_path / "report.md"
    assert cli.main(["report", str(run_file), "--out", str(report_file)]) == 0
    assert "Key-fact recall" in report_file.read_text()

    assert cli.main(["compare", str(run_file), str(run_file)]) == 0
    assert "No regressions" in capsys.readouterr().out
