# llm-eval-harness

A small, dependency-light tool for answering a practical question: **which model and prompt should we actually ship?**

It runs every model × prompt × test case several times, scores each answer on **key-fact recall**, **hallucination rate** and **latency**, and saves the raw results. You can then compare any two runs and fail a CI build if quality regresses.

The bundled example evaluates LLMs on summarising security incidents, where an invented IP address or hostname is worse than a missing one.

## Why it exists

During an AI studentship at BT I benchmarked local LLMs against Google Gemini for summarising cybersecurity incident data. The main lesson was that LLM quality has to be measured, not assumed: elaborate prompt-optimisation methods gave no measurable gain over well-designed baseline prompts, and that only became clear by scoring every output the same way.

This repository is a clean, open rebuild of that approach. **It contains no BT code or data.** The incidents in `datasets/` are synthetic and use reserved documentation IP ranges (RFC 5737) and `example.*` domains.

The code was written with Claude as a coding agent; the evaluation approach comes from my BT study, and I reviewed and tested it.

## What it measures

| Metric | How it is computed | What it tells you |
| --- | --- | --- |
| **Key-fact recall** | Each test case lists the facts a good answer must contain. A fact counts as found if it appears verbatim, or if at least 80% of its content words appear. | Did the model include what matters? |
| **Hallucination rate** | Checkable entities are extracted from the output (IPs, CVEs, hashes, domains, emails, times, numbers). The rate is the share that **do not appear anywhere in the source text**. | Did the model invent hard facts? |
| **Instability** | For each case, the spread in recall across repeated runs, averaged over cases. | Is the model reliable, or does quality vary from run to run? |
| **Latency** | p50 and p95 wall-clock time per request. | Is it fast enough to use? |

Every score keeps its evidence: the run file records which facts were missed and which entities were unsupported, so any number can be checked by hand.

The metrics are deliberately deterministic. An LLM-as-judge could catch wrong claims made in plain words, which the entity check cannot, but it adds its own non-determinism and cost. It is the natural next extension (see Roadmap).

## Quick start

```bash
git clone https://github.com/hashirab/llm-eval-harness.git
cd llm-eval-harness
pip install -e ".[dev]"

# Offline dry run, no models or keys needed
evalharness run configs/mock.yaml

# Real run: a local model via Ollama and a hosted model via Gemini
ollama pull llama3.1:8b
export GEMINI_API_KEY=...
evalharness run configs/example.yaml
```

Each run prints a summary table and saves the raw results to `runs/`:

```
| Model | Prompt | Key-fact recall | Hallucination rate | Instability | p50 latency (ms) | p95 latency (ms) | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mock:echo | baseline | 65.1% ± 17.5% | 0.0% | 0.000 | 0 | 0 | 0/16 |
```

*(Output above is from the offline mock run, which simply echoes part of the incident text. Run `configs/example.yaml` against real models to get meaningful numbers.)*

## Catching regressions

Save a run you are happy with as a baseline, then compare every change against it:

```bash
evalharness compare runs/baseline.json runs/candidate.json
```

```
| Model | Prompt | Δ recall | Δ hallucination | Δ p95 latency | Status |
| --- | --- | --- | --- | --- | --- |
| ollama:llama3.1:8b | structured | -8.3% | +2.1% | +4% | REGRESSION: recall fell 8.3% |
```

*(Illustrative output showing the format.)*

The command exits with status `1` on any regression, so it can gate a pull request. Thresholds are configurable:

```bash
evalharness compare base.json cand.json --max-recall-drop 0.03 --max-hallucination-rise 0.02 --max-latency-rise 0.2
```

## Using your own data

A test set is a JSONL file with one case per line:

```json
{"id": "inc-001", "source": "the full input text...", "key_facts": ["203.0.113.45", "web-prod-03", "account disabled"]}
```

A prompt is a text file containing `{incident}` where the source text should go. A config ties them together:

```yaml
name: my-eval
dataset: ../datasets/my_cases.jsonl
repeats: 3
output_dir: ../runs
prompts:
  baseline: ../prompts/baseline.txt
models:
  - provider: ollama        # ollama | gemini | openai | mock
    model: llama3.1:8b
    temperature: 0.7
```

The `openai` provider works with any OpenAI-compatible endpoint (set `OPENAI_BASE_URL`), which covers most hosted and self-hosted inference servers.

## Project layout

```
evalharness/
  providers.py   Ollama, Gemini, OpenAI-compatible and mock providers (stdlib HTTP only)
  metrics.py     key-fact recall, entity extraction, hallucination check
  runner.py      runs model x prompt x case x repeat, saves raw results
  report.py      aggregates a run into a Markdown table
  compare.py     run-vs-run deltas and the regression gate
  cli.py         `evalharness run | report | compare`
datasets/        synthetic security-incident test set
prompts/         baseline and structured prompt templates
configs/         example (real models) and mock (offline) configs
tests/           unit and end-to-end tests, run in CI on every push
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

CI runs the test suite on Python 3.10 and 3.12, plus an offline smoke run of the CLI.

## Roadmap

- LLM-as-judge scoring for claims that entity matching cannot verify, calibrated against hand labels
- Structured-output (JSON) extraction tasks with field-level precision and recall
- Cost tracking per run for hosted models
- Concurrency for large test sets

## Licence

MIT
