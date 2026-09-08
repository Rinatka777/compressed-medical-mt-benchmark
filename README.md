# Compressed Medical MT Benchmark (en→fi)

> **Status: setup complete, phases not started.** This README is a placeholder.
> Per the plan it is rewritten last (Phase 6) to lead with the results table, the
> quality-vs-precision plot, hardware, deduplication counts, entity-checker
> accuracy, and the findings.

## What this will measure

What quantization (via CTranslate2) does to English→Finnish medical machine
translation quality — with particular attention to safety-critical content:
dosages, drug names, and negation. The deliverable is **a table of measurements**,
not the API.

Base models: `Helsinki-NLP/opus-mt-en-fi` (Marian, ~75M) and
`facebook/nllb-200-distilled-600M`.

## Documentation

| File | Contents |
|------|----------|
| [`docs/project-brief.md`](docs/project-brief.md) | The original spec, verbatim. Spec of record. |
| [`docs/progress.md`](docs/progress.md) | Phase-by-phase status and agreed deviations from the brief. |
| [`docs/stack.md`](docs/stack.md) | Every installed package, what it does, which phase uses it. |
| [`SETUP.md`](SETUP.md) | Environment: Python 3.12, the venv, dependency-pin rationale, Windows CUDA. |

## Phases

| Phase | Output | Scripts |
|-------|--------|---------|
| 0 | Smoke test — 10 hand-written sentences translate | — |
| 1 | Deduplicated EMEA train/dev/test split (test = 2000) | `src/prepare_data.py` |
| 2 | Baseline + quantization sweep: BLEU / chrF++ / COMET with CIs, timing, size | `src/convert.py`, `src/translate.py`, `src/evaluate.py` |
| 3 | Entity checks: numeric fidelity, negation, drug-name presence | `src/entity_checks.py` |
| 4 | Fine-tune both base models on cleaned medical data, re-run 2 + 3 | `src/filter_config.yaml`, `src/finetune.py` |
| 5 | FastAPI `/translate` service, model baked into a Docker image | `service/app.py`, `service/Dockerfile` |
| 6 | This README | — |
| 7 | Upstream `filter_config.yaml` to `Helsinki-NLP/OpusFilter-hub` | — |

## Setup

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
```

Full notes, including the CUDA `torch` swap for the GPU box, in [`SETUP.md`](SETUP.md).
