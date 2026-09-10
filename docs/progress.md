# Progress vs. the brief

Tracks where we are against `docs/project-brief.md`. Update the status column as
each "Done when" condition is met; note deviations from the brief here with a reason.

| Phase | Brief "done when" | Status | Notes |
|------|-------------------|--------|-------|
| 0 — Smoke test | 10 hand-written sentences translate and look like Finnish | ✅ done | `src/phase0_smoke.py` → `results/phase0_smoke.json`. 10 sentences, 3.16 s on M2 `mps`, HF pipeline. Fluent Finnish; "5 mg" preserved, all four negations kept (`älä`, `ei tule`, `ei saa`), drug names present & inflected (amlodipiinia, ibuprofeeni-, penisilliinille). |
| 1 — Data | 3 split files exist; duplicate-removal count recorded | ◻ not started | `src/prepare_data.py`. Source: **OPUS EMEA v3 en–fi moses** (HF `Helsinki-NLP/emea` is dead — see deviations). Dedup **before** split. Test = 2000. |
| 2 — Baseline + sweep | Results table (both models × every precision) + plot | ◻ not started | `convert.py` / `translate.py` / `evaluate.py`. |
| 3 — Entity checks | Every condition has entity error rates + CIs; checker accuracy documented | ◻ not started | `src/entity_checks.py`. |
| 4 — Fine-tuning | Full table: 2 models × 2 states × every precision | ◻ not started | `filter_config.yaml` + `finetune.py`. GPU (Windows) only. |
| 5 — Service | `docker run` translates with networking disabled | ◻ not started | `service/app.py` + `Dockerfile`. |
| 6 — README | Report with findings first | ◻ not started | Fill `README.md`. |
| 7 — Upstream PR | `filter_config.yaml` PR to `Helsinki-NLP/OpusFilter-hub` | ◻ not started | Optional / non-blocking. |

## Deviations from the brief (agreed)

| Brief says | We do | Why |
|------------|-------|-----|
| Python 3.11 | Python **3.12** | 3.11 unavailable on the dev Mac; 3.13 breaks the `comet-score` CLI. 3.12 is the closest working version. |
| `requirements.txt` = the 9 listed packages | `requirements.in` (top-level) + `requirements.txt` (full frozen lock, 107 pkgs) | Same intent, plus a reproducible lock. `opusfilter`, `accelerate`, `sacremoses`, plotting libs added — all implied by later phases. |
| GPU-only precisions "if a GPU is available" | Run `float16`/`bfloat16` on the Windows RTX 4070 only | Dev Mac has no CUDA. |
| "one consumer GPU" for fine-tuning | RTX 4070, 12 GB. Marian: trivial. NLLB-600M: needs fp16 + grad checkpointing + small batch + grad accumulation. | 12 GB is enough but not generous for 600M. |
| Repo root `medmt/` | Repo root is the project folder directly (`src/`, `data/`, …) | Cosmetic; the `medmt/` in the brief is just the project name. |
| Data from `Helsinki-NLP/emea` on HF | Download EMEA v3 en–fi moses files straight from OPUS (`https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/en-fi.txt.zip`, ~34 MB) | The HF dataset is a loading-script dataset (`emea.py`); `datasets` 5.0.1 removed script support entirely and `trust_remote_code` now raises. OPUS is the original source anyway. |

## Open questions / decisions still to make

- Decoding parameters (beam size, length penalty) — must be fixed and recorded; brief doesn't specify. Proposed: beam 5, default length penalty, greedy fallback documented.
- Entity metrics: report on the **full** test set and on the **subset that actually contains** a number / negation / drug name (the informative denominator).
- Paired comparison across precisions: use McNemar (same sentences) for the entity-error deltas, not just overlapping CIs.
- COMET model choice: `Unbabel/wmt22-comet-da` (reference-based) as primary; consider `wmt22-cometkiwi-da` (reference-free) as a cross-check.
- Manual validation set for the negation/drug checkers: brief says 100; 200–300 gives a usable CI on the checker's own accuracy.
