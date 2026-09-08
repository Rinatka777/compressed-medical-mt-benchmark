# The stack — what's installed, what it does, when you use it

Audience note: assumes you know Python packaging, HTTP APIs, Docker, and the
basics of NN training/inference and tokenization. Explains the MT- and
evaluation-specific pieces.

---

## The big picture

```
                    Phase 1                Phase 2 / 4                 Phase 2 / 3
  HF Hub  ──────►  prepare_data.py  ──►  convert.py  ──►  translate.py  ──►  evaluate.py
 (EMEA,           dedup + split          HF ckpt →        CT2 model →        BLEU / chrF++
  models)         train/dev/test         CT2 int8/…       test set out      / COMET + CIs
                                                                            entity_checks.py
                                              │                             (numeric / neg / drug)
                                              ▼
                                        Phase 5: app.py (FastAPI) loads one CT2 model,
                                        POST /translate, baked into a Docker image
```

Two model families run through the same pipeline:

| Model | Arch | Size | Language handling |
|-------|------|------|-------------------|
| `Helsinki-NLP/opus-mt-en-fi` | Marian (encoder-decoder Transformer, Helsinki's OPUS-MT) | ~75M | dedicated en→fi model, no language tokens |
| `facebook/nllb-200-distilled-600M` | NLLB (multilingual, 200 languages) | 600M | **must** prepend source lang `eng_Latn` and force target `fin_Latn`, or output is garbage |

---

## Packages by role

### Model conversion & fast inference

| Package | Version | What it is | Used in |
|---------|---------|------------|---------|
| **ctranslate2** | 4.8.2 | A C++ inference engine for Transformer MT models. Re-implements the forward pass outside PyTorch with heavy optimization (fused ops, cache, CPU SIMD / CUDA). Its converter turns an HF checkpoint into its own format and **quantizes the weights** at the same time (`float32` → `int8` etc.). This is the thing whose effect on quality the whole project measures. CLI: `ct2-transformers-converter`. | Phase 2, 4, 5 |
| **transformers** | 4.57.6 | HuggingFace model/tokenizer library. Here it is used for: (a) the Phase 0 smoke test via `pipeline("translation", …)`, (b) loading tokenizers (CTranslate2 does tensors, not tokenization — you still tokenize/detokenize with the HF tokenizer), (c) `Seq2SeqTrainer` for Phase 4 fine-tuning. Pinned to 4.x by COMET; 5.x also removed the `"translation"` pipeline. | 0, 2, 4 |
| **sentencepiece** | 0.2.2 | The subword tokenizer both Marian and NLLB use (unigram LM / BPE over raw bytes). `transformers` calls into it. You rarely touch it directly. | 0, 2, 4 |
| **sacremoses** | 0.2.0 | Moses-style rule-based punctuation normalization / (de)tokenization. The Marian tokenizer needs it for pre/post-processing. | 2, 4 |
| **torch** | 2.14.0 | Backend for `transformers` (fine-tuning, smoke test) and for **COMET**. CTranslate2 does **not** use it at inference. CPU wheel on Mac; install the `cu126` wheel on the RTX 4070 (see `SETUP.md`). | 0, 2 (COMET), 4 |
| **accelerate** | 1.14.0 | HuggingFace's device/precision/distributed launcher. `Seq2SeqTrainer` imports it for mixed-precision (fp16) and device placement. | 4 |

### Data

| Package | Version | What it is | Used in |
|---------|---------|------------|---------|
| **datasets** | 5.0.1 | HuggingFace dataset loader (Arrow-backed, memory-mapped, streaming). You load `Helsinki-NLP/emea` with it. EMEA = the European Medicines Agency corpus: sentence-aligned text extracted from **drug package leaflets and regulatory documents**, part of the OPUS collection. Highly repetitive (same boilerplate across hundreds of leaflets) — hence the mandatory dedup. | 1 |

### Evaluation

| Package | Version | What it is | Used in |
|---------|---------|------------|---------|
| **sacrebleu** | 2.6.0 | Standard, reproducible implementation of **BLEU** (n-gram precision vs. reference, brevity penalty) and **chrF++** (character n-gram F-score + word n-grams; better for morphologically rich targets like Finnish). Crucially it also does **paired bootstrap resampling**: resample the test set with replacement N times, recompute the metric for each system on each resample, and derive a confidence interval and a significance test for the *difference* between two systems. This is why the brief forbids bare score deltas. | 2, 3, 4 |
| **unbabel-comet** | 2.2.7 | **COMET** — a *learned* metric. A pretrained encoder (XLM-R large, ~560M) fine-tuned on human translation-quality judgments; it scores (source, hypothesis, reference) triples and correlates with human ratings far better than BLEU. You download a checkpoint (`Unbabel/wmt22-comet-da`, ~2.3 GB) once, then call `model.predict()`. It is itself a neural net running on torch — GPU strongly preferred. This package is the dependency ceiling: it pins `numpy<2`, `transformers` 4.x, `torchmetrics` 0.10.3. | 2, 4 |

### Phase 4 — training-data cleaning

| Package | Version | What it is | Used in |
|---------|---------|------------|---------|
| **opusfilter** | 3.3.1 | A YAML-configured pipeline for cleaning parallel corpora. You declare a sequence of filters (length, length-ratio, language ID, character scripts, alignment scores, …) in `src/filter_config.yaml`; it reads the bitext and drops pairs that fail. Used here to strip PDF-extraction damage (headers, broken tables, misaligned or wrong-language pairs) from the **training split only**. CLI is a plain script — run it as `.venv/bin/python .venv/bin/opusfilter filter_config.yaml`. Phase 7 contributes this config upstream. | 4, 7 |
| py3langid, lingua-language-detector, morfessor, sentence-splitter, subword-nmt, opustools, … | (deps) | OpusFilter's backends: language identification, morphology, sentence segmentation, subword models, OPUS download tools. Pulled in automatically. `py3langid` is pinned to 0.2.2 because newer versions demand `numpy>=2` and COMET forbids that. | 4 |

### Phase 5 — service

| Package | Version | What it is | Used in |
|---------|---------|------------|---------|
| **fastapi** | 0.141.1 | The web framework. One endpoint: `POST /translate` `{"text": …}` → `{"translation": …}`. Use the **lifespan** context manager to load the CT2 model once at startup, not per request. | 5 |
| **uvicorn[standard]** | 0.52.4 | ASGI server that runs the FastAPI app. `[standard]` adds `uvloop`, `httptools`, `watchfiles`, `websockets` (faster loop + dev reload). | 5 |
| **starlette**, **pydantic** | (deps) | FastAPI internals: Starlette is the ASGI toolkit under FastAPI; Pydantic validates/serializes the request and response models. | 5 |

### Plotting, tables, misc

| Package | Version | Role |
|---------|---------|------|
| **matplotlib** | 3.11.1 | The Phase 2 quality-vs-precision plot. |
| **pandas** | 2.2.3 | Assembling the results tables from the per-run JSON files. Pinned to 2.2.x — pandas 3.x needs `numpy>=2`. |
| **numpy** | 1.26.4 | Everywhere. Pinned `<2` by COMET; this cascades to pandas and py3langid. |
| **pyyaml** | 6.0.3 | Reading `filter_config.yaml` and writing config into results files. |
| **psutil** | 7.2.2 | Hardware fingerprint for results files — RAM, physical core count, per the "every latency number names its hardware" rule. |
| **setuptools** | 80.10.2 | Pinned `<81`: `torchmetrics` 0.10.3 (via COMET) imports `pkg_resources`, removed in setuptools 81. |

---

## Which packages matter in which phase

| Phase | Primary packages |
|-------|------------------|
| 0 Smoke | transformers, torch, sentencepiece, sacremoses |
| 1 Data | datasets |
| 2 Convert + sweep | ctranslate2, transformers (tokenizers), sacrebleu, unbabel-comet, torch, matplotlib, pandas, psutil |
| 3 Entity checks | sacrebleu (bootstrap), plain Python regex/stdlib; pandas |
| 4 Fine-tune | opusfilter (+ backends), transformers (`Seq2SeqTrainer`), accelerate, torch, datasets — then re-run Phase 2/3 stack |
| 5 Service | fastapi, uvicorn, ctranslate2, transformers (tokenizer) |
| 6 README | matplotlib, pandas |
| 7 Upstream | opusfilter (config only) |

---

## Mental model of one "condition"

A **condition** = (base model) × (adaptation: base or fine-tuned) × (precision: float32 / int16 / int8 / float16 / bfloat16).
For each condition you produce one translation file and one results JSON holding:
BLEU, chrF++, COMET (each with a bootstrap CI), wall-clock translation time (+ hardware),
model size on disk, peak memory, and — from Phase 3 — numeric / negation / drug-name
error rates (each with a CI). The deliverable is the table over all conditions.
