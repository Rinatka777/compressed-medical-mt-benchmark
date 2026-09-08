# Setup

This file is for getting the environment running. Project findings live in `README.md` (written in Phase 6).

## Machines

| Role | Machine | Used for |
|------|---------|----------|
| Development | Apple M2, 8 GB RAM, macOS (no CUDA) | writing code; Phases 0, 1, 3; 100-sentence smoke subsets |
| Benchmark / training | i7-13700F, 32 GB RAM, RTX 4070 (12 GB), Windows | Phases 2, 4 at full scale; **all latency numbers** |

Every latency figure written to `results/` must record the machine it ran on (`src` helpers capture this automatically via `psutil` + `torch`).

## Python

**Python 3.12.** Not 3.11 (unavailable on the dev machine), not 3.13:

- `unbabel-comet` 2.2.7 pins `torchmetrics` 0.10.3, which imports `pkg_resources`; on 3.13 with current setuptools this needs the `setuptools<81` pin (also applied on 3.12).
- COMET's `comet-score` CLI crashes on 3.13 (`argparse` internal signature change). The Python API is used instead, but 3.12 keeps the CLI working too.

## Install (both machines)

```
python3.12 -m venv .venv
# macOS/Linux:
.venv/bin/pip install -r requirements.txt
# Windows:
.venv\Scripts\pip install -r requirements.txt
```

### Windows: CUDA build of torch

`requirements.txt` pins the CPU `torch==2.14.0`. On the RTX 4070 box, after
`pip install -r requirements.txt`, replace it with the CUDA build (same version;
`torch 2.14.0` ships a **cu126** cp312 wheel — there is no cu124 one):

```
.venv\Scripts\pip install --force-reinstall torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
```

Then confirm `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
prints `True 12.6`. Record the CUDA version in the results files.

## Dependency constraints (do not relax without re-checking)

`unbabel-comet` is the pin that constrains everything else:

| Package | Pinned | Forced by |
|---------|--------|-----------|
| `numpy` | `<2` (1.26.4) | `unbabel-comet` requires `numpy<2.0.0` |
| `transformers` | 4.x | `unbabel-comet`; also keeps the `"translation"` pipeline that 5.x removed |
| `torchmetrics` | 0.10.3 | `unbabel-comet` |
| `pandas` | `2.2.3` | pandas 3.x wants `numpy>=2` |
| `py3langid` | `0.2.2` | `opusfilter` dep; 0.3.0+ requires `numpy>=2` on Python >= 3.9 |
| `setuptools` | `<81` | `torchmetrics` 0.10.3 imports the removed `pkg_resources` |

If COMET is ever dropped, most of these can be relaxed.

## Regenerating the lock

```
pip install -r requirements.in
pip freeze > requirements.txt
```

## Folder name

The folder was renamed from `Compressed Medical MT Benchmark (en→fi)` to
`compressed-medical-mt-benchmark` on 2026-09-08 and the venv recreated. The old
name (spaces + a non-ASCII arrow) broke the `opusfilter` script's shebang and
would have broken the Phase 5 Docker build context and Windows consoles.

If you ever hit a console script whose shebang path is wrong, run it through the
interpreter explicitly: `.venv/bin/python .venv/bin/opusfilter filter_config.yaml`.

## Verified toolchain (dev machine, 2026-09-08)

`python3.12`, all of `pip check` clean. Confirmed working end to end:

- `transformers` `pipeline("translation", "Helsinki-NLP/opus-mt-en-fi")` → correct Finnish
- `ct2-transformers-converter --quantization int8` → 77 MB CT2 model
- `ctranslate2.Translator(..., compute_type="int8")` inference → correct Finnish, "5 mg" preserved
- `opusfilter` `LangidFilter` scores en/fi text
- `sacrebleu` `BLEU` + `CHRF(word_order=2)` (chrF++), `comet-score` CLI

On this Mac the HF pipeline auto-selects `mps`. On the RTX 4070 box it will pick `cuda`
once the CUDA `torch` wheel is installed. CTranslate2 chooses CPU vs CUDA via its
`device=` argument.

### Handy converters

- Marian: `ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-fi --output_dir models/ct2/... --quantization <q>`
- NLLB:   same tool, `--model facebook/nllb-200-distilled-600M`; remember `eng_Latn` / `fin_Latn` at inference
- `ct2-opus-mt-converter` also exists (pulls straight from the OPUS-MT store) if the transformers route misbehaves
