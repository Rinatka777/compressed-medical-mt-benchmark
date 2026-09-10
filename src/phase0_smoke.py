"""Phase 0 smoke test.

The smallest end-to-end run that proves the toolchain is wired together:
load model -> tokenize -> translate -> detokenize -> print -> save.
Not measurement infrastructure; just a "does smoke come out" check.

Usage:
    .venv/bin/python src/phase0_smoke.py

Done when: the ten sentences translate and look like Finnish, and
results/phase0_smoke.json exists.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

# Use the model already downloaded into the repo (models/hf/.hfcache), not ~/.cache.
REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "models" / "hf" / ".hfcache"))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import pipeline  # noqa: E402

MODEL = "Helsinki-NLP/opus-mt-en-fi"
SEED = 0
OUT_PATH = REPO_ROOT / "results" / "phase0_smoke.json"

# Hand-written. A few deliberately exercise the safety-critical cases the whole
# project is about: dosages/units, negation, drug names.
SENTENCES = [
    "Take one tablet twice a day.",
    "Do not exceed the recommended dose.",
    "This medicine contains 5 mg of amlodipine.",
    "Patients should not drink alcohol while taking ibuprofen.",
    "The tablet must not be chewed.",
    "Store below 25 degrees Celsius.",
    "Tell your doctor if you are pregnant or breast-feeding.",
    "The most common side effects include nausea and headache.",
    "Do not use this medicine if you are allergic to penicillin.",
    "Keep this medicine out of the sight and reach of children.",
]


def main() -> int:
    torch.manual_seed(SEED)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    translator = pipeline("translation", model=MODEL, device=device)

    t0 = time.perf_counter()
    outputs = [o["translation_text"] for o in translator(SENTENCES)]
    elapsed = time.perf_counter() - t0

    # Cheap invariants. Fail loudly if the plumbing is broken.
    assert len(outputs) == len(SENTENCES), "lost sentences in translation"
    assert all(o.strip() for o in outputs), "at least one empty translation"
    assert all(o != s for o, s in zip(outputs, SENTENCES)), "output == input (no-op)"

    for s, o in zip(SENTENCES, outputs):
        print(f"EN: {s}\nFI: {o}\n")
    print(f"{len(SENTENCES)} sentences in {elapsed:.2f}s on {device}")

    record = {
        "phase": 0,
        "model": MODEL,
        "seed": SEED,
        "n_sentences": len(SENTENCES),
        "elapsed_sec": round(elapsed, 3),
        "device": device,
        "pairs": [{"en": s, "fi": o} for s, o in zip(SENTENCES, outputs)],
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
