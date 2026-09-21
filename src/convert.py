"""convert.py — CT2 conversion sweep for Phase 2."""

import json
from datetime import datetime, timezone
from pathlib import Path
import os.path
import ctranslate2
import torch
from ctranslate2.converters import TransformersConverter
from transformers import AutoTokenizer

# convert.py lives at <repo>/scripts/ — adjust if this file moves
REPO_ROOT = Path(__file__).resolve().parent.parent
CT2_ROOT = REPO_ROOT / "models" / "ct2"
MANIFEST_PATH = CT2_ROOT / "manifest.json"

# revision: pin to a commit SHA from the HF repo's "Files and versions" tab.
# None means "whatever main points at today" — fine while prototyping,
# not fine once you're reporting numbers.
MODELS = [
    {
        "alias": "opus-en-fi",
        "hf_name": "Helsinki-NLP/opus-mt-en-fi",
        "revision": None,
        "family": "marian",
        # MarianTokenizer needs these to be reconstructed standalone, without
        # re-resolving the HF repo at translate-time.
        "copy_files": [
            "source.spm",
            "target.spm",
            "vocab.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    },
    {
        "alias": "nllb-600m",
        "hf_name": "facebook/nllb-200-distilled-600M",
        "revision": None,
        "family": "nllb",
        # NLLB uses one shared sentencepiece model plus config; no
        # per-direction .spm files like Marian. tokenizer.json is the fast
        # tokenizer — without it you silently fall back to the slow path.
        "copy_files": [
            "sentencepiece.bpe.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    },
]

CPU_PRECISIONS = ["float32", "int16", "int8_float32"]
GPU_PRECISIONS = ["float16", "bfloat16"]


def available_precisions() -> list[str]:
    """CPU_PRECISIONS + whichever GPU precisions this machine actually supports."""
    cpu_supported = ctranslate2.get_supported_compute_types("cpu")
    precisions = [p for p in CPU_PRECISIONS if p in cpu_supported]

    if ctranslate2.get_cuda_device_count() > 0:
        gpu_supported = ctranslate2.get_supported_compute_types("cuda")
        precisions += [p for p in GPU_PRECISIONS if p in gpu_supported]
    return precisions


def output_dir_for(alias: str, precision: str) -> Path:
    """models/ct2/<alias>/<precision>/"""
    return CT2_ROOT / alias / precision


def convert_one(model_cfg: dict, precision: str, force: bool = False) -> Path:
    """Run TransformersConverter for one (model, precision) pair, return output dir.

    Skips if model.bin is already present, so reruns are idempotent. Checked on
    model.bin rather than the directory: a conversion that died halfway leaves
    the directory behind, and a directory check would skip it forever.
    """
    out = output_dir_for(model_cfg["alias"], precision)

    if (out / "model.bin").exists() and not force:
        return out

    converter = TransformersConverter(
        model_cfg["hf_name"],
        copy_files=model_cfg["copy_files"],
        revision=model_cfg["revision"],
    )
    converter.convert(str(out), quantization=precision, force=True)

    # Fail here, loudly, if copy_files missed something — not in translate.py
    # three days from now with a corrupt-looking translation and no error.
    AutoTokenizer.from_pretrained(str(out))

    return out



def load_manifest() -> dict:
    """Read existing manifest.json, or return an empty structure if absent."""
    if not MANIFEST_PATH.exists():
        return {"entries": []}
    with MANIFEST_PATH.open(encoding = "utf-8") as f:
        return json.load(f)


def record_conversion(manifest: dict, model_cfg: dict, precision: str, output_dir: Path) -> None:
    """Add/update one manifest entry (alias, precision, path, timestamp)."""
    new_entry = {
        "alias": model_cfg["alias"],
        "precision": precision,
        "path": str(output_dir),
        "hf_name": model_cfg["hf_name"],
        "revision": model_cfg["revision"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "size_bytes": (output_dir / "model.bin").stat().st_size,
    }
    for i, entry in enumerate(manifest["entries"]):
        if entry["alias"] == new_entry["alias"] and entry["precision"] == new_entry["precision"]:
            manifest["entries"][i] = new_entry
            return
    manifest["entries"].append(new_entry)

def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    """Loop: for each model, for each available precision, convert_one + record.
    Save manifest once at the end (or after each conversion — your call, think about
    what happens if the script dies halfway through a 600M-param NLLB conversion)."""
    ...


if __name__ == "__main__":
    main()