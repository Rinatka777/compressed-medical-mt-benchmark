"""convert.py — CT2 conversion sweep for Phase 2."""

import json
from pathlib import Path

import ctranslate2
import torch

MODELS = [
    {
        "alias": "marian",
        "hf_name": "Helsinki-NLP/opus-mt-en-fi",
        # MarianTokenizer needs these to be reconstructed standalone, without
        # re-resolving the HF repo at translate-time.
        "copy_files": ["source.spm", "target.spm", "vocab.json", "tokenizer_config.json"],
    },
    {
        "alias": "nllb",
        "hf_name": "facebook/nllb-200-distilled-600M",
        # NLLBTokenizer is one shared sentencepiece model plus config; no
        # per-direction .spm files like Marian.
        "copy_files": ["sentencepiece.bpe.model", "tokenizer_config.json", "special_tokens_map.json"],
    },
]
GPU_PRECISIONS = ["float16", "bfloat16"]
CPU_PRECISIONS = ["float32", "int16", "int8"]

CT2_ROOT = Path("models/ct2")
MANIFEST_PATH = CT2_ROOT / "manifest.json"


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
    ...


def convert_one(model_cfg: dict, precision: str) -> Path:
    """Run TransformersConverter for one (model, precision) pair, return output dir.
    Skip/return early if the directory already exists (idempotent reruns)."""
    ...


def load_manifest() -> dict:
    """Read existing manifest.json, or return an empty structure if absent."""
    ...


def record_conversion(manifest: dict, model_cfg: dict, precision: str, output_dir: Path) -> None:
    """Add/update one manifest entry (alias, precision, path, timestamp)."""
    ...


def save_manifest(manifest: dict) -> None:
    ...


def main() -> None:
    """Loop: for each model, for each available precision, convert_one + record.
    Save manifest once at the end (or after each conversion — your call, think about
    what happens if the script dies halfway through a 600M-param NLLB conversion)."""
    ...


if __name__ == "__main__":
    main()