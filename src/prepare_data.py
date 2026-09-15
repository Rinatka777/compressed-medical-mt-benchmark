"""Phase 1: build the deduplicated EMEA en->fi train/dev/test split.

Only the data-acquisition half is implemented here (download + verify + unpack).
The pipeline itself -- load pairs, drop empty / implausible length ratio,
deduplicate on the source sentence, seeded shuffle, split (test = 2000),
write data/processed/* and results/phase1_prepare.json -- is still to be written.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT/ "data"/ "processed"

# OPUS EMEA v3, Moses format. Line N of .en aligns to line N of .fi.
EMEA_URL = "https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/en-fi.txt.zip"
EMEA_ZIP_SHA256 = "f559de4103fa00871498fac6ebd5cc01f3addec31c7120bd2fccf30e266cb87b"
SRC_FILE = RAW_DIR / "EMEA.en-fi.en"
TGT_FILE = RAW_DIR / "EMEA.en-fi.fi"


PRC_TRAIN_FI = PROC_DIR/"train.fi"
PRC_DEV_FI = PROC_DIR/"dev.fi"
PRC_TEST_FI = PROC_DIR/ "test.fi"
PRC_TRAIN_EN = PROC_DIR/"train.en"
PRC_DEV_EN = PROC_DIR/"dev.en"
PRC_TEST_EN = PROC_DIR/ "test.en"




def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_raw_data() -> tuple[Path, Path]:
    """Download + unpack EMEA into data/raw/ if not already there. Idempotent."""
    if SRC_FILE.exists() and TGT_FILE.exists():
        print(f"raw data already present: {SRC_FILE.name}, {TGT_FILE.name}")
        return SRC_FILE, TGT_FILE

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "en-fi.txt.zip"

    if not zip_path.exists() or _sha256(zip_path) != EMEA_ZIP_SHA256:
        print(f"downloading {EMEA_URL}")
        tmp = zip_path.with_suffix(".zip.part")
        urllib.request.urlretrieve(EMEA_URL, tmp)
        digest = _sha256(tmp)
        if digest != EMEA_ZIP_SHA256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha256 mismatch for EMEA zip\n"
                f"  expected {EMEA_ZIP_SHA256}\n"
                f"  got      {digest}"
            )
        tmp.replace(zip_path)
        print(f"sha256 ok: {digest}")
    else:
        print(f"zip already present, sha256 ok: {EMEA_ZIP_SHA256}")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    print(f"extracted -> {SRC_FILE.name}, {TGT_FILE.name}")
    return SRC_FILE, TGT_FILE


def load_pairs() -> list[tuple[str, str]]:
    """Load all pairs from the EMEA dataset."""
    with SRC_FILE.open("r", encoding="utf-8") as f_en, TGT_FILE.open("r", encoding="utf-8") as f_fi:
        en_lines = [x.strip() for x in f_en]
        fi_lines = [x.strip() for x in f_fi]
        pairs = list(zip(en_lines, fi_lines, strict=True))
    return pairs

LENGTH_RATIO_MAX = 3


def data_clean() -> list[tuple[str, str]]:
    """Drop empty pairs and pairs with an implausible en/fi length ratio.

    Length is measured in characters (not words): Finnish is agglutinative,
    so word counts run structurally skewed vs. English even for correct
    translations, whereas character counts don't have that bias.
    """
    pairs = load_pairs()
    result = []
    for en, fi in pairs:
        if not en or not fi:
            continue
        ratio = max(len(en), len(fi)) / min(len(en), len(fi))
        if ratio > LENGTH_RATIO_MAX:
            continue
        result.append((en, fi))
    return result

def dedup() -> list[tuple[str, str]]:
    """Keep one row per distinct English source sentence (first occurrence wins)."""
    pairs = data_clean()
    seen = set()
    result = []
    for en, fi in pairs:
        normalized_en = en.strip().lower()
        if normalized_en in seen:
            continue
        seen.add(normalized_en)
        result.append((en, fi))
    return result

def data_shuffle() -> list[tuple[str,str]]:
    pairs = dedup()
    random.seed(42)
    random.shuffle(pairs)
    return pairs

def data_split(test_size=2000, dev_size=1000):
    pairs = data_shuffle()
    test = pairs[:test_size]
    dev = pairs[test_size:test_size + dev_size]
    train = pairs[test_size + dev_size:]
    return train, dev, test


def write_split(pairs: list[tuple[str, str]], en_path: Path, fi_path: Path) -> None:
    """Write a list of (en, fi) pairs to two line-aligned files, one sentence per line."""
    en_lines, fi_lines = zip(*pairs)
    en_path.write_text("\n".join(en_lines) + "\n", encoding="utf-8")
    fi_path.write_text("\n".join(fi_lines) + "\n", encoding="utf-8")




if __name__ == "__main__":
    ensure_raw_data()

    raw_pairs = load_pairs()
    cleaned_pairs = data_clean()
    deduped_pairs = dedup()
    train, dev, test = data_split()

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    write_split(train, PRC_TRAIN_EN, PRC_TRAIN_FI)
    write_split(dev, PRC_DEV_EN, PRC_DEV_FI)
    write_split(test, PRC_TEST_EN, PRC_TEST_FI)

    stats = {
        "seed": 42,
        "raw_pairs": len(raw_pairs),
        "after_clean": len(cleaned_pairs),
        "after_dedup": len(deduped_pairs),
        "dropped_by_clean": len(raw_pairs) - len(cleaned_pairs),
        "dropped_by_dedup": len(cleaned_pairs) - len(deduped_pairs),
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "emea_zip_sha256": EMEA_ZIP_SHA256,
    }
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "phase1_prepare.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"wrote {stats}")
