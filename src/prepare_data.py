"""Phase 1: build the deduplicated EMEA en->fi train/dev/test split.

Only the data-acquisition half is implemented here (download + verify + unpack).
The pipeline itself -- load pairs, drop empty / implausible length ratio,
deduplicate on the source sentence, seeded shuffle, split (test = 2000),
write data/processed/* and results/phase1_prepare.json -- is still to be written.
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

# OPUS EMEA v3, Moses format. Line N of .en aligns to line N of .fi.
EMEA_URL = "https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/en-fi.txt.zip"
EMEA_ZIP_SHA256 = "f559de4103fa00871498fac6ebd5cc01f3addec31c7120bd2fccf30e266cb87b"
SRC_FILE = RAW_DIR / "EMEA.en-fi.en"
TGT_FILE = RAW_DIR / "EMEA.en-fi.fi"


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





if __name__ == "__main__":
    en_path, fi_path = ensure_raw_data()
    # TODO Phase 1 pipeline:
    #   load pairs -> drop empty / bad length ratio -> dedup on source
    #   -> seeded shuffle -> split (test = 2000) -> write data/processed/*
    #   -> write results/phase1_prepare.json (counts per step, seed, sha256)
