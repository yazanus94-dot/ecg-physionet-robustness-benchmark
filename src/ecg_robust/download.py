"""Download the official PhysioNet MIT-BIH databases with WFDB.

Databases (official PhysioNet, attribution and licensing per PhysioNet):

1. MIT-BIH Arrhythmia Database — ``https://physionet.org/content/mitdb/1.0.0/``
2. MIT-BIH Noise Stress Test Database — ``https://physionet.org/content/nstdb/1.0.0/``

Raw data is stored locally under ``data/raw/`` and is never uploaded.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import wfdb

RAW_ROOT = Path("data/raw")


def _download_with_wfdb(db_dir: str, dl_dir: Path) -> bool:
    """Download with ``wfdb.dl_database``; return True on success."""
    try:
        wfdb.dl_database(db_dir, str(dl_dir))
        return True
    except Exception:  # noqa: BLE001 -- intentional fallback to urllib
        return False


def _download_with_urllib(db_dir: str, dl_dir: Path, version: str = "1.0.0") -> None:
    """Manual HTTP fallback: download the PhysioNet database listing + files."""
    base = f"https://physionet.org/files/{db_dir}/{version}/"
    listing = urllib.request.urlopen(base).read().decode("utf-8", "replace")
    import re

    dl_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(set(re.findall(r'href="([^"#]+)"', listing)))
    files = [n for n in names if n.endswith((".dat", ".hea", ".atr", ".qrs"))]
    for name in files:
        url = base + name
        dest = dl_dir / name
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
            print(f"[download] {name}")


def download_database(db_dir: str, version: str = "1.0.0") -> Path:
    """Download ``db_dir`` (e.g. ``mitdb``) into ``data/raw/<db_dir>``."""
    dl_dir = RAW_ROOT / db_dir
    needs_download = not dl_dir.exists() or not list(dl_dir.glob("*.dat"))
    if needs_download and not _download_with_wfdb(db_dir, dl_dir):
        _download_with_urllib(db_dir, dl_dir, version)
    return dl_dir


def download_mitdb() -> Path:
    """Download the MIT-BIH Arrhythmia Database into ``data/raw/mitdb``."""
    return download_database("mitdb")


def download_nstdb() -> Path:
    """Download the MIT-BIH Noise Stress Test Database into ``data/raw/nstdb``."""
    return download_database("nstdb")


def download_all() -> tuple[Path, Path]:
    """Download both official databases; returns (mitdb_dir, nstdb_dir)."""
    return download_mitdb(), download_nstdb()
