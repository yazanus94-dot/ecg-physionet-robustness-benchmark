"""MIT-BIH Arrhythmia Database access and metadata.

Records are discovered from the downloaded ``data/raw/mitdb`` directory
(the MITDB contains 48 recordings; all segmentation and classification
is performed on these recordings). The MIT-BIH Noise Stress Test
Database (``data/raw/nstdb``) provides the matched noisy versions and
the pure noise-stress records 118 and 119.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import wfdb

RAW_ROOT = Path("data/raw")
MITDB_DIR = RAW_ROOT / "mitdb"
NSTDB_DIR = RAW_ROOT / "nstdb"


def mitdb_directory() -> Path:
    """Directory of the downloaded MIT-BIH Arrhythmia Database."""
    return MITDB_DIR


def nstdb_directory() -> Path:
    """Directory of the downloaded MIT-BIH Noise Stress Test Database."""
    return NSTDB_DIR


def database_records(directory: Path) -> list[str]:
    """Record IDs present in a downloaded PhysioNet directory (scan ``*.dat``)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted({p.stem for p in directory.glob("*.dat") if p.is_file()})


class RecordLike(NamedTuple):
    """Normalized WFDB record: ``samples`` (n_samples, n_channels), ``fs``, ``sig_name``.

    ``wfdb.rdsamp`` returns ``(signals, fields)`` in wfdb 4.x (an older
    Record object in earlier versions); both are normalized to this shape.
    """

    samples: np.ndarray
    fs: int
    sig_name: tuple


def _normalize_record(out):
    """Normalize ``wfdb.rdsamp`` output: ``(sig, fields_dict)`` -> :class:`RecordLike`."""
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        sig, fields = out
        return RecordLike(
            samples=np.asarray(sig),
            fs=int(fields.get("fs", 0)),
            sig_name=tuple(fields.get("sig_name", ())),
        )
    return out  # legacy Record object


def _read_local(record_id: str, directory: Path, fn_name: str,
                extension: str | None = None):
    """Read a WFDB record/annotation from the local downloaded files.

    ``extension`` is required by ``wfdb.rdann`` (e.g. ``"atr"`` for MITDB
    annotations); ``rdsamp`` takes none. Records are read via a temporary
    ``os.chdir`` into ``directory`` because ``wfdb.rdsamp(pn_dir=...)``
    misbehaves for local paths.
    """
    directory = Path(directory)
    prev = os.getcwd()
    try:
        os.chdir(directory)
        if extension is None:
            out = getattr(wfdb, fn_name)(record_id)
        else:
            out = getattr(wfdb, fn_name)(record_id, extension)
        return _normalize_record(out) if fn_name == "rdsamp" else out
    finally:
        os.chdir(prev)


def _signal_and_fs(rec) -> tuple[np.ndarray, int]:
    """Extract ``(signal (n_samples, n_channels), fs)`` from a WFDB record."""
    if hasattr(rec, "samples"):
        sig = rec.samples
    else:
        sig = rec.to_dataframe().to_numpy()
    return np.asarray(sig), int(getattr(rec, "fs", 0))


def load_ecg_record(record_id: str, directory: Path = MITDB_DIR):
    """Load an ECG recording and its reference annotations via WFDB.

    Returns ``(rec, ann)``: the recording with ``rec.sig_name`` (channels),
    ``rec.fs`` (sampling frequency) and the signal array; and the
    annotation with ``ann.symbol`` (codes) and ``ann.sample`` (positions).
    """
    rec = _read_local(record_id, directory, "rdsamp")
    ann = _read_local(record_id, directory, "rdann", extension="atr")
    return rec, ann


def load_nstdb_signal(record_id: str, directory: Path = NSTDB_DIR) -> np.ndarray:
    """Load a noise-stress recording via WFDB (used as a noise realization)."""
    rec = _read_local(record_id, directory, "rdsamp")
    sig, _fs = _signal_and_fs(rec)
    return sig


def nstdb_metadata() -> dict:
    """Metadata of the NSTDB: records and channel/sampling info."""
    out = {}
    for rec_id in database_records(NSTDB_DIR):
        rec = _read_local(rec_id, NSTDB_DIR, "rdsamp")
        sig, fs = _signal_and_fs(rec)
        out[rec_id] = {
            "fs": fs,
            "channels": [str(c) for c in rec.sig_name],
            "n_samples": int(sig.shape[0]),
        }
    return out
