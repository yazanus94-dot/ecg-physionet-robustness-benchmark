"""Tests for MIT-BIH dataset access (skipped unless data/raw/mitdb is present)."""

import pytest

from ecg_robust.dataset import MITDB_DIR, database_records, load_ecg_record

pytestmark = pytest.mark.skipif(
    not MITDB_DIR.exists() or not database_records(MITDB_DIR),
    reason="MITDB raw data not downloaded",
)


def test_database_records_nonempty():
    records = database_records(MITDB_DIR)
    assert len(records) >= 40
    assert records == sorted(records)
    assert "100" in records


def test_load_ecg_record_100():
    rec, ann = load_ecg_record("100", MITDB_DIR)
    assert int(rec.fs) == 360
    assert len(ann.symbol) == len(ann.sample) > 0
    assert len(rec.sig_name) >= 1
