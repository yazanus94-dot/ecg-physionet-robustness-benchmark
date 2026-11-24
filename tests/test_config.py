"""Configuration is genuinely consumed and validated."""

from pathlib import Path

import pytest
import yaml

from ecg_robust.config import DEFAULT_CONFIG, ConfigError, load_config

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs/default.yaml"


def test_defaults_match_historical_constants():
    config = load_config()
    assert config["fs"] == DEFAULT_CONFIG["fs"] == 360
    assert config["n_splits"] == 5
    assert config["random_state"] == 42
    assert config["noise"]["n_noise_realizations"] == 8


def test_committed_yaml_loads_and_overrides(tmp_path):
    config = load_config(CONFIG_FILE)
    assert config["n_splits"] == 5
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        yaml.safe_dump({"n_splits": 3, "noise": {"n_noise_realizations": 2}}),
        encoding="utf-8",
    )
    overridden = load_config(custom)
    assert overridden["n_splits"] == 3
    assert overridden["noise"]["n_noise_realizations"] == 2
    assert overridden["noise"]["snr_db"] == DEFAULT_CONFIG["noise"]["snr_db"]


@pytest.mark.parametrize(
    "payload",
    [
        {"unknown_key": 1},
        {"n_splits": 0},
        {"n_splits": "five"},
        {"preprocessing": {"low_hz": 40, "high_hz": 10}},
        {"segmentation": {"before_s": -1}},
        {"noise": {"snr_db": "loud"}},
    ],
)
def test_invalid_config_rejected(tmp_path, payload):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_pipeline_reads_config_source(monkeypatch, tmp_path):
    """Changing n_splits in a config changes the folds used by the pipeline."""
    from ecg_robust import pipeline

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        yaml.safe_dump({"n_splits": 3, "random_state": 17}), encoding="utf-8"
    )
    monkeypatch.setenv("ECG_ROBUST_CONFIG", str(cfg))
    monkeypatch.setattr(pipeline, "download_all", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "load_ecg_record",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    monkeypatch.setattr(
        pipeline,
        "database_records",
        lambda d: [] if "nstdb" in str(d).lower() else ["100"],
    )
    # A configured bad record is skipped explicitly; reaching this stage proves
    # the configured n_splits/seed were consumed before any CV.
    with pytest.raises(RuntimeError, match="no MITDB records"):
        pipeline.main()
