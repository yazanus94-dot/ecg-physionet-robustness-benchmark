"""Validated configuration loading for ECG-ROBUST.

The pipeline's defaults are the historical hardcoded values that produced the
committed results. `configs/default.yaml` is an optional override file: only
values explicitly present in it are applied. This keeps historical results
reproducible while making configuration genuinely functional and testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "fs": 360,
    "n_splits": 5,
    "random_state": 42,
    "segmentation": {"before_s": 0.25, "after_s": 0.45, "min_length": 3},
    "preprocessing": {"low_hz": 0.5, "high_hz": 40.0, "order": 4},
    "noise": {
        "snr_db": [24.0, 18.0, 12.0, 6.0, 0.0, -6.0],
        "n_noise_realizations": 8,
    },
    "classes": ["N", "S", "V", "F", "Q"],
}

ALLOWED_TOP_LEVEL = set(DEFAULT_CONFIG)
BOOL_VALUES = {True, False}


class ConfigError(ValueError):
    """Raised for invalid configuration values."""


def _validate(overrides: dict[str, Any]) -> None:
    unknown = set(overrides) - ALLOWED_TOP_LEVEL
    if unknown:
        raise ConfigError(f"unknown configuration keys: {sorted(unknown)}")
    for key in ("fs", "n_splits", "random_state"):
        if key in overrides and (
            not isinstance(overrides[key], int)
            or isinstance(overrides[key], bool)
            or overrides[key] <= 0
        ):
            raise ConfigError(f"{key} must be a positive integer")
    if "classes" in overrides and (
        not isinstance(overrides["classes"], list)
        or not all(isinstance(c, str) for c in overrides["classes"])
    ):
        raise ConfigError("classes must be a list of strings")
    seg = overrides.get("segmentation", {})
    if not isinstance(seg, dict):
        raise ConfigError("segmentation must be a mapping")
    for key, limits in (
        ("before_s", (0.01, 5.0)),
        ("after_s", (0.01, 5.0)),
        ("min_length", (1, 10000)),
    ):
        if key in seg:
            value = seg[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not limits[0] <= value <= limits[1]
            ):
                raise ConfigError(f"segmentation.{key} must be between {limits}")
    prep = overrides.get("preprocessing", {})
    if not isinstance(prep, dict):
        raise ConfigError("preprocessing must be a mapping")
    if "low_hz" in prep and (
        not isinstance(prep["low_hz"], (int, float)) or prep["low_hz"] <= 0
    ):
        raise ConfigError("preprocessing.low_hz must be positive")
    if "high_hz" in prep and (
        not isinstance(prep["high_hz"], (int, float)) or prep["high_hz"] <= 0
    ):
        raise ConfigError("preprocessing.high_hz must be positive")
    if "order" in prep and (not isinstance(prep["order"], int) or prep["order"] < 1):
        raise ConfigError("preprocessing.order must be a positive integer")
    if (
        isinstance(prep.get("low_hz"), (int, float))
        and isinstance(prep.get("high_hz"), (int, float))
        and prep["low_hz"] >= prep["high_hz"]
    ):
        raise ConfigError("preprocessing.low_hz must be below high_hz")
    noise = overrides.get("noise", {})
    if not isinstance(noise, dict):
        raise ConfigError("noise must be a mapping")
    if "snr_db" in noise and (
        not isinstance(noise["snr_db"], list)
        or not all(isinstance(v, (int, float)) for v in noise["snr_db"])
    ):
        raise ConfigError("noise.snr_db must be a list of numbers")
    if "n_noise_realizations" in noise and (
        not isinstance(noise["n_noise_realizations"], int)
        or noise["n_noise_realizations"] < 1
    ):
        raise ConfigError("noise.n_noise_realizations must be a positive integer")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load defaults, optionally overridden by a validated YAML file."""
    config: dict[str, Any] = {
        key: dict(value)
        if isinstance(value, dict)
        else list(value)
        if isinstance(value, list)
        else value
        for key, value in DEFAULT_CONFIG.items()
    }
    if path is None:
        return config
    config_file = Path(path)
    if not config_file.is_file():
        raise ConfigError(f"configuration file not found: {config_file}")
    try:
        raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_file}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    _validate(raw)
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def config_path_from_env() -> Path | None:
    """Optional `ECG_ROBUST_CONFIG` environment variable; no path means defaults."""
    import os

    value = os.environ.get("ECG_ROBUST_CONFIG")
    return Path(value) if value else None
