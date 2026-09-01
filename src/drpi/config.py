"""Configuration loading and runtime validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    if config.get("schema_version") != "drpi-config-v1":
        raise ValueError("unsupported or missing config schema_version")
    return config


def torch_dtype(name: str):
    """Resolve a configured dtype without silently falling back."""
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc

