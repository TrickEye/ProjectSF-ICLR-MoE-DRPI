"""Version-bound DRPI artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


REQUIRED_ARTIFACT_METADATA = {
    "schema_version",
    "model_id",
    "model_revision",
    "router_weight_sha256",
    "hook_path",
    "backend",
    "dtype",
    "quantization",
    "injection_layer",
    "basis_rank",
    "dangerous_rank",
    "horizon",
    "calibration_prompt_ids",
    "config",
}


def save_drpi_artifact(
    path: str | Path,
    *,
    direction: torch.Tensor,
    blind_basis: torch.Tensor,
    dangerous_basis: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    missing = sorted(REQUIRED_ARTIFACT_METADATA - set(metadata))
    if missing:
        raise ValueError(f"artifact metadata missing: {', '.join(missing)}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "direction": direction.detach().cpu(),
        "blind_basis": blind_basis.detach().cpu(),
        "dangerous_basis": dangerous_basis.detach().cpu(),
        "metadata_json": json.dumps(metadata, ensure_ascii=True, sort_keys=True),
    }
    torch.save(payload, destination)


def load_drpi_artifact(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    payload["metadata"] = json.loads(payload.pop("metadata_json"))
    return payload

