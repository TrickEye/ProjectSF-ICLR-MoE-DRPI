"""Append-only experiment records and provenance metadata."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REQUIRED_RUN_FIELDS = {
    "schema_version",
    "generated_at",
    "model_id",
    "model_revision",
    "router_weight_sha256",
    "code_commit",
    "device",
    "dtype",
    "quantization",
    "hook_path",
    "prompt_id",
    "split",
    "seed",
    "injection_layer",
    "token_position",
    "alpha",
    "horizon",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(cwd: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def validate_run_record(record: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_RUN_FIELDS - set(record))
    if missing:
        raise ValueError(f"run record missing required fields: {', '.join(missing)}")


class JsonlWriter:
    """Flush and fsync append-only records so long runs are resumable."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any], *, validate: bool = True) -> None:
        if validate:
            validate_run_record(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def completed_keys(self, fields: Iterable[str]) -> set[tuple[object, ...]]:
        names = tuple(fields)
        if not self.path.exists():
            return set()
        result: set[tuple[object, ...]] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    result.add(tuple(record.get(name) for name in names))
        return result

