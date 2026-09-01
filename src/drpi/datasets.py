"""Deterministic prompt records, splits, and counterfactual pairs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    text: str
    split: str


@dataclass(frozen=True)
class CounterfactualPair:
    pair_id: str
    source: str
    target: str
    source_value: str
    target_value: str
    split: str


def stable_split(identifier: str, seed: int = 1729) -> str:
    """Map IDs deterministically to equal train/calibration/validation/test bins."""
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 4
    return ("train", "calibration", "validation", "test")[bucket]


def prompt_id(text: str) -> str:
    """Create a stable content-derived prompt identifier."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_jsonl(path: str | Path) -> Iterator[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def load_prompts(path: str | Path, split: str | None = None) -> list[PromptRecord]:
    records = [PromptRecord(**value) for value in load_jsonl(path)]
    return records if split is None else [record for record in records if record.split == split]


def load_counterfactuals(
    path: str | Path, split: str | None = None
) -> list[CounterfactualPair]:
    records = [CounterfactualPair(**value) for value in load_jsonl(path)]
    return records if split is None else [record for record in records if record.split == split]


def serialize_records(records: Iterable[PromptRecord | CounterfactualPair]) -> str:
    """Serialize records as deterministic JSONL for an explicit write step."""
    return "".join(
        json.dumps(asdict(record), ensure_ascii=True, sort_keys=True) + "\n" for record in records
    )

