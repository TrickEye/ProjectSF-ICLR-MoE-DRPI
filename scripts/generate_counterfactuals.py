#!/usr/bin/env python3
"""Generate deterministic local mechanism prompts and counterfactual pairs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from drpi.datasets import CounterfactualPair, PromptRecord, serialize_records


ENTITIES = [
    ("Alice", "Rome", "Madrid"),
    ("Bob", "Paris", "Lyon"),
    ("Carol", "Berlin", "Vienna"),
    ("David", "Tokyo", "Osaka"),
    ("Eve", "Dublin", "Cork"),
    ("Frank", "Lisbon", "Porto"),
    ("Grace", "Toronto", "Ottawa"),
    ("Heidi", "Sydney", "Perth"),
    ("Ivan", "Prague", "Brno"),
    ("Judy", "Zurich", "Geneva"),
]
COLORS = [("red", "blue"), ("green", "yellow"), ("black", "white"), ("orange", "purple")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/counterfactuals.jsonl")
    parser.add_argument("--prompts", default="data/prompts.jsonl")
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs: list[CounterfactualPair] = []
    prompts: list[PromptRecord] = []
    identifiers = [f"cf-{index:04d}" for index in range(400)]
    ordered = sorted(
        identifiers,
        key=lambda value: hashlib.sha256(f"{args.seed}:{value}".encode("ascii")).digest(),
    )
    exact_splits = {
        identifier: ("train", "calibration", "validation", "test")[rank // 100]
        for rank, identifier in enumerate(ordered)
    }
    for index in range(400):
        person, source_city, target_city = ENTITIES[index % len(ENTITIES)]
        source_color, target_color = COLORS[(index // len(ENTITIES)) % len(COLORS)]
        if index % 2 == 0:
            source = f"Record {index}: {person} lives in {source_city}. The answer is"
            target = f"Record {index}: {person} lives in {target_city}. The answer is"
            source_value, target_value = source_city, target_city
        else:
            source = f"Record {index}: the marked box is {source_color}. Its color is"
            target = f"Record {index}: the marked box is {target_color}. Its color is"
            source_value, target_value = source_color, target_color
        identifier = f"cf-{index:04d}"
        split = exact_splits[identifier]
        pairs.append(
            CounterfactualPair(
                pair_id=identifier,
                source=source,
                target=target,
                source_value=source_value,
                target_value=target_value,
                split=split,
            )
        )
        prompts.append(PromptRecord(prompt_id=identifier, text=source, split=split))
    pair_path, prompt_path = Path(args.pairs), Path(args.prompts)
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    pair_path.write_text(serialize_records(pairs), encoding="utf-8")
    prompt_path.write_text(serialize_records(prompts), encoding="utf-8")


if __name__ == "__main__":
    main()
