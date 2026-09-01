#!/usr/bin/env python3
"""Build a train-only mean counterfactual shared-state direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from drpi.config import load_config
from drpi.datasets import load_counterfactuals
from drpi.loading import load_model_and_tokenizer
from drpi.model_adapter import OlmoeAdapter
from drpi.records import git_commit, utc_now
from drpi.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/olmoe_pilot.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pairs = load_counterfactuals(config["data"]["counterfactuals_jsonl"], split="train")
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("no train counterfactual pairs")
    model, tokenizer = load_model_and_tokenizer(config["model"])
    adapter = OlmoeAdapter(model)
    adapter.freeze_weights()
    runner = ExperimentRunner(adapter)
    device = next(model.parameters()).device
    differences = []
    for pair in pairs:
        states = []
        for text in (pair.source, pair.target):
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config["runtime"]["max_length"],
            )
            batch = {name: value.to(device) for name, value in encoded.items()}
            token = int(batch["attention_mask"].sum().item()) - 1
            result = runner.teacher_forced(batch, layers=[args.layer])
            state = result.routes[args.layer].shared_state
            assert state is not None
            states.append(state[0, token].float())
        differences.append(states[1] - states[0])
    direction = torch.stack(differences).mean(dim=0)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "target-direction-v1",
        "generated_at": utc_now(),
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "router_weight_sha256": adapter.describe_layer(args.layer).router_weight_sha256,
        "hook_path": adapter.describe_layer(args.layer).shared_state_path,
        "injection_layer": args.layer,
        "train_pair_ids": [pair.pair_id for pair in pairs],
        "code_commit": git_commit(),
    }
    torch.save(
        {"direction": direction.cpu(), "metadata_json": json.dumps(metadata, sort_keys=True)},
        destination,
    )


if __name__ == "__main__":
    main()

