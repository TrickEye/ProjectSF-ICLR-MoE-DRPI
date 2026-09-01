#!/usr/bin/env python3
"""Build equal-budget target-only, output-KL, or retain-loss direction baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from drpi.config import load_config
from drpi.datasets import load_counterfactuals
from drpi.loading import load_model_and_tokenizer
from drpi.metrics import next_token_nll, output_kl
from drpi.model_adapter import OlmoeAdapter
from drpi.records import git_commit, utc_now
from drpi.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/olmoe_pilot.yaml")
    parser.add_argument("--initial-direction", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--mode", choices=("target_only", "output_kl", "retain_loss"), required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--l2-weight", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.limit <= 0:
        raise ValueError("steps and limit must be positive")
    config = load_config(args.config)
    pairs = load_counterfactuals(
        config["data"]["counterfactuals_jsonl"], split="calibration"
    )[: args.limit]
    payload = torch.load(args.initial_direction, map_location="cpu", weights_only=True)
    model, tokenizer = load_model_and_tokenizer(config["model"])
    adapter = OlmoeAdapter(model)
    adapter.freeze_weights()
    runner = ExperimentRunner(adapter)
    device = next(model.parameters()).device
    direction = payload["direction"].to(device=device, dtype=next(model.parameters()).dtype)
    direction = direction.float().detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([direction], lr=args.learning_rate)

    batches = []
    for pair in pairs:
        encoded = tokenizer(
            pair.source,
            return_tensors="pt",
            truncation=True,
            max_length=config["runtime"]["max_length"],
        )
        batch = {name: value.to(device) for name, value in encoded.items()}
        target_ids = tokenizer(pair.target_value, add_special_tokens=False)["input_ids"]
        if not target_ids:
            continue
        with torch.no_grad():
            baseline = runner.teacher_forced(batch)
        batches.append((batch, int(target_ids[0]), baseline.output_logits.detach()))
    if not batches:
        raise ValueError("no usable calibration examples")

    history = []
    for step in range(args.steps):
        batch, target_token, baseline_logits = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        token = int(batch["attention_mask"].sum().item()) - 1
        edited = runner.teacher_forced(
            batch,
            injection_layer=args.layer,
            direction=direction,
            alpha=1.0,
            require_grad=True,
        )
        target_loss = -torch.log_softmax(edited.output_logits[0, token].float(), dim=-1)[
            target_token
        ]
        constraint = torch.zeros((), device=device)
        if args.mode == "output_kl":
            constraint = output_kl(baseline_logits, edited.output_logits).mean()
        elif args.mode == "retain_loss":
            base_nll = next_token_nll(baseline_logits, batch["input_ids"])
            edit_nll = next_token_nll(edited.output_logits, batch["input_ids"])
            constraint = torch.relu(edit_nll - base_nll).mean()
        loss = target_loss + args.constraint_weight * constraint + args.l2_weight * direction.square().mean()
        loss.backward()
        optimizer.step()
        history.append(
            {
                "step": step,
                "target_loss": float(target_loss.detach().item()),
                "constraint": float(constraint.detach().item()),
                "total_loss": float(loss.detach().item()),
            }
        )
    metadata = {
        "schema_version": "optimized-baseline-v1",
        "generated_at": utc_now(),
        "mode": args.mode,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "router_weight_sha256": adapter.describe_layer(args.layer).router_weight_sha256,
        "hook_path": adapter.describe_layer(args.layer).shared_state_path,
        "injection_layer": args.layer,
        "calibration_pair_ids": [pair.pair_id for pair in pairs],
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "constraint_weight": args.constraint_weight,
        "l2_weight": args.l2_weight,
        "code_commit": git_commit(),
        "history": history,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "direction": direction.detach().cpu(),
            "metadata_json": json.dumps(metadata, sort_keys=True),
        },
        destination,
    )


if __name__ == "__main__":
    main()
