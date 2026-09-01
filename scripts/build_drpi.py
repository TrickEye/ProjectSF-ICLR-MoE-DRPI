#!/usr/bin/env python3
"""Stage 4: build a calibration-derived hard-projection DRPI artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from drpi.artifacts import save_drpi_artifact
from drpi.config import load_config
from drpi.datasets import load_prompts
from drpi.gradients import InjectionStateLeaf, margin_gradient
from drpi.loading import load_model_and_tokenizer
from drpi.margins import weakest_inside_best_outside
from drpi.model_adapter import OlmoeAdapter
from drpi.records import git_commit, utc_now
from drpi.router_capture import RouterCapture
from drpi.static_space import assert_router_blind, blind_basis
from drpi.subspace import dangerous_basis, drpi_direction, target_retention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/olmoe_pilot.yaml")
    parser.add_argument("--target-direction", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default="results/summary/drpi_build_report.json")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prompts = load_prompts(config["data"]["prompts_jsonl"], split="calibration")
    if args.limit is not None:
        prompts = prompts[: args.limit]
    if not prompts:
        raise ValueError("no calibration prompts")
    target_payload = torch.load(args.target_direction, map_location="cpu", weights_only=True)
    target = target_payload["direction"]

    model, tokenizer = load_model_and_tokenizer(config["model"])
    adapter = OlmoeAdapter(model)
    adapter.freeze_weights()
    device = next(model.parameters()).device
    basis = blind_basis(
        adapter.router_weight(args.layer), config["experiment"]["blind_rtol"]
    )
    target = target.to(device=device, dtype=basis.dtype)
    projected_gradients = []
    boundaries = []
    downstream_layers = list(
        range(args.layer + 1, min(adapter.num_layers(), args.layer + args.horizon + 1))
    )
    if not downstream_layers:
        raise ValueError("horizon contains no downstream layer")

    for prompt in prompts:
        encoded = tokenizer(
            prompt.text,
            return_tensors="pt",
            truncation=True,
            max_length=config["runtime"]["max_length"],
        )
        batch = {name: value.to(device) for name, value in encoded.items()}
        token = int(batch["attention_mask"].sum().item()) - 1
        with (
            InjectionStateLeaf(adapter.shared_state_module(args.layer)) as leaf,
            RouterCapture(adapter, layers=[args.layer, *downstream_layers]) as capture,
            torch.enable_grad(),
        ):
            model(**batch, use_cache=False, return_dict=True)
            assert leaf.state is not None
            for index, downstream_layer in enumerate(downstream_layers):
                logits = capture.records[downstream_layer].logits
                assert logits is not None
                boundary = weakest_inside_best_outside(
                    logits[0, token].detach(), adapter.top_k(downstream_layer)
                )
                gradient = margin_gradient(
                    leaf.state,
                    logits,
                    injection_token=token,
                    downstream_token=token,
                    inside_expert=boundary.inside_expert,
                    outside_expert=boundary.outside_expert,
                    retain_graph=index < len(downstream_layers) - 1,
                )
                projected_gradients.append(gradient.detach() @ basis)
                boundaries.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "layer": downstream_layer,
                        "token": token,
                        "inside": boundary.inside_expert,
                        "outside": boundary.outside_expert,
                        "margin": boundary.value,
                    }
                )
    gradient_matrix = torch.stack(projected_gradients)
    dangerous, summary = dangerous_basis(gradient_matrix, args.rank)
    direction = drpi_direction(target, basis, dangerous)
    assert_router_blind(adapter.router_weight(args.layer), direction)
    description = adapter.describe_layer(args.layer)
    metadata = {
        "schema_version": "drpi-artifact-v1",
        "generated_at": utc_now(),
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "router_weight_sha256": description.router_weight_sha256,
        "hook_path": description.shared_state_path,
        "backend": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "quantization": config["model"]["quantization"],
        "injection_layer": args.layer,
        "basis_rank": int(basis.shape[1]),
        "dangerous_rank": summary.usable_rank,
        "horizon": args.horizon,
        "calibration_prompt_ids": [prompt.prompt_id for prompt in prompts],
        "config": config,
        "code_commit": git_commit(),
        "target_retention": target_retention(target, direction),
    }
    save_drpi_artifact(
        args.out,
        direction=direction,
        blind_basis=basis,
        dangerous_basis=dangerous,
        metadata=metadata,
    )
    report = {
        **metadata,
        "boundaries": boundaries,
        "singular_values": summary.singular_values.detach().cpu().tolist(),
        "stop_target_removed": metadata["target_retention"] < 0.20,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
