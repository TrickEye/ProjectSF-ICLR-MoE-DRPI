#!/usr/bin/env python3
"""Stage 5: effect-ready evaluation of shared-state intervention directions."""

from __future__ import annotations

import argparse
import json

import torch

from drpi.artifacts import load_drpi_artifact
from drpi.baselines import random_blind_subspace_direction
from drpi.config import load_config
from drpi.datasets import load_counterfactuals
from drpi.loading import load_model_and_tokenizer
from drpi.metrics import next_token_nll, output_kl, route_metrics
from drpi.model_adapter import OlmoeAdapter
from drpi.records import JsonlWriter, git_commit, utc_now
from drpi.runner import ExperimentRunner
from drpi.static_space import (
    blind_basis,
    project_to_blind,
    project_to_visible,
    rms_scaled_delta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/olmoe_pilot.yaml")
    parser.add_argument("--target-direction", required=True)
    parser.add_argument("--drpi-artifact", required=True)
    parser.add_argument("--out", default="results/raw/interventions.jsonl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--optimized-baseline",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add a calibrated direction artifact, e.g. output_kl=artifacts/output_kl.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pairs = load_counterfactuals(config["data"]["counterfactuals_jsonl"], split=args.split)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError(f"no pairs for split={args.split!r}")
    target_payload = torch.load(args.target_direction, map_location="cpu", weights_only=True)
    artifact = load_drpi_artifact(args.drpi_artifact)
    metadata = artifact["metadata"]
    layer = int(metadata["injection_layer"])
    model, tokenizer = load_model_and_tokenizer(config["model"])
    adapter = OlmoeAdapter(model)
    adapter.freeze_weights()
    description = adapter.describe_layer(layer)
    if metadata["model_revision"] != config["model"]["revision"]:
        raise ValueError("DRPI artifact model revision does not match configuration")
    if metadata["router_weight_sha256"] != description.router_weight_sha256:
        raise ValueError("DRPI artifact router weight hash does not match loaded model")
    device = next(model.parameters()).device
    basis = blind_basis(adapter.router_weight(layer), config["experiment"]["blind_rtol"])
    target = target_payload["direction"].to(device=device, dtype=basis.dtype)
    drpi = artifact["direction"].to(device=device, dtype=basis.dtype)
    static = project_to_blind(target, basis)
    retained_dim = basis.shape[1] - artifact["dangerous_basis"].shape[1]
    random_matched = random_blind_subspace_direction(
        target, basis, retained_dim, seed=config["runtime"]["seed"]
    )
    directions = {
        "full": target,
        "static_blind": static,
        "route_visible": project_to_visible(target, basis),
        "drpi": drpi,
        "random_blind_matched": random_matched,
    }
    for specification in args.optimized_baseline:
        if "=" not in specification:
            raise ValueError("optimized baseline must use NAME=PATH")
        name, path = specification.split("=", 1)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        directions[name] = payload["direction"].to(device=device, dtype=basis.dtype)
    runner = ExperimentRunner(adapter)
    writer = JsonlWriter(args.out)
    commit = git_commit()
    for pair in pairs:
        encoded = tokenizer(
            pair.source,
            return_tensors="pt",
            truncation=True,
            max_length=config["runtime"]["max_length"],
        )
        batch = {name: value.to(device) for name, value in encoded.items()}
        token = int(batch["attention_mask"].sum().item()) - 1
        target_tokens = tokenizer(pair.target_value, add_special_tokens=False)["input_ids"]
        if not target_tokens:
            raise ValueError(f"target value tokenized to empty sequence: {pair.pair_id}")
        target_token = int(target_tokens[0])
        baseline = runner.teacher_forced(batch)
        base_target_logp = float(
            torch.log_softmax(baseline.output_logits[0, token].float(), dim=-1)[target_token].item()
        )
        for method, raw_direction in directions.items():
            for alpha in config["experiment"]["alphas"]:
                reference = baseline.routes[layer].shared_state
                assert reference is not None
                delta = rms_scaled_delta(raw_direction, reference[0, token], alpha)
                edited = runner.teacher_forced(
                    batch, injection_layer=layer, direction=delta, alpha=1.0
                )
                edit_target_logp = float(
                    torch.log_softmax(edited.output_logits[0, token].float(), dim=-1)[
                        target_token
                    ].item()
                )
                full_output_kl = float(output_kl(baseline.output_logits, edited.output_logits).mean())
                base_nll = next_token_nll(baseline.output_logits, batch["input_ids"])
                edit_nll = next_token_nll(edited.output_logits, batch["input_ids"])
                nll_delta = float((edit_nll - base_nll).mean().item())
                for downstream_layer in range(layer, adapter.num_layers()):
                    base_route = baseline.routes[downstream_layer]
                    edit_route = edited.routes[downstream_layer]
                    assert base_route.logits is not None and edit_route.logits is not None
                    assert base_route.topk_indices is not None
                    assert edit_route.topk_indices is not None
                    metrics = route_metrics(
                        base_route.logits,
                        edit_route.logits,
                        base_route.topk_indices,
                        edit_route.topk_indices,
                    )
                    values = {
                        name: bool(value[0, token].item())
                        if value.dtype == torch.bool
                        else float(value[0, token].item())
                        for name, value in metrics.items()
                    }
                    writer.append(
                        {
                            "schema_version": "drpi-run-v1",
                            "generated_at": utc_now(),
                            "model_id": config["model"]["id"],
                            "model_revision": config["model"]["revision"],
                            "router_weight_sha256": description.router_weight_sha256,
                            "code_commit": commit,
                            "device": str(device),
                            "dtype": str(next(model.parameters()).dtype),
                            "quantization": config["model"]["quantization"],
                            "hook_path": description.shared_state_path,
                            "prompt_id": pair.pair_id,
                            "split": pair.split,
                            "seed": config["runtime"]["seed"],
                            "injection_layer": layer,
                            "token_position": token,
                            "alpha": alpha,
                            "horizon": downstream_layer - layer,
                            "downstream_layer": downstream_layer,
                            "method": method,
                            "strength_space": "shared_state_rms",
                            "target_token": target_token,
                            "target_logprob_baseline": base_target_logp,
                            "target_logprob_edited": edit_target_logp,
                            "target_effect": edit_target_logp - base_target_logp,
                            "output_kl": full_output_kl,
                            "next_token_nll_delta": nll_delta,
                            **values,
                        }
                    )
        router_direction = adapter.router_weight(layer).float() @ target.float()
        base_router_logits = baseline.routes[layer].logits
        assert base_router_logits is not None
        router_direction_rms = router_direction.square().mean().sqrt()
        if float(router_direction_rms.item()) > 0.0:
            baseline_router_rms = base_router_logits[0, token].float().square().mean().sqrt()
            for alpha in config["experiment"]["alphas"]:
                router_bias = router_direction * (
                    float(alpha) * baseline_router_rms / router_direction_rms
                )
                edited = runner.teacher_forced(
                    batch, router_bias_layer=layer, router_bias=router_bias
                )
                edit_target_logp = float(
                    torch.log_softmax(edited.output_logits[0, token].float(), dim=-1)[
                        target_token
                    ].item()
                )
                full_output_kl = float(output_kl(baseline.output_logits, edited.output_logits).mean())
                base_nll = next_token_nll(baseline.output_logits, batch["input_ids"])
                edit_nll = next_token_nll(edited.output_logits, batch["input_ids"])
                nll_delta = float((edit_nll - base_nll).mean().item())
                for downstream_layer in range(layer, adapter.num_layers()):
                    base_route = baseline.routes[downstream_layer]
                    edit_route = edited.routes[downstream_layer]
                    assert base_route.logits is not None and edit_route.logits is not None
                    assert base_route.topk_indices is not None
                    assert edit_route.topk_indices is not None
                    metrics = route_metrics(
                        base_route.logits,
                        edit_route.logits,
                        base_route.topk_indices,
                        edit_route.topk_indices,
                    )
                    values = {
                        name: bool(value[0, token].item())
                        if value.dtype == torch.bool
                        else float(value[0, token].item())
                        for name, value in metrics.items()
                    }
                    writer.append(
                        {
                            "schema_version": "drpi-run-v1",
                            "generated_at": utc_now(),
                            "model_id": config["model"]["id"],
                            "model_revision": config["model"]["revision"],
                            "router_weight_sha256": description.router_weight_sha256,
                            "code_commit": commit,
                            "device": str(device),
                            "dtype": str(next(model.parameters()).dtype),
                            "quantization": config["model"]["quantization"],
                            "hook_path": description.router_path,
                            "prompt_id": pair.pair_id,
                            "split": pair.split,
                            "seed": config["runtime"]["seed"],
                            "injection_layer": layer,
                            "token_position": token,
                            "alpha": alpha,
                            "horizon": downstream_layer - layer,
                            "downstream_layer": downstream_layer,
                            "method": "direct_router_bias",
                            "strength_space": "router_logit_rms",
                            "target_token": target_token,
                            "target_logprob_baseline": base_target_logp,
                            "target_logprob_edited": edit_target_logp,
                            "target_effect": edit_target_logp - base_target_logp,
                            "output_kl": full_output_kl,
                            "next_token_nll_delta": nll_delta,
                            **values,
                        }
                    )


if __name__ == "__main__":
    main()
