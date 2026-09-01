#!/usr/bin/env python3
"""Stage 2: measure teacher-forced downstream leakage of static blind edits."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from drpi.config import load_config
from drpi.datasets import load_prompts
from drpi.loading import load_model_and_tokenizer
from drpi.margins import weakest_inside_best_outside
from drpi.metrics import output_kl, route_metrics
from drpi.model_adapter import OlmoeAdapter
from drpi.records import JsonlWriter, git_commit, utc_now
from drpi.runner import ExperimentRunner
from drpi.static_space import (
    assert_router_blind,
    blind_basis,
    project_to_blind,
    rms_scaled_delta,
    router_null_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/olmoe_pilot.yaml")
    parser.add_argument("--out", default="results/raw/leakage.jsonl")
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def scalar_at(tensor: torch.Tensor, token: int):
    value = tensor[0, token]
    return bool(value.item()) if value.dtype == torch.bool else float(value.item())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prompts = load_prompts(config["data"]["prompts_jsonl"], split=args.split)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    if not prompts:
        raise ValueError(f"no prompts available for split={args.split!r}")

    model, tokenizer = load_model_and_tokenizer(config["model"])
    adapter = OlmoeAdapter(model)
    adapter.freeze_weights()
    runner = ExperimentRunner(adapter)
    device = next(model.parameters()).device
    writer = JsonlWriter(args.out)
    key_fields = ("prompt_id", "injection_layer", "seed", "alpha", "downstream_layer")
    completed = writer.completed_keys(key_fields)
    commit = git_commit()

    for prompt in prompts:
        encoded = tokenizer(
            prompt.text,
            return_tensors="pt",
            truncation=True,
            max_length=config["runtime"]["max_length"],
        )
        batch = {name: value.to(device) for name, value in encoded.items()}
        token = int(batch["attention_mask"].sum(dim=1).item()) - 1
        baseline = runner.teacher_forced(batch)
        for injection_layer in config["experiment"]["injection_layers"]:
            basis = blind_basis(
                adapter.router_weight(injection_layer), config["experiment"]["blind_rtol"]
            )
            reference = baseline.routes[injection_layer].shared_state
            assert reference is not None
            for seed in config["experiment"]["direction_seeds"]:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                random_direction = torch.randn(
                    basis.shape[0], generator=generator, dtype=torch.float32
                ).to(device=basis.device, dtype=basis.dtype)
                static_direction = project_to_blind(random_direction, basis)
                assert_router_blind(adapter.router_weight(injection_layer), static_direction)
                for alpha in config["experiment"]["alphas"]:
                    delta = rms_scaled_delta(static_direction, reference[0, token], alpha)
                    edited = runner.teacher_forced(
                        batch,
                        injection_layer=injection_layer,
                        direction=delta,
                        alpha=1.0,
                    )
                    base_injection = baseline.routes[injection_layer]
                    edit_injection = edited.routes[injection_layer]
                    assert base_injection.logits is not None and edit_injection.logits is not None
                    assert base_injection.topk_indices is not None
                    assert edit_injection.topk_indices is not None
                    if not torch.allclose(
                        base_injection.logits[0, token].float(),
                        edit_injection.logits[0, token].float(),
                        rtol=1e-4,
                        atol=2e-5,
                    ) or not torch.equal(
                        base_injection.topk_indices[0, token],
                        edit_injection.topk_indices[0, token],
                    ):
                        raise AssertionError("T1 failed during leakage run; downstream run aborted")

                    external_kl = float(
                        output_kl(baseline.output_logits, edited.output_logits)[0, token].item()
                    )
                    null_error = router_null_error(adapter.router_weight(injection_layer), delta)
                    description = adapter.describe_layer(injection_layer)
                    for downstream_layer in range(injection_layer, adapter.num_layers()):
                        key = (prompt.prompt_id, injection_layer, seed, alpha, downstream_layer)
                        if key in completed:
                            continue
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
                        boundary = weakest_inside_best_outside(
                            base_route.logits[0, token], adapter.top_k(downstream_layer)
                        )
                        record = {
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
                            "prompt_id": prompt.prompt_id,
                            "split": prompt.split,
                            "seed": seed,
                            "injection_layer": injection_layer,
                            "token_position": token,
                            "alpha": alpha,
                            "horizon": downstream_layer - injection_layer,
                            "downstream_layer": downstream_layer,
                            "direction_source": "seeded_gaussian_static_blind",
                            "router_null_absolute": null_error["absolute"],
                            "router_null_relative": null_error["relative"],
                            "baseline_margin": boundary.value,
                            "output_kl": external_kl,
                            **{name: scalar_at(value, token) for name, value in metrics.items()},
                        }
                        writer.append(record)
                        completed.add(key)


if __name__ == "__main__":
    main()

