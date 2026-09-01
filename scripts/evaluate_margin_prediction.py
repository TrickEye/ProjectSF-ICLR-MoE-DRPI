#!/usr/bin/env python3
"""Stage 3: write actual switches and first-order margin predictions."""

from __future__ import annotations

import argparse

import torch

from drpi.config import load_config
from drpi.datasets import load_prompts
from drpi.gradients import InjectionStateLeaf, margin_gradient
from drpi.loading import load_model_and_tokenizer
from drpi.margins import weakest_inside_best_outside
from drpi.metrics import topk_metrics
from drpi.model_adapter import OlmoeAdapter
from drpi.records import JsonlWriter, git_commit, utc_now
from drpi.router_capture import RouterCapture
from drpi.runner import ExperimentRunner
from drpi.static_space import blind_basis, project_to_blind, rms_scaled_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/olmoe_pilot.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--downstream-layer", type=int, required=True)
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prompts = load_prompts(config["data"]["prompts_jsonl"], split=args.split)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    model, tokenizer = load_model_and_tokenizer(config["model"])
    adapter = OlmoeAdapter(model)
    adapter.freeze_weights()
    runner = ExperimentRunner(adapter)
    device = next(model.parameters()).device
    basis = blind_basis(
        adapter.router_weight(args.layer), config["experiment"]["blind_rtol"]
    )
    writer = JsonlWriter(args.out)
    description = adapter.describe_layer(args.layer)
    for prompt_index, prompt in enumerate(prompts):
        encoded = tokenizer(
            prompt.text,
            return_tensors="pt",
            truncation=True,
            max_length=config["runtime"]["max_length"],
        )
        batch = {name: value.to(device) for name, value in encoded.items()}
        token = int(batch["attention_mask"].sum().item()) - 1
        baseline = runner.teacher_forced(batch)
        with (
            InjectionStateLeaf(adapter.shared_state_module(args.layer)) as leaf,
            RouterCapture(adapter, layers=[args.layer, args.downstream_layer]) as capture,
            torch.enable_grad(),
        ):
            model(**batch, use_cache=False, return_dict=True)
            logits = capture.records[args.downstream_layer].logits
            assert leaf.state is not None and logits is not None
            boundary = weakest_inside_best_outside(
                logits[0, token].detach(), adapter.top_k(args.downstream_layer)
            )
            gradient = margin_gradient(
                leaf.state,
                logits,
                injection_token=token,
                downstream_token=token,
                inside_expert=boundary.inside_expert,
                outside_expert=boundary.outside_expert,
            ).detach()
        generator = torch.Generator(device="cpu").manual_seed(
            config["runtime"]["seed"] + prompt_index
        )
        random_direction = torch.randn(basis.shape[0], generator=generator).to(
            device=basis.device, dtype=basis.dtype
        )
        static = project_to_blind(random_direction, basis)
        reference = baseline.routes[args.layer].shared_state
        assert reference is not None
        for alpha in config["experiment"]["alphas"]:
            delta = rms_scaled_delta(static, reference[0, token], alpha)
            edited = runner.teacher_forced(
                batch,
                injection_layer=args.layer,
                direction=delta,
                layers=[args.downstream_layer],
            )
            base_topk = baseline.routes[args.downstream_layer].topk_indices
            edit_topk = edited.routes[args.downstream_layer].topk_indices
            assert base_topk is not None and edit_topk is not None
            switched = not bool(
                topk_metrics(base_topk[0, token], edit_topk[0, token])["topk_set_exact"].item()
            )
            predicted_margin = boundary.value + float((gradient.float() * delta.float()).sum())
            writer.append(
                {
                    "schema_version": "drpi-run-v1",
                    "generated_at": utc_now(),
                    "model_id": config["model"]["id"],
                    "model_revision": config["model"]["revision"],
                    "router_weight_sha256": description.router_weight_sha256,
                    "code_commit": git_commit(),
                    "device": str(device),
                    "dtype": str(next(model.parameters()).dtype),
                    "quantization": config["model"]["quantization"],
                    "hook_path": description.shared_state_path,
                    "prompt_id": prompt.prompt_id,
                    "split": prompt.split,
                    "seed": config["runtime"]["seed"] + prompt_index,
                    "injection_layer": args.layer,
                    "token_position": token,
                    "alpha": alpha,
                    "horizon": args.downstream_layer - args.layer,
                    "downstream_layer": args.downstream_layer,
                    "baseline_margin": boundary.value,
                    "predicted_margin": predicted_margin,
                    "switch_score": -predicted_margin,
                    "actual_switch": switched,
                }
            )


if __name__ == "__main__":
    main()

