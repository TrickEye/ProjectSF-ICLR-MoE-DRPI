#!/usr/bin/env python3
"""Stage 3 T4: finite-difference validation of downstream margin gradients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from drpi.config import load_config
from drpi.datasets import load_prompts
from drpi.gradients import (
    InjectionStateLeaf,
    best_finite_difference,
    finite_difference_checks,
    margin_gradient,
)
from drpi.loading import load_model_and_tokenizer
from drpi.margins import margin_value, weakest_inside_best_outside
from drpi.model_adapter import OlmoeAdapter
from drpi.records import git_commit, utc_now
from drpi.router_capture import RouterCapture
from drpi.runner import ExperimentRunner
from drpi.static_space import blind_basis, project_to_blind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/olmoe_pilot.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--downstream-layer", type=int, required=True)
    parser.add_argument("--out", default="results/summary/margin_gradient_report.json")
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--limit", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.downstream_layer <= args.layer:
        raise ValueError("downstream layer must follow injection layer")
    config = load_config(args.config)
    prompts = load_prompts(config["data"]["prompts_jsonl"], split=args.split)[: args.limit]
    model, tokenizer = load_model_and_tokenizer(config["model"])
    adapter = OlmoeAdapter(model)
    adapter.freeze_weights()
    runner = ExperimentRunner(adapter)
    device = next(model.parameters()).device
    basis = blind_basis(
        adapter.router_weight(args.layer), config["experiment"]["blind_rtol"]
    )
    records = []
    for prompt_index, prompt in enumerate(prompts):
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
        direction = project_to_blind(random_direction, basis)

        def scalar_function(delta: torch.Tensor) -> torch.Tensor:
            result = runner.teacher_forced(
                batch,
                injection_layer=args.layer,
                direction=delta,
                layers=[args.downstream_layer],
            )
            edited_logits = result.routes[args.downstream_layer].logits
            assert edited_logits is not None
            return margin_value(
                edited_logits[0, token], boundary.inside_expert, boundary.outside_expert
            )

        checks = finite_difference_checks(
            scalar_function,
            torch.zeros_like(direction),
            direction,
            gradient,
            config["experiment"]["finite_difference_epsilons"],
        )
        best = best_finite_difference(checks)
        records.append(
            {
                "prompt_id": prompt.prompt_id,
                "baseline_margin": boundary.value,
                "gradient_norm": float(torch.linalg.vector_norm(gradient.float()).item()),
                "checks": [check.to_dict() for check in checks],
                "best": best.to_dict(),
            }
        )
    relative_pass_rate = sum(r["best"]["relative_error"] <= 0.10 for r in records) / len(records)
    sign_pass_rate = sum(bool(r["best"]["sign_matches"]) for r in records) / len(records)
    report = {
        "schema_version": "margin-gradient-report-v1",
        "generated_at": utc_now(),
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "code_commit": git_commit(),
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "quantization": config["model"]["quantization"],
        "injection_layer": args.layer,
        "downstream_layer": args.downstream_layer,
        "relative_error_pass_rate": relative_pass_rate,
        "sign_pass_rate": sign_pass_rate,
        "passed": relative_pass_rate >= 0.90 and sign_pass_rate >= 0.95,
        "records": records,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

