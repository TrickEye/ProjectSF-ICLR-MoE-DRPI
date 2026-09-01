#!/usr/bin/env python3
"""Stage 0: inspect OLMoE hooks, routing behavior, cache, and backend."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from drpi.config import load_config
from drpi.gradients import InjectionStateLeaf, margin_gradient
from drpi.loading import check_weight_cache, load_model_and_tokenizer
from drpi.margins import weakest_inside_best_outside
from drpi.model_adapter import OlmoeAdapter
from drpi.records import git_commit, utc_now
from drpi.router_capture import RouterCapture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/olmoe_pilot.yaml")
    parser.add_argument("--out", default="results/summary/router_hook_report.json")
    parser.add_argument("--prompt", default="The capital of France is")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    report: dict[str, object] = {
        "schema_version": "router-hook-report-v1",
        "generated_at": utc_now(),
        "model_id": model_config["id"],
        "model_revision": model_config["revision"],
        "requested_device": model_config["device"],
        "requested_dtype": model_config["dtype"],
        "quantization": model_config["quantization"],
        "code_commit": git_commit(),
        "status": "started",
    }
    cache = check_weight_cache(model_config["id"], model_config["revision"])
    report["cache"] = asdict(cache)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if bool(model_config.get("local_files_only", True)) and not cache.complete:
        report["status"] = "blocked_incomplete_cache"
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    try:
        started = time.perf_counter()
        model, tokenizer = load_model_and_tokenizer(model_config)
        load_seconds = time.perf_counter() - started
        adapter = OlmoeAdapter(model)
        adapter.freeze_weights()
        device = next(model.parameters()).device
        encoded = tokenizer(
            args.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config["runtime"]["max_length"],
        )
        batch = {name: value.to(device) for name, value in encoded.items()}

        started = time.perf_counter()
        with torch.no_grad(), RouterCapture(adapter, layers=list(range(adapter.num_layers()))) as cap:
            output = model(**batch, use_cache=False, return_dict=True)
            no_grad_records = cap.detached()
        no_grad_seconds = time.perf_counter() - started

        layer = 0
        state = no_grad_records[layer].shared_state
        captured_logits = no_grad_records[layer].logits
        assert state is not None and captured_logits is not None
        manual = adapter.router_logits_from_input(layer, state.to(device)).detach().cpu()
        manual_error = float((manual - captured_logits).abs().max().item())

        gradient_report: dict[str, object]
        if adapter.num_layers() > 1:
            started = time.perf_counter()
            with (
                InjectionStateLeaf(adapter.shared_state_module(0)) as leaf,
                RouterCapture(adapter, layers=[0, 1]) as capture,
                torch.enable_grad(),
            ):
                model(**batch, use_cache=False, return_dict=True)
                downstream = capture.records[1].logits
                assert leaf.state is not None and downstream is not None
                token = downstream.shape[1] - 1
                boundary = weakest_inside_best_outside(
                    downstream[0, token].detach(), adapter.top_k(1)
                )
                gradient = margin_gradient(
                    leaf.state,
                    downstream,
                    injection_token=token,
                    downstream_token=token,
                    inside_expert=boundary.inside_expert,
                    outside_expert=boundary.outside_expert,
                )
            gradient_report = {
                "status": "passed",
                "seconds": time.perf_counter() - started,
                "finite": bool(torch.isfinite(gradient).all().item()),
                "norm": float(torch.linalg.vector_norm(gradient.float()).item()),
            }
        else:
            gradient_report = {"status": "skipped_single_layer"}

        report.update(
            {
                "status": "passed",
                "actual_device": str(device),
                "actual_dtype": str(next(model.parameters()).dtype),
                "load_seconds": load_seconds,
                "no_grad_forward_seconds": no_grad_seconds,
                "output_shape": list(output.logits.shape),
                "manual_router_max_abs_error": manual_error,
                "shared_state_verified": manual_error <= 2e-5,
                "gradient_smoke": gradient_report,
                "layers": [
                    adapter.describe_layer(index).to_dict()
                    for index in range(adapter.num_layers())
                ],
                "named_modules": [
                    {"name": name, "class": module.__class__.__name__}
                    for name, module in model.named_modules()
                ],
            }
        )
        if device.type == "mps":
            report["mps_current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
            report["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise

    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

