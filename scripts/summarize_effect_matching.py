#!/usr/bin/env python3
"""Summarize effect-matched route and external-loss comparisons from raw JSONL."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from drpi.records import utc_now
from drpi.statistics import paired_bootstrap_interval, paired_sign_permutation_pvalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/raw/interventions.jsonl")
    parser.add_argument("--out", default="results/summary/effect_matching.json")
    parser.add_argument("--reference", default="static_blind")
    parser.add_argument("--candidate", default="drpi")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def interpolate(rows: list[dict[str, object]], target_effect: float, metric: str) -> float | None:
    """Linearly interpolate a metric on the alpha sweep at a target effect."""
    ordered = sorted(rows, key=lambda row: float(row["target_effect"]))
    effects = [float(row["target_effect"]) for row in ordered]
    if target_effect < effects[0] or target_effect > effects[-1]:
        return None
    for left, right in zip(ordered, ordered[1:], strict=False):
        x0, x1 = float(left["target_effect"]), float(right["target_effect"])
        if x0 <= target_effect <= x1:
            y0, y1 = float(left[metric]), float(right[metric])
            if x1 == x0:
                return 0.5 * (y0 + y1)
            fraction = (target_effect - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    if target_effect == effects[-1]:
        return float(ordered[-1][metric])
    return None


def main() -> None:
    args = parse_args()
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["horizon"]) == args.horizon:
                grouped[(str(row["prompt_id"]), str(row["method"]))].append(row)

    metrics = ["topk_jaccard", "routing_js", "output_kl", "next_token_nll_delta"]
    differences: dict[str, list[float]] = defaultdict(list)
    quadrants = {
        "same_route_different_behavior": 0,
        "different_route_same_behavior": 0,
        "different_route_different_behavior": 0,
        "same_route_same_behavior": 0,
    }
    matched_prompts = []
    prompt_ids = sorted({prompt_id for prompt_id, _ in grouped})
    for prompt_id in prompt_ids:
        reference_rows = grouped.get((prompt_id, args.reference), [])
        candidate_rows = grouped.get((prompt_id, args.candidate), [])
        if not reference_rows or not candidate_rows:
            continue
        reference_effects = [float(row["target_effect"]) for row in reference_rows]
        candidate_effects = [float(row["target_effect"]) for row in candidate_rows]
        overlap_low = max(min(reference_effects), min(candidate_effects))
        overlap_high = min(max(reference_effects), max(candidate_effects))
        if overlap_low > overlap_high:
            continue
        target_effect = 0.5 * (overlap_low + overlap_high)
        matched = {"prompt_id": prompt_id, "target_effect": target_effect}
        valid = True
        for metric in metrics:
            reference_value = interpolate(reference_rows, target_effect, metric)
            candidate_value = interpolate(candidate_rows, target_effect, metric)
            if reference_value is None or candidate_value is None:
                valid = False
                break
            matched[f"{args.reference}_{metric}"] = reference_value
            matched[f"{args.candidate}_{metric}"] = candidate_value
            differences[metric].append(candidate_value - reference_value)
        if not valid:
            continue
        route_changed = abs(float(matched[f"{args.candidate}_topk_jaccard"]) - 1.0) > 1e-6
        behavior_changed = abs(target_effect) > 1e-4
        if route_changed and behavior_changed:
            quadrants["different_route_different_behavior"] += 1
        elif route_changed:
            quadrants["different_route_same_behavior"] += 1
        elif behavior_changed:
            quadrants["same_route_different_behavior"] += 1
        else:
            quadrants["same_route_same_behavior"] += 1
        matched_prompts.append(matched)

    if not matched_prompts:
        raise ValueError("no prompt has overlapping target-effect sweeps for both methods")
    summaries = {}
    for metric, values in differences.items():
        array = np.asarray(values)
        interval = paired_bootstrap_interval(
            array, resamples=args.bootstrap, seed=args.seed
        )
        summaries[metric] = {
            "candidate_minus_reference_mean": interval.estimate,
            "bootstrap_95_ci": [interval.lower, interval.upper],
            "paired_permutation_pvalue": paired_sign_permutation_pvalue(
                array, resamples=args.bootstrap, seed=args.seed
            ),
        }
    report = {
        "schema_version": "effect-matching-summary-v1",
        "generated_at": utc_now(),
        "reference": args.reference,
        "candidate": args.candidate,
        "horizon": args.horizon,
        "matched_prompt_count": len(matched_prompts),
        "quadrants": quadrants,
        "paired_summaries": summaries,
        "matched_prompts": matched_prompts,
        "interpretation_boundary": (
            "Route differences are internal mediator measurements; this report does not "
            "establish external utility without a preregistered external-loss endpoint."
        ),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

