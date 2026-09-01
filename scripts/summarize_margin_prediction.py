#!/usr/bin/env python3
"""Fit calibration-only Platt scaling and report held-out switch prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drpi.records import utc_now
from drpi.statistics import (
    bootstrap_auroc_interval,
    brier_score,
    calibrated_probabilities,
    fit_logistic_calibrator,
    reliability_curve,
)


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return (
        np.asarray([row["switch_score"] for row in rows], dtype=float),
        np.asarray([row["actual_switch"] for row in rows], dtype=bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", default="results/summary/margin_prediction.json")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    calibration_scores, calibration_labels = load(args.calibration)
    test_scores, test_labels = load(args.test)
    slope, intercept = fit_logistic_calibrator(calibration_scores, calibration_labels)
    probabilities = calibrated_probabilities(test_scores, slope, intercept)
    interval = bootstrap_auroc_interval(
        test_scores, test_labels, resamples=args.bootstrap, seed=args.seed
    )
    prevalence = float(test_labels.mean())
    report = {
        "schema_version": "margin-prediction-summary-v1",
        "generated_at": utc_now(),
        "calibrator": {"slope": slope, "intercept": intercept},
        "test_count": int(test_labels.size),
        "test_prevalence": prevalence,
        "auroc": interval.estimate,
        "auroc_bootstrap_95_ci": [interval.lower, interval.upper],
        "brier": brier_score(probabilities, test_labels),
        "constant_prevalence_brier": brier_score(
            np.full(test_labels.shape, prevalence), test_labels
        ),
        "reliability_curve": reliability_curve(probabilities, test_labels),
    }
    report["go"] = (
        report["auroc_bootstrap_95_ci"][0] > 0.5
        and report["brier"] < report["constant_prevalence_brier"]
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
