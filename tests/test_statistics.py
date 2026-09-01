import numpy as np

from drpi.statistics import (
    binary_auroc,
    bootstrap_auroc_interval,
    brier_score,
    calibrated_probabilities,
    fit_logistic_calibrator,
    paired_bootstrap_interval,
    paired_sign_permutation_pvalue,
    reliability_curve,
)


def test_predictive_metrics_and_reliability():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert binary_auroc(scores, labels) == 1.0
    assert brier_score(scores, labels) < 0.05
    curve = reliability_curve(scores, labels, bins=2)
    assert sum(row["count"] for row in curve) == 4


def test_paired_inference_is_deterministic():
    differences = np.array([0.5, 0.7, 0.9, 1.1])
    interval = paired_bootstrap_interval(differences, resamples=500, seed=4)
    assert interval.lower > 0
    first = paired_sign_permutation_pvalue(differences, resamples=500, seed=4)
    second = paired_sign_permutation_pvalue(differences, resamples=500, seed=4)
    assert first == second


def test_calibration_and_bootstrap_auroc():
    scores = np.array([-2.0, -1.0, 1.0, 2.0])
    labels = np.array([0, 0, 1, 1])
    slope, intercept = fit_logistic_calibrator(scores, labels)
    probabilities = calibrated_probabilities(scores, slope, intercept)
    assert probabilities[0] < probabilities[-1]
    interval = bootstrap_auroc_interval(scores, labels, resamples=100, seed=2)
    assert interval.estimate == 1.0
