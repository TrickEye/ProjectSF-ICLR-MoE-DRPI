"""Dependency-light predictive and paired statistical summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via pairwise ranking with half credit for ties."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positive = scores[labels]
    negative = scores[~labels]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("AUROC requires positive and negative examples")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must align")
    return float(np.mean((probabilities - labels) ** 2))


def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> list[dict[str, float | int]]:
    """Equal-width reliability bins including empty-bin omission."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    result = []
    for index in range(bins):
        mask = assignments == index
        if np.any(mask):
            result.append(
                {
                    "bin": index,
                    "count": int(mask.sum()),
                    "mean_prediction": float(probabilities[mask].mean()),
                    "event_rate": float(labels[mask].mean()),
                }
            )
    return result


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> BootstrapInterval:
    """Percentile bootstrap CI for a paired mean difference."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    estimates = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(values.mean()),
        lower=float(np.quantile(estimates, tail)),
        upper=float(np.quantile(estimates, 1.0 - tail)),
    )


def paired_sign_permutation_pvalue(
    differences: np.ndarray, *, resamples: int = 10_000, seed: int = 1729
) -> float:
    """Two-sided paired random-sign permutation test."""
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    signs = rng.choice((-1.0, 1.0), size=(resamples, values.size))
    permuted = np.abs((signs * values).mean(axis=1))
    return float((1 + np.sum(permuted >= observed)) / (resamples + 1))


def bootstrap_auroc_interval(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> BootstrapInterval:
    """Stratified bootstrap interval for binary AUROC."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positive = scores[labels]
    negative = scores[~labels]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("AUROC requires both classes")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_positive = rng.choice(positive, size=positive.size, replace=True)
        sampled_negative = rng.choice(negative, size=negative.size, replace=True)
        sampled_scores = np.concatenate((sampled_positive, sampled_negative))
        sampled_labels = np.concatenate(
            (np.ones(positive.size, dtype=bool), np.zeros(negative.size, dtype=bool))
        )
        estimates[index] = binary_auroc(sampled_scores, sampled_labels)
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=binary_auroc(scores, labels),
        lower=float(np.quantile(estimates, tail)),
        upper=float(np.quantile(estimates, 1.0 - tail)),
    )


def fit_logistic_calibrator(
    scores: np.ndarray, labels: np.ndarray, *, steps: int = 100, l2: float = 1e-4
) -> tuple[float, float]:
    """Fit one-dimensional Platt scaling by damped Newton updates."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("scores and labels must be aligned vectors")
    slope, intercept = 0.0, float(np.log((y.mean() + 1e-3) / (1.0 - y.mean() + 1e-3)))
    design = np.column_stack((x, np.ones_like(x)))
    parameters = np.array([slope, intercept])
    for _ in range(steps):
        logits = np.clip(design @ parameters, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - y) + l2 * parameters
        weights = probabilities * (1.0 - probabilities)
        hessian = design.T @ (weights[:, None] * design) + l2 * np.eye(2)
        update = np.linalg.solve(hessian, gradient)
        parameters -= update
        if np.linalg.norm(update) < 1e-8:
            break
    return float(parameters[0]), float(parameters[1])


def calibrated_probabilities(scores: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    logits = np.clip(slope * np.asarray(scores, dtype=float) + intercept, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))
