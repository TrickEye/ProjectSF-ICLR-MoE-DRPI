"""Routing and external-behavior metrics with aligned-token semantics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def centered_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.float() - logits.float().mean(dim=-1, keepdim=True)


def centered_logit_metrics(baseline: torch.Tensor, edited: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-token centered L2 drift and cosine similarity."""
    base = centered_logits(baseline)
    edit = centered_logits(edited)
    return {
        "centered_l2": torch.linalg.vector_norm(edit - base, dim=-1),
        "centered_cosine": functional.cosine_similarity(base, edit, dim=-1, eps=1e-12),
    }


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-token JS divergence in nats for normalized expert distributions."""
    p = p.float().clamp_min(torch.finfo(torch.float32).tiny)
    q = q.float().clamp_min(torch.finfo(torch.float32).tiny)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.log() - midpoint.log())).sum(dim=-1)
        + (q * (q.log() - midpoint.log())).sum(dim=-1)
    )


def topk_metrics(baseline: torch.Tensor, edited: torch.Tensor) -> dict[str, torch.Tensor]:
    """Separate set-based and order-sensitive top-k metrics."""
    if baseline.shape != edited.shape or baseline.ndim < 1:
        raise ValueError("top-k tensors must have equal [..., k] shapes")
    ordered_equal = baseline.eq(edited)
    order_exact = ordered_equal.all(dim=-1)
    ordered_hamming = (~ordered_equal).float().mean(dim=-1)

    intersections = baseline.unsqueeze(-1).eq(edited.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    k = baseline.shape[-1]
    unions = 2 * k - intersections
    jaccard = intersections.float() / unions.float()
    set_exact = intersections.eq(k)
    return {
        "topk_set_exact": set_exact,
        "topk_order_exact": order_exact,
        "topk_jaccard": jaccard,
        "topk_ordered_hamming": ordered_hamming,
    }


def route_metrics(
    baseline_logits: torch.Tensor,
    edited_logits: torch.Tensor,
    baseline_topk: torch.Tensor,
    edited_topk: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute continuous and discrete aligned-token route metrics."""
    base_prob = torch.softmax(baseline_logits.float(), dim=-1)
    edit_prob = torch.softmax(edited_logits.float(), dim=-1)
    result = centered_logit_metrics(baseline_logits, edited_logits)
    result["routing_js"] = jensen_shannon_divergence(base_prob, edit_prob)
    result.update(topk_metrics(baseline_topk, edited_topk))
    return result


def first_switch_layer(
    baseline_by_layer: dict[int, torch.Tensor], edited_by_layer: dict[int, torch.Tensor]
) -> int | None:
    """Return the first layer with a changed expert set, not merely changed order."""
    for layer in sorted(set(baseline_by_layer) & set(edited_by_layer)):
        exact = topk_metrics(baseline_by_layer[layer], edited_by_layer[layer])["topk_set_exact"]
        if not bool(exact.all().item()):
            return layer
    return None


def output_kl(baseline_logits: torch.Tensor, edited_logits: torch.Tensor) -> torch.Tensor:
    """Per-token KL(baseline || edited)."""
    baseline_log_prob = functional.log_softmax(baseline_logits.float(), dim=-1)
    edited_log_prob = functional.log_softmax(edited_logits.float(), dim=-1)
    baseline_prob = baseline_log_prob.exp()
    return (baseline_prob * (baseline_log_prob - edited_log_prob)).sum(dim=-1)


def next_token_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-position teacher-forced next-token negative log likelihood."""
    if logits.shape[:-1] != input_ids.shape:
        raise ValueError("logits and input_ids leading shapes differ")
    shifted_logits = logits[:, :-1].float().transpose(1, 2)
    shifted_labels = input_ids[:, 1:]
    return functional.cross_entropy(shifted_logits, shifted_labels, reduction="none")


def target_log_probability(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Gather log probabilities for explicit target tokens."""
    if token_ids.shape != logits.shape[:-1]:
        raise ValueError("token_ids must align with logits leading dimensions")
    return functional.log_softmax(logits.float(), dim=-1).gather(
        -1, token_ids.unsqueeze(-1)
    ).squeeze(-1)


def finite_values(values: torch.Tensor) -> bool:
    return bool(torch.isfinite(values).all().item())


def relative_change(baseline: float, edited: float) -> float:
    denominator = max(abs(baseline), math.ulp(1.0))
    return (edited - baseline) / denominator

