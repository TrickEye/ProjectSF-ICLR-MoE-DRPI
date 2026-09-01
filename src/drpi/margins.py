"""Critical top-k routing boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CriticalMargin:
    """Weakest selected versus strongest unselected expert boundary."""

    inside_expert: int
    outside_expert: int
    value: float


def weakest_inside_best_outside(logits: torch.Tensor, top_k: int) -> CriticalMargin:
    """Extract the active top-k set boundary from one expert-logit vector."""
    if logits.ndim != 1:
        raise ValueError("logits must have shape [experts]")
    if not 0 < top_k < logits.numel():
        raise ValueError("top_k must be between 1 and num_experts - 1")
    order = torch.argsort(logits, descending=True, stable=True)
    inside = int(order[top_k - 1].item())
    outside = int(order[top_k].item())
    margin = float((logits[inside] - logits[outside]).item())
    return CriticalMargin(inside_expert=inside, outside_expert=outside, value=margin)


def margin_value(logits: torch.Tensor, inside_expert: int, outside_expert: int) -> torch.Tensor:
    """Return a differentiable selected-minus-unselected logit margin."""
    return logits[..., inside_expert] - logits[..., outside_expert]

