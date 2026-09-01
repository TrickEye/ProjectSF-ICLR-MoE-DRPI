"""Downstream routing-margin gradients and finite-difference validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import nn

from drpi.margins import margin_value


class InjectionStateLeaf:
    """Expose a shared-state input as an autograd leaf for local Jacobians.

    Model parameters may remain frozen. The hook intentionally severs gradients
    to earlier layers because DRPI only needs derivatives with respect to the
    state at the injection point.
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self.state: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, _module: nn.Module, args: tuple[object, ...]):
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError("shared-state module first argument must be a tensor")
        self.state = args[0].detach().clone().requires_grad_(True)
        return (self.state, *args[1:])

    def __enter__(self) -> InjectionStateLeaf:
        self._handle = self.module.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def margin_gradient(
    injection_state: torch.Tensor,
    downstream_logits: torch.Tensor,
    *,
    injection_token: int,
    downstream_token: int,
    inside_expert: int,
    outside_expert: int,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Differentiate one downstream margin with respect to one injected state.

    The first version deliberately requires batch size one to eliminate padding
    and broadcast ambiguity during scientific validation.
    """
    if injection_state.ndim != 3 or injection_state.shape[0] != 1:
        raise ValueError("margin gradients currently require [1, seq, hidden]")
    if downstream_logits.ndim != 3 or downstream_logits.shape[0] != 1:
        raise ValueError("downstream logits currently require [1, seq, experts]")
    margin = margin_value(
        downstream_logits[0, downstream_token], inside_expert, outside_expert
    )
    (gradient,) = torch.autograd.grad(margin, injection_state, retain_graph=retain_graph)
    return gradient[0, injection_token]


@dataclass(frozen=True)
class FiniteDifferenceCheck:
    epsilon: float
    finite_difference: float
    gradient_inner_product: float
    relative_error: float
    sign_matches: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def finite_difference_checks(
    scalar_function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    direction: torch.Tensor,
    gradient: torch.Tensor,
    epsilons: list[float],
) -> list[FiniteDifferenceCheck]:
    """Compare centered finite differences with ``gradient dot direction``."""
    if point.shape != direction.shape or point.shape != gradient.shape:
        raise ValueError("point, direction, and gradient must have equal shapes")
    norm = torch.linalg.vector_norm(direction.float())
    if float(norm.item()) == 0.0:
        raise ValueError("finite-difference direction cannot be zero")
    unit_direction = direction / norm.to(direction.dtype)
    predicted = float((gradient.float() * unit_direction.float()).sum().item())
    checks: list[FiniteDifferenceCheck] = []
    for epsilon in epsilons:
        if epsilon <= 0:
            raise ValueError("finite-difference epsilons must be positive")
        plus = float(scalar_function(point + epsilon * unit_direction).detach().float().item())
        minus = float(scalar_function(point - epsilon * unit_direction).detach().float().item())
        observed = (plus - minus) / (2.0 * epsilon)
        denominator = max(abs(observed), abs(predicted), torch.finfo(torch.float32).eps)
        checks.append(
            FiniteDifferenceCheck(
                epsilon=float(epsilon),
                finite_difference=observed,
                gradient_inner_product=predicted,
                relative_error=abs(observed - predicted) / denominator,
                sign_matches=(observed == 0.0 and predicted == 0.0)
                or (observed * predicted > 0.0),
            )
        )
    return checks


def best_finite_difference(checks: list[FiniteDifferenceCheck]) -> FiniteDifferenceCheck:
    if not checks:
        raise ValueError("at least one finite-difference check is required")
    return min(checks, key=lambda check: check.relative_error)
