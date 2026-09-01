"""Direction-level baselines with matched dimensions and budgets."""

from __future__ import annotations

from collections.abc import Callable

import torch


@torch.no_grad()
def random_subspace_projection(
    direction: torch.Tensor,
    retained_dimension: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Project onto a seeded random subspace of a requested dimension."""
    if direction.ndim != 1:
        raise ValueError("direction must have shape [hidden]")
    if not 0 <= retained_dimension <= direction.numel():
        raise ValueError("retained_dimension outside valid range")
    if retained_dimension == 0:
        return torch.zeros_like(direction)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(
        direction.numel(), retained_dimension, generator=generator, dtype=torch.float32
    )
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    basis = basis.to(device=direction.device, dtype=direction.dtype)
    return (direction @ basis) @ basis.T


@torch.no_grad()
def random_blind_subspace_direction(
    direction: torch.Tensor,
    blind_basis: torch.Tensor,
    retained_dimension: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Matched random subspace baseline restricted to static blind coordinates."""
    if blind_basis.ndim != 2 or blind_basis.shape[0] != direction.numel():
        raise ValueError("blind basis and direction are incompatible")
    coordinates = direction @ blind_basis
    projected = random_subspace_projection(coordinates, retained_dimension, seed=seed)
    return projected @ blind_basis.T


def optimize_direction(
    initial: torch.Tensor,
    objective: Callable[[torch.Tensor], torch.Tensor],
    *,
    steps: int,
    learning_rate: float,
    l2_weight: float = 0.0,
) -> torch.Tensor:
    """Equal-budget generic optimizer used by target/KL/retain-loss baselines."""
    value = initial.detach().float().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([value], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(value)
        if loss.ndim != 0:
            raise ValueError("baseline objective must return a scalar loss")
        total = loss + l2_weight * value.square().mean()
        total.backward()
        optimizer.step()
    return value.detach().to(device=initial.device, dtype=initial.dtype)
