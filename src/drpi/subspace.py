"""Low-rank route-sensitive subspaces and hard-projection DRPI."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SubspaceSummary:
    requested_rank: int
    usable_rank: int
    singular_values: torch.Tensor


@torch.no_grad()
def dangerous_basis(
    projected_gradients: torch.Tensor,
    rank: int,
    *,
    atol: float = 1e-8,
) -> tuple[torch.Tensor, SubspaceSummary]:
    """Find the dominant row space of gradients in blind coordinates."""
    if projected_gradients.ndim != 2:
        raise ValueError("projected_gradients must have shape [boundaries, blind_dim]")
    if rank < 0:
        raise ValueError("rank cannot be negative")
    blind_dim = projected_gradients.shape[1]
    if projected_gradients.shape[0] == 0 or rank == 0:
        empty = projected_gradients.new_empty((blind_dim, 0))
        return empty, SubspaceSummary(rank, 0, projected_gradients.new_empty(0))
    work = projected_gradients.detach().to(device="cpu", dtype=torch.float32)
    _, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
    usable = min(rank, int((singular_values > atol).sum().item()))
    basis = vh[:usable].T.contiguous().to(
        device=projected_gradients.device, dtype=projected_gradients.dtype
    )
    return basis, SubspaceSummary(
        requested_rank=rank,
        usable_rank=usable,
        singular_values=singular_values.to(projected_gradients.device),
    )


@torch.no_grad()
def drpi_direction(
    target_direction: torch.Tensor,
    blind: torch.Tensor,
    dangerous: torch.Tensor,
) -> torch.Tensor:
    """Project to static blind coordinates and remove route-sensitive components."""
    if target_direction.ndim != 1:
        raise ValueError("target_direction must have shape [hidden]")
    if blind.ndim != 2 or blind.shape[0] != target_direction.shape[0]:
        raise ValueError("blind basis has incompatible shape")
    if dangerous.ndim != 2 or dangerous.shape[0] != blind.shape[1]:
        raise ValueError("dangerous basis must be expressed in blind coordinates")
    coordinates = target_direction @ blind
    if dangerous.numel():
        coordinates = coordinates - (coordinates @ dangerous) @ dangerous.T
    return coordinates @ blind.T


@torch.no_grad()
def target_retention(target: torch.Tensor, edited: torch.Tensor) -> float:
    """Return the fraction of target L2 norm retained by a projection."""
    denominator = float(torch.linalg.vector_norm(target.float()).item())
    if denominator == 0.0:
        return 0.0
    return float(torch.linalg.vector_norm(edited.float()).item()) / denominator


@torch.no_grad()
def principal_angle_cosines(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Cosines of principal angles between orthonormal column bases."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError("bases must have shapes [dimension, rank]")
    if first.shape[1] == 0 or second.shape[1] == 0:
        return first.new_empty(0)
    return torch.linalg.svdvals(first.float().T @ second.float()).clamp(0.0, 1.0)

