"""Conservative static router-null spaces and numerical checks."""

from __future__ import annotations

import torch


@torch.no_grad()
def blind_basis(weight: torch.Tensor, rtol: float = 1e-6) -> torch.Tensor:
    """Return orthonormal columns spanning ``ker(weight)``.

    ``weight`` has shape ``[experts, hidden]`` and the result has shape
    ``[hidden, hidden-rank]``. SVD is deliberately evaluated in float32.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected [experts, hidden], got {tuple(weight.shape)}")
    if rtol <= 0:
        raise ValueError("rtol must be positive")
    original_device = weight.device
    original_dtype = weight.dtype
    work = weight.detach().to(device="cpu", dtype=torch.float32)
    _, singular_values, vh = torch.linalg.svd(work, full_matrices=True)
    largest = singular_values[0] if singular_values.numel() else torch.tensor(0.0)
    tolerance = float(rtol * max(work.shape) * largest.item())
    rank = int((singular_values > tolerance).sum().item())
    basis = vh[rank:].T.contiguous()
    return basis.to(device=original_device, dtype=original_dtype)


def project_to_blind(direction: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project vectors ``[..., hidden]`` onto an orthonormal blind basis."""
    if direction.shape[-1] != basis.shape[0]:
        raise ValueError("direction and basis hidden dimensions differ")
    if direction.device != basis.device or direction.dtype != basis.dtype:
        raise ValueError("direction and basis must share dtype and device")
    return (direction @ basis) @ basis.T


def project_to_visible(direction: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Return the component orthogonal to the conservative static blind space."""
    return direction - project_to_blind(direction, basis)


@torch.no_grad()
def router_null_error(weight: torch.Tensor, delta: torch.Tensor) -> dict[str, float]:
    """Return absolute and scale-normalized errors for ``W delta``."""
    residual = delta.float() @ weight.float().T
    absolute = float(torch.linalg.vector_norm(residual, dim=-1).max().item())
    weight_norm = torch.linalg.matrix_norm(weight.float(), ord=2)
    delta_norm = torch.linalg.vector_norm(delta.float(), dim=-1).max()
    denominator = max(float((weight_norm * delta_norm).item()), torch.finfo(torch.float32).tiny)
    return {"absolute": absolute, "relative": absolute / denominator}


@torch.no_grad()
def assert_router_blind(
    weight: torch.Tensor,
    delta: torch.Tensor,
    *,
    atol: float = 2e-5,
    rtol: float = 1e-5,
) -> None:
    """Raise when a candidate direction is not numerically router-null."""
    error = router_null_error(weight, delta)
    if error["absolute"] > atol and error["relative"] > rtol:
        raise AssertionError(
            "router-null check failed: "
            f"absolute={error['absolute']:.3e}, relative={error['relative']:.3e}"
        )


def root_mean_square(value: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Compute RMS in float32 and return it on the original device."""
    return value.float().square().mean(dim=dim, keepdim=keepdim).sqrt()


def rms_scaled_delta(
    direction: torch.Tensor,
    reference_state: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Scale a direction so its RMS is ``alpha`` times the reference RMS."""
    if direction.ndim != 1 or reference_state.shape[-1] != direction.shape[0]:
        raise ValueError("expected direction [hidden] and reference [..., hidden]")
    direction_rms = root_mean_square(direction)
    if float(direction_rms.item()) == 0.0:
        raise ValueError("cannot RMS-scale a zero direction")
    reference_rms = float(root_mean_square(reference_state).mean().item())
    direction_rms_value = float(direction_rms.item())
    scale = float(alpha) * reference_rms / direction_rms_value
    return direction * scale
