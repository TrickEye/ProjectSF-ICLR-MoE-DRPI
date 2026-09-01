import pytest
import torch

from drpi.static_space import (
    assert_router_blind,
    blind_basis,
    project_to_blind,
    project_to_visible,
    rms_scaled_delta,
    root_mean_square,
)


def test_blind_basis_dimensions_orthogonality_dtype_and_device():
    torch.manual_seed(1)
    weight = torch.randn(3, 7, dtype=torch.float64)
    basis = blind_basis(weight)
    assert basis.shape == (7, 4)
    assert basis.dtype == weight.dtype
    assert basis.device == weight.device
    assert torch.allclose(basis.T @ basis, torch.eye(4, dtype=weight.dtype), atol=2e-6)
    assert_router_blind(weight, basis.T, atol=2e-5, rtol=2e-5)


def test_projector_and_zero_direction_edges():
    weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    basis = blind_basis(weight)
    direction = torch.tensor([3.0, 4.0, 5.0])
    projected = project_to_blind(direction, basis)
    assert torch.allclose(projected, torch.tensor([0.0, 0.0, 5.0]))
    visible = project_to_visible(direction, basis)
    assert torch.allclose(visible, torch.tensor([3.0, 4.0, 0.0]))
    with pytest.raises(ValueError, match="zero direction"):
        rms_scaled_delta(torch.zeros(3), torch.ones(3), 0.1)


def test_rms_strength_definition():
    delta = rms_scaled_delta(torch.tensor([1.0, -1.0]), torch.tensor([2.0, -2.0]), 0.25)
    assert torch.allclose(root_mean_square(delta), 0.25 * root_mean_square(torch.tensor([2.0, -2.0])))
