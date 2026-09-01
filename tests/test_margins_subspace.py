import torch

from drpi.margins import weakest_inside_best_outside
from drpi.subspace import dangerous_basis, drpi_direction, principal_angle_cosines


def test_critical_margin_uses_weakest_inside_and_best_outside():
    margin = weakest_inside_best_outside(torch.tensor([0.1, 1.0, 0.8, 0.7]), top_k=2)
    assert margin.inside_expert == 2
    assert margin.outside_expert == 3
    assert abs(margin.value - 0.1) < 1e-6


def test_empty_rank_truncation_and_drpi_geometry():
    gradients = torch.zeros(0, 3)
    dangerous, summary = dangerous_basis(gradients, rank=2)
    assert dangerous.shape == (3, 0)
    assert summary.usable_rank == 0
    blind = torch.eye(4)[:, :3]
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    direction = drpi_direction(target, blind, dangerous)
    assert torch.equal(direction, torch.tensor([1.0, 2.0, 3.0, 0.0]))


def test_dangerous_basis_removes_dominant_coordinate():
    gradients = torch.tensor([[5.0, 0.0], [4.0, 0.1]])
    dangerous, summary = dangerous_basis(gradients, rank=1)
    assert summary.usable_rank == 1
    blind = torch.eye(2)
    direction = drpi_direction(torch.tensor([2.0, 3.0]), blind, dangerous)
    assert abs(float(direction @ dangerous[:, 0])) < 1e-5
    assert principal_angle_cosines(dangerous, dangerous).item() > 0.999

