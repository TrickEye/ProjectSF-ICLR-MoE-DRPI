import torch

from drpi.gradients import best_finite_difference, finite_difference_checks


def test_t4_centered_finite_difference_matches_gradient():
    point = torch.tensor([0.2, -0.4, 0.7], requires_grad=True)

    def function(value):
        return (value.sin() + 0.3 * value.square()).sum()

    (gradient,) = torch.autograd.grad(function(point), point)
    direction = torch.tensor([0.1, 0.2, -0.3])
    checks = finite_difference_checks(
        function, point.detach(), direction, gradient, [1e-4, 1e-3, 1e-2]
    )
    best = best_finite_difference(checks)
    assert best.relative_error < 1e-3
    assert best.sign_matches

