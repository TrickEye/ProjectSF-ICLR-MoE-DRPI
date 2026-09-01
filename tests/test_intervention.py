import pytest
import torch

from drpi.interventions import (
    Injection,
    RouterLogitBiasInjector,
    SharedStateInjector,
    final_valid_positions,
)
from drpi.runner import ExperimentRunner
from drpi.static_space import blind_basis, project_to_blind

from conftest import ToyAdapter, make_batch


def test_last_valid_position_handles_left_and_right_padding():
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    assert torch.equal(final_valid_positions(mask), torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError):
        final_valid_positions(torch.zeros(1, 3, dtype=torch.long))


def test_t1_router_null_and_t3_locality_and_cleanup():
    adapter = ToyAdapter()
    runner = ExperimentRunner(adapter)
    batch = make_batch()
    baseline = runner.teacher_forced(batch)
    basis = blind_basis(adapter.router_weight(0))
    direction = project_to_blind(torch.arange(8, dtype=torch.float32), basis)
    edited = runner.teacher_forced(
        batch, injection_layer=0, direction=direction, alpha=0.5
    )
    base_route = baseline.routes[0]
    edit_route = edited.routes[0]
    assert torch.allclose(base_route.logits, edit_route.logits, atol=1e-5)
    assert torch.equal(base_route.topk_indices, edit_route.topk_indices)
    assert torch.allclose(
        base_route.logits[:, :-1], edit_route.logits[:, :-1], atol=1e-6
    )
    restored = runner.teacher_forced(batch)
    assert torch.allclose(baseline.output_logits, restored.output_logits)


def test_alpha_zero_reproduces_baseline():
    adapter = ToyAdapter()
    runner = ExperimentRunner(adapter)
    batch = make_batch()
    baseline = runner.teacher_forced(batch)
    edited = runner.teacher_forced(
        batch, injection_layer=0, direction=torch.randn(8), alpha=0.0
    )
    assert torch.equal(baseline.output_logits, edited.output_logits)


def test_injector_removes_hook_after_exception():
    adapter = ToyAdapter()
    module = adapter.shared_state_module(0)
    with pytest.raises(RuntimeError):
        with SharedStateInjector(module, Injection(torch.ones(8))):
            raise RuntimeError("expected")
    assert len(module._forward_pre_hooks) == 0


def test_router_bias_changes_only_target_flattened_token():
    router = torch.nn.Linear(4, 3, bias=False)
    state = torch.randn(2 * 3, 4)
    baseline = router(state)
    with RouterLogitBiasInjector(
        router,
        torch.ones(3),
        token_positions=torch.tensor([1, 2]),
        sequence_length=3,
    ):
        edited = router(state)
    changed_rows = (edited - baseline).abs().sum(dim=-1).nonzero().flatten()
    assert torch.equal(changed_rows, torch.tensor([1, 5]))
