import torch

from drpi.router_capture import RouterCapture

from conftest import ToyAdapter, make_batch


def test_t2_capture_matches_manual_router_forward_and_preserves_graph():
    adapter = ToyAdapter()
    batch = make_batch()
    with RouterCapture(adapter, layers=[1]) as capture, torch.enable_grad():
        adapter.model(**batch)
        record = capture.records[1]
        assert record.shared_state is not None
        assert record.logits is not None
        manual = adapter.router_logits_from_input(1, record.shared_state)
        assert torch.allclose(record.logits, manual)
        assert record.logits.grad_fn is not None
    assert len(adapter.router(1)._forward_hooks) == 0
    assert len(adapter.shared_state_module(1)._forward_pre_hooks) == 0

