import torch

from drpi.metrics import jensen_shannon_divergence, topk_metrics


def test_set_and_order_sensitive_metrics_are_distinct():
    baseline = torch.tensor([[[1, 2]]])
    reordered = torch.tensor([[[2, 1]]])
    metrics = topk_metrics(baseline, reordered)
    assert metrics["topk_set_exact"].item()
    assert not metrics["topk_order_exact"].item()
    assert metrics["topk_jaccard"].item() == 1.0
    assert metrics["topk_ordered_hamming"].item() == 1.0


def test_js_is_zero_for_equal_distributions_and_finite_with_zeros():
    p = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.allclose(jensen_shannon_divergence(p, p), torch.zeros(1), atol=1e-7)
    assert torch.isfinite(jensen_shannon_divergence(p, torch.tensor([[0.0, 1.0, 0.0]]))).all()

