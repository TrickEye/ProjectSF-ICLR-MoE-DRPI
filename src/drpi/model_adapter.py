"""Model-specific boundary for MoE structure and routing behavior."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn


@dataclass(frozen=True)
class LayerDescription:
    """Auditable structural facts for one MoE layer."""

    layer: int
    shared_state_path: str
    router_path: str
    router_weight_shape: tuple[int, ...]
    router_has_bias: bool
    top_k: int
    num_experts: int
    norm_topk_prob: bool
    shared_experts: bool
    router_weight_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@runtime_checkable
class MoEAdapter(Protocol):
    """Architecture-independent interface used by DRPI modules."""

    model: nn.Module

    def num_layers(self) -> int: ...

    def router(self, layer: int) -> nn.Module: ...

    def router_weight(self, layer: int) -> torch.Tensor: ...

    def shared_state_module(self, layer: int) -> nn.Module: ...

    def top_k(self, layer: int) -> int: ...

    def router_logits_from_input(self, layer: int, state: torch.Tensor) -> torch.Tensor: ...

    def describe_layer(self, layer: int) -> LayerDescription: ...


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor values together with shape and dtype."""
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class OlmoeAdapter:
    """Adapter for Hugging Face ``OlmoeForCausalLM``.

    This is the only module allowed to know that OLMoE stores sparse blocks at
    ``model.model.layers[*].mlp`` and routers at ``mlp.gate``.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        if model_type != "olmoe":
            raise TypeError(f"expected an OLMoE model, got model_type={model_type!r}")
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise TypeError("OLMoE model does not expose decoder layers as expected")

    def _layer(self, layer: int) -> nn.Module:
        if layer < 0 or layer >= self.num_layers():
            raise IndexError(f"layer {layer} outside [0, {self.num_layers()})")
        return self.model.model.layers[layer]

    def _moe(self, layer: int) -> nn.Module:
        module = self._layer(layer).mlp
        if not hasattr(module, "gate") or not hasattr(module, "experts"):
            raise TypeError(f"layer {layer} is not a supported sparse MoE block")
        return module

    def num_layers(self) -> int:
        return len(self.model.model.layers)

    def router(self, layer: int) -> nn.Module:
        return self._moe(layer).gate

    def router_weight(self, layer: int) -> torch.Tensor:
        return self.router(layer).weight

    def shared_state_module(self, layer: int) -> nn.Module:
        # OlmoeSparseMoeBlock.forward feeds the same hidden_states to gate and experts.
        return self._moe(layer)

    def top_k(self, layer: int) -> int:
        return int(self._moe(layer).top_k)

    def router_logits_from_input(self, layer: int, state: torch.Tensor) -> torch.Tensor:
        hidden = state.shape[-1]
        flat_logits = self.router(layer)(state.reshape(-1, hidden))
        return flat_logits.reshape(*state.shape[:-1], flat_logits.shape[-1])

    def shared_state_path(self, layer: int) -> str:
        return f"model.layers.{layer}.mlp"

    def router_path(self, layer: int) -> str:
        return f"model.layers.{layer}.mlp.gate"

    def describe_layer(self, layer: int) -> LayerDescription:
        moe = self._moe(layer)
        router = self.router(layer)
        return LayerDescription(
            layer=layer,
            shared_state_path=self.shared_state_path(layer),
            router_path=self.router_path(layer),
            router_weight_shape=tuple(router.weight.shape),
            router_has_bias=router.bias is not None,
            top_k=int(moe.top_k),
            num_experts=int(moe.num_experts),
            norm_topk_prob=bool(moe.norm_topk_prob),
            shared_experts=False,
            router_weight_sha256=tensor_sha256(router.weight),
        )

    def freeze_weights(self) -> None:
        """Freeze all model parameters while preserving input autograd."""
        self.model.requires_grad_(False)
        self.model.eval()

