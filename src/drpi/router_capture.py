"""Capture shared router inputs and route decisions without breaking autograd."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from drpi.model_adapter import MoEAdapter


@dataclass
class RouteRecord:
    """Tensors captured for one layer and one forward pass."""

    shared_state: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    gate_probabilities: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None
    topk_values: torch.Tensor | None = None


class RouterCapture:
    """Context-managed capture for selected adapter layers."""

    def __init__(self, adapter: MoEAdapter, layers: list[int] | None = None):
        self.adapter = adapter
        self.layers = list(range(adapter.num_layers())) if layers is None else list(layers)
        self.records = {layer: RouteRecord() for layer in self.layers}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _shared_pre_hook(self, layer: int):
        def hook(_module, args):
            state = args[0]
            if state.ndim != 3:
                raise ValueError(f"expected [batch, seq, hidden], got {tuple(state.shape)}")
            self.records[layer].shared_state = state

        return hook

    def _router_hook(self, layer: int):
        def hook(_module, _args, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise ValueError("expected flattened router logits [batch*seq, experts]")
            state = self.records[layer].shared_state
            if state is None:
                raise RuntimeError("router ran before its shared input was captured")
            batch, sequence, _ = state.shape
            if output.shape[0] != batch * sequence:
                raise ValueError("router output cannot be aligned to captured tokens")
            logits = output.reshape(batch, sequence, output.shape[-1])
            probabilities = torch.softmax(logits.float(), dim=-1)
            values, indices = probabilities.topk(self.adapter.top_k(layer), dim=-1)
            record = self.records[layer]
            record.logits = logits
            record.gate_probabilities = probabilities
            record.topk_values = values
            record.topk_indices = indices

        return hook

    def __enter__(self) -> RouterCapture:
        if self._handles:
            raise RuntimeError("capture is already active")
        for layer in self.layers:
            self._handles.append(
                self.adapter.shared_state_module(layer).register_forward_pre_hook(
                    self._shared_pre_hook(layer)
                )
            )
            self._handles.append(
                self.adapter.router(layer).register_forward_hook(self._router_hook(layer))
            )
        return self

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.remove()

    def detached(self) -> dict[int, RouteRecord]:
        """Return CPU records for serialization after gradients are no longer needed."""
        result: dict[int, RouteRecord] = {}
        for layer, record in self.records.items():
            result[layer] = RouteRecord(
                **{
                    name: value.detach().cpu() if isinstance(value, torch.Tensor) else None
                    for name, value in vars(record).items()
                }
            )
        return result

