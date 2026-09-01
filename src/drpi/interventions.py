"""Lifecycle-safe shared-state and router-output interventions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def final_valid_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Resolve the final non-padding token for left- or right-padded masks."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, seq]")
    valid = attention_mask.to(dtype=torch.bool)
    if not torch.all(valid.any(dim=1)):
        raise ValueError("every sample must contain at least one valid token")
    indices = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    return indices.masked_fill(~valid, -1).max(dim=1).values


@dataclass(frozen=True)
class Injection:
    """One shared-state intervention specification."""

    direction: torch.Tensor
    alpha: float = 1.0
    token_positions: int | torch.Tensor = -1


class SharedStateInjector:
    """Clone and edit the tensor received by both router and experts."""

    def __init__(self, module: nn.Module, injection: Injection):
        if injection.direction.ndim != 1:
            raise ValueError("direction must have shape [hidden]")
        self.module = module
        self.injection = injection
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, _module: nn.Module, args: tuple[object, ...]):
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError("shared-state module first argument must be a tensor")
        state = args[0]
        if state.ndim != 3:
            raise ValueError(f"expected [batch, seq, hidden], got {tuple(state.shape)}")
        direction = self.injection.direction.to(device=state.device, dtype=state.dtype)
        if direction.shape[0] != state.shape[-1]:
            raise ValueError("direction hidden dimension does not match shared state")

        positions = self.injection.token_positions
        if isinstance(positions, int):
            positions = torch.full(
                (state.shape[0],), positions, dtype=torch.long, device=state.device
            )
        else:
            positions = positions.to(device=state.device, dtype=torch.long)
        if positions.shape != (state.shape[0],):
            raise ValueError("token_positions must contain one index per batch item")
        positions = positions.remainder(state.shape[1])

        edited = state.clone()
        rows = torch.arange(state.shape[0], device=state.device)
        edited[rows, positions] = (
            edited[rows, positions] + float(self.injection.alpha) * direction
        )
        return (edited, *args[1:])

    def __enter__(self) -> SharedStateInjector:
        if self._handle is not None:
            raise RuntimeError("injector is already active")
        self._handle = self.module.register_forward_pre_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.remove()


class RouterLogitBiasInjector:
    """Router-only baseline; this is not a shared-state intervention."""

    def __init__(
        self,
        router: nn.Module,
        bias: torch.Tensor,
        *,
        token_positions: torch.Tensor,
        sequence_length: int,
    ):
        if bias.ndim != 1:
            raise ValueError("router bias must have shape [experts]")
        self.router = router
        self.bias = bias
        self.token_positions = token_positions
        self.sequence_length = sequence_length
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, _module: nn.Module, _args: tuple[object, ...], output: torch.Tensor):
        if output.ndim != 2 or output.shape[0] % self.sequence_length:
            raise ValueError("router output does not match configured sequence length")
        batch_size = output.shape[0] // self.sequence_length
        positions = self.token_positions.to(device=output.device, dtype=torch.long)
        if positions.shape != (batch_size,):
            raise ValueError("router token positions do not match flattened batch")
        rows = torch.arange(batch_size, device=output.device) * self.sequence_length
        rows = rows + positions.remainder(self.sequence_length)
        edited = output.clone()
        edited[rows] = edited[rows] + self.bias.to(device=output.device, dtype=output.dtype)
        return edited

    def __enter__(self) -> RouterLogitBiasInjector:
        self._handle = self.router.register_forward_hook(self._hook)
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
