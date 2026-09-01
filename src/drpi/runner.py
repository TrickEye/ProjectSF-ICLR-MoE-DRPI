"""Unified teacher-forced execution entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from drpi.interventions import (
    Injection,
    RouterLogitBiasInjector,
    SharedStateInjector,
    final_valid_positions,
)
from drpi.model_adapter import MoEAdapter
from drpi.router_capture import RouteRecord, RouterCapture


@dataclass
class RunResult:
    """Model output and captured routes from one aligned forward pass."""

    output_logits: torch.Tensor
    routes: dict[int, RouteRecord]


class ExperimentRunner:
    """Run baseline and edited teacher-forced passes with explicit hook order."""

    def __init__(self, adapter: MoEAdapter):
        self.adapter = adapter

    def teacher_forced(
        self,
        batch: dict[str, torch.Tensor],
        *,
        injection_layer: int | None = None,
        direction: torch.Tensor | None = None,
        alpha: float = 1.0,
        layers: list[int] | None = None,
        require_grad: bool = False,
        router_bias_layer: int | None = None,
        router_bias: torch.Tensor | None = None,
    ) -> RunResult:
        """Run on identical input tokens; no autoregressive token substitution occurs."""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
        injector = None
        if injection_layer is not None:
            if direction is None:
                raise ValueError("direction is required when injection_layer is set")
            injection = Injection(
                direction=direction,
                alpha=alpha,
                token_positions=final_valid_positions(attention_mask),
            )
            injector = SharedStateInjector(
                self.adapter.shared_state_module(injection_layer), injection
            )
        bias_injector = None
        if router_bias_layer is not None:
            if router_bias is None:
                raise ValueError("router_bias is required when router_bias_layer is set")
            bias_injector = RouterLogitBiasInjector(
                self.adapter.router(router_bias_layer),
                router_bias,
                token_positions=final_valid_positions(attention_mask),
                sequence_length=input_ids.shape[1],
            )

        context = torch.enable_grad() if require_grad else torch.no_grad()
        if injector is not None:
            injector.__enter__()
        if bias_injector is not None:
            bias_injector.__enter__()
        try:
            # Register capture after injection so the captured state is the actual edited state.
            with RouterCapture(self.adapter, layers=layers) as capture, context:
                output: Any = self.adapter.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                routes = capture.records if require_grad else capture.detached()
                logits = output.logits if require_grad else output.logits.detach()
            return RunResult(output_logits=logits, routes=routes)
        finally:
            if injector is not None:
                injector.remove()
            if bias_injector is not None:
                bias_injector.__exit__(None, None, None)
