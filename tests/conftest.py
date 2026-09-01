from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from drpi.model_adapter import LayerDescription


class ToyBlock(nn.Module):
    def __init__(self, hidden: int, experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden, experts, bias=False)
        self.mix = nn.Linear(hidden, hidden, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        self.gate(state.reshape(-1, state.shape[-1]))
        return state + torch.tanh(self.mix(state))


class ToyModel(nn.Module):
    def __init__(self, layers: int = 3, hidden: int = 8, experts: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(32, hidden)
        self.blocks = nn.ModuleList([ToyBlock(hidden, experts) for _ in range(layers)])
        self.lm_head = nn.Linear(hidden, 32, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        state = self.embedding(input_ids)
        for block in self.blocks:
            state = block(state)
        return SimpleNamespace(logits=self.lm_head(state))


class ToyAdapter:
    def __init__(self):
        torch.manual_seed(4)
        self.model = ToyModel()

    def num_layers(self):
        return len(self.model.blocks)

    def router(self, layer):
        return self.model.blocks[layer].gate

    def router_weight(self, layer):
        return self.router(layer).weight

    def shared_state_module(self, layer):
        return self.model.blocks[layer]

    def top_k(self, layer):
        return 2

    def router_logits_from_input(self, layer, state):
        logits = self.router(layer)(state.reshape(-1, state.shape[-1]))
        return logits.reshape(*state.shape[:-1], logits.shape[-1])

    def describe_layer(self, layer):
        return LayerDescription(
            layer=layer,
            shared_state_path=f"blocks.{layer}",
            router_path=f"blocks.{layer}.gate",
            router_weight_shape=tuple(self.router_weight(layer).shape),
            router_has_bias=False,
            top_k=2,
            num_experts=3,
            norm_topk_prob=False,
            shared_experts=False,
            router_weight_sha256="toy",
        )


def make_batch():
    return {
        "input_ids": torch.tensor([[2, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }

