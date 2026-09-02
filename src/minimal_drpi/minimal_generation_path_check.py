#!/usr/bin/env python3
"""Minimal OLMoE check: does preserving downstream routes change generation outcomes?

The experiment uses one counterfactual pair.  Its target-minus-source shared
state is the edit direction; static-blind and DRPI are two projections of it.
Free-generation metrics are behavior evidence only.  Router paths are compared
only on the identical source prompt during the prefill forward pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


# DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0924"
DEFAULT_MODEL = "/Users/trickeye/Documents/Study/MoESteering/models/OLMoE-1B-7B-0924"
DEFAULT_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
DEFAULT_PAIR = {
    "pair_id": "meeting-note-finance-to-operations-v1",
    "source": (
        "Read the meeting note and answer the question in one concise sentence.\n\n"
        "The team agreed that travel reimbursement forms must be sent to the finance office "
        "by Friday. Equipment requests go to the lab manager, and venue questions go to "
        "the events coordinator.\n\n"
        "Question: Where should a travel reimbursement form be sent?\n"
        "Answer:"
    ),
    "source_value": "finance office",
    "target": (
        "Read the meeting note and answer the question in one concise sentence.\n\n"
        "The team agreed that travel reimbursement forms must be sent to the operations "
        "office by Friday. Equipment requests go to the lab manager, and venue questions "
        "go to the events coordinator.\n\n"
        "Question: Where should a travel reimbursement form be sent?\n"
        "Answer:"
    ),
    "target_value": "operations office",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/counterfactuals.jsonl")
    parser.add_argument(
        "--pair-id",
        default=None,
        help="Use one record from --data; omit to use this script's richer fixed meeting-note sample.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--t1-atol", type=float, default=2e-5)
    parser.add_argument("--t1-rtol", type=float, default=1e-5)
    parser.add_argument("--out", default="results/summary/minimal_generation_path_check.json")
    return parser.parse_args()


def load_pair(path: str, pair_id: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("pair_id") == pair_id:
                return record
    raise ValueError(f"pair_id={pair_id!r} was not found in {path}")


def dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def model_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    if getattr(getattr(model, "config", None), "model_type", None) != "olmoe":
        raise TypeError("this script only supports Hugging Face OLMoE causal-LM checkpoints")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("OLMoE decoder layers are unavailable at model.model.layers")
    return list(layers)


def encode(tokenizer: Any, text: str, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    return {name: value.to(device) for name, value in encoded.items()}


def final_position(batch: dict[str, torch.Tensor]) -> int:
    return int(batch["attention_mask"][0].sum().item()) - 1


def rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().item())


def rms_scale(direction: torch.Tensor, reference: torch.Tensor, alpha: float) -> torch.Tensor:
    magnitude = rms(direction)
    if magnitude == 0.0:
        raise ValueError("cannot scale a zero direction")
    return direction * (float(alpha) * rms(reference) / magnitude)


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def blind_basis(weight: torch.Tensor, rtol: float = 1e-6) -> torch.Tensor:
    """Return the conservative FP32 basis for ker(W), shaped [hidden, blind]."""
    work = weight.detach().float().cpu()
    _, singular_values, vh = torch.linalg.svd(work, full_matrices=True)
    tolerance = rtol * max(work.shape) * float(singular_values[0].item())
    rank = int((singular_values > tolerance).sum().item())
    return vh[rank:].T.contiguous()


def route_summary(logits: torch.Tensor, top_k: int) -> dict[str, Any]:
    indices = logits.topk(top_k, dim=-1).indices
    return {"logits": logits, "topk": indices}


@dataclass
class PrefillTrace:
    shared_state: torch.Tensor
    routes: dict[int, dict[str, Any]]


class OLMoEHooks(AbstractContextManager["OLMoEHooks"]):
    """Capture prefill routes and inject once at the source prompt's final token."""

    def __init__(
        self,
        layers: list[torch.nn.Module],
        injection_layer: int,
        prompt_length: int,
        direction: torch.Tensor | None,
    ) -> None:
        self.layers = layers
        self.injection_layer = injection_layer
        self.prompt_length = prompt_length
        self.direction = direction
        self.shared_state: torch.Tensor | None = None
        self.routes: dict[int, dict[str, Any]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._injected = False

    def _shared_hook(self, _module: torch.nn.Module, args: tuple[object, ...]):
        state = args[0]
        if not isinstance(state, torch.Tensor) or state.ndim != 3:
            raise TypeError("OLMoE shared MoE input must be [batch, sequence, hidden]")
        if state.shape[0] != 1:
            raise ValueError("minimal experiment requires batch_size=1")
        if state.shape[1] != self.prompt_length or self._injected:
            return None
        self.shared_state = state[0, -1].detach().clone()
        if self.direction is None:
            return None
        delta = self.direction.to(device=state.device, dtype=state.dtype)
        if delta.shape != state.shape[-1:]:
            raise ValueError("direction does not match OLMoE hidden size")
        edited = state.clone()
        edited[0, -1] = edited[0, -1] + delta
        self._injected = True
        return (edited, *args[1:])

    def _gate_hook(self, layer: int):
        def hook(_module: torch.nn.Module, _args: tuple[object, ...], output: Any) -> None:
            logits = output[0] if isinstance(output, tuple) else output
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                raise TypeError("OLMoE router output must be [batch * sequence, experts]")
            if logits.shape[0] != self.prompt_length or layer in self.routes:
                return
            top_k = int(self.layers[layer].mlp.top_k)
            self.routes[layer] = route_summary(logits[-1].detach().float().cpu(), top_k)

        return hook

    def __enter__(self) -> "OLMoEHooks":
        mlp = self.layers[self.injection_layer].mlp
        self._handles.append(mlp.register_forward_pre_hook(self._shared_hook))
        for layer, decoder_layer in enumerate(self.layers):
            handle = decoder_layer.mlp.gate.register_forward_hook(self._gate_hook(layer))
            self._handles.append(handle)
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def trace(self) -> PrefillTrace:
        if self.shared_state is None:
            raise RuntimeError("the injection-layer prompt state was not captured")
        if len(self.routes) != len(self.layers):
            raise RuntimeError("not all OLMoE prefill routes were captured")
        return PrefillTrace(self.shared_state, self.routes)


class GradientHooks(AbstractContextManager["GradientHooks"]):
    """Expose downstream margin gradients with respect to the injection state."""

    def __init__(
        self, layers: list[torch.nn.Module], injection_layer: int, prompt_length: int, horizon: int
    ) -> None:
        self.layers = layers
        self.injection_layer = injection_layer
        self.prompt_length = prompt_length
        self.target_layers = range(
            injection_layer + 1, min(len(layers), injection_layer + horizon + 1)
        )
        self.leaf: torch.Tensor | None = None
        self.logits: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _leaf_hook(self, _module: torch.nn.Module, args: tuple[object, ...]):
        state = args[0]
        if not isinstance(state, torch.Tensor) or state.shape[1] != self.prompt_length:
            return None
        leaf = state.detach().requires_grad_(True)
        self.leaf = leaf
        return (leaf, *args[1:])

    def _gate_hook(self, layer: int):
        def hook(_module: torch.nn.Module, _args: tuple[object, ...], output: Any) -> None:
            logits = output[0] if isinstance(output, tuple) else output
            if isinstance(logits, torch.Tensor) and logits.shape[0] == self.prompt_length:
                self.logits[layer] = logits.reshape(1, self.prompt_length, -1)

        return hook

    def __enter__(self) -> "GradientHooks":
        handle = self.layers[self.injection_layer].mlp.register_forward_pre_hook(self._leaf_hook)
        self._handles.append(handle)
        for layer in self.target_layers:
            handle = self.layers[layer].mlp.gate.register_forward_hook(self._gate_hook(layer))
            self._handles.append(handle)
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def dangerous_basis(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    layers: list[torch.nn.Module],
    injection_layer: int,
    horizon: int,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, int]]]:
    """Use one weakest-selected versus best-outside margin per downstream layer."""
    prompt_length, token = batch["input_ids"].shape[1], final_position(batch)
    with GradientHooks(layers, injection_layer, prompt_length, horizon) as capture:
        model(**batch, use_cache=False, return_dict=True)
        if capture.leaf is None:
            raise RuntimeError("gradient injection state was not captured")
        rows: list[torch.Tensor] = []
        boundaries: list[dict[str, int]] = []
        for layer in capture.target_layers:
            logits = capture.logits.get(layer)
            if logits is None:
                raise RuntimeError(f"downstream router layer {layer} was not captured")
            score = logits[0, token]
            top_k = int(layers[layer].mlp.top_k)
            selected = score.topk(top_k).indices
            inside = int(selected[-1].item())
            outside_mask = torch.ones_like(score, dtype=torch.bool)
            outside_mask[selected] = False
            outside = int(score.masked_fill(~outside_mask, -torch.inf).argmax().item())
            margin = score[inside] - score[outside]
            gradient = torch.autograd.grad(margin, capture.leaf, retain_graph=True)[0][0, token]
            rows.append(gradient.detach().float().cpu() @ basis)
            boundaries.append({"layer": layer, "inside_expert": inside, "outside_expert": outside})
    matrix = torch.stack(rows)
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    rank = int((singular_values > 1e-8).sum().item())
    return vh[:rank].T.contiguous(), boundaries


def route_comparison(
    baseline: PrefillTrace, edited: PrefillTrace, injection_layer: int
) -> dict[str, Any]:
    current_base, current_edit = baseline.routes[injection_layer], edited.routes[injection_layer]
    logit_delta = current_edit["logits"] - current_base["logits"]
    current_exact = bool(torch.equal(current_base["topk"], current_edit["topk"]))
    baseline_scale = max(
        float(current_base["logits"].abs().max().item()), torch.finfo(torch.float32).tiny
    )
    downstream = []
    for layer in range(injection_layer + 1, len(baseline.routes)):
        exact = bool(torch.equal(baseline.routes[layer]["topk"], edited.routes[layer]["topk"]))
        downstream.append({"layer": layer, "topk_exact": exact})
    first_switch = next((row["layer"] for row in downstream if not row["topk_exact"]), None)
    return {
        "injection_logit_max_abs_delta": float(logit_delta.abs().max().item()),
        "injection_logit_max_relative_delta": float(logit_delta.abs().max().item())
        / baseline_scale,
        "injection_topk_exact": current_exact,
        "first_downstream_switch_layer": first_switch,
        "downstream_topk_exact_fraction": (
            sum(row["topk_exact"] for row in downstream) / len(downstream) if downstream else 1.0
        ),
        "downstream": downstream,
    }


def greedy_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, torch.Tensor],
    layers: list[torch.nn.Module],
    injection_layer: int,
    direction: torch.Tensor | None,
    max_new_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, PrefillTrace]:
    """Inject during prefill only; cache steps never receive the intervention."""
    prompt_length = batch["input_ids"].shape[1]
    with torch.no_grad(), OLMoEHooks(layers, injection_layer, prompt_length, direction) as hooks:
        cache = DynamicCache()
        output = model(**batch, past_key_values=cache, use_cache=True, return_dict=True)
        prefill_logits = output.logits[:, -1].detach()
        trace = hooks.trace()
        full_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        for step in range(max_new_tokens):
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            full_ids = torch.cat((full_ids, next_token), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)
            is_eos = (
                tokenizer.eos_token_id is not None
                and int(next_token.item()) == tokenizer.eos_token_id
            )
            if is_eos:
                break
            if step + 1 == max_new_tokens:
                break
            output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
    return full_ids, prefill_logits, trace


def clean_continuation_nll(
    model: torch.nn.Module, full_ids: torch.Tensor, prompt_length: int
) -> float:
    """Score the generated continuation under the unedited model as a compatibility proxy."""
    continuation = full_ids[:, prompt_length:]
    if continuation.numel() == 0:
        return math.nan
    with torch.no_grad():
        output = model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            use_cache=False,
        )
        logits = output.logits
    predicted = logits[:, prompt_length - 1 : prompt_length - 1 + continuation.shape[1]].float()
    loss = functional.cross_entropy(predicted.transpose(1, 2), continuation, reduction="mean")
    return float(loss.item())


def repetition_metrics(ids: torch.Tensor) -> dict[str, float]:
    values = ids[0].tolist()
    if not values:
        return {"repeat_bigram_rate": math.nan, "max_run": 0.0}
    bigrams = list(zip(values, values[1:]))
    repeat_rate = 1.0 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    longest, current = 1, 1
    for left, right in zip(values, values[1:]):
        current = current + 1 if left == right else 1
        longest = max(longest, current)
    return {"repeat_bigram_rate": repeat_rate, "max_run": float(longest)}


def target_logprob(logits: torch.Tensor, target_token: int) -> float:
    return float(torch.log_softmax(logits.float(), dim=-1)[0, target_token].item())


def main() -> None:
    args = parse_args()
    if args.horizon < 1 or args.alpha <= 0.0 or args.max_new_tokens < 1:
        raise ValueError("horizon, alpha, and max-new-tokens must be positive")
    torch.manual_seed(args.seed)
    device, dtype = torch.device(args.device), dtype_from_name(args.dtype)
    pair = DEFAULT_PAIR if args.pair_id is None else load_pair(args.data, args.pair_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval().requires_grad_(False)
    layers = model_layers(model)
    if args.layer < 0 or args.layer + args.horizon >= len(layers):
        raise ValueError("layer + horizon must remain inside the OLMoE decoder")

    print("model ok")

    source = encode(tokenizer, str(pair["source"]), device)
    target = encode(tokenizer, str(pair["target"]), device)
    prompt_length = source["input_ids"].shape[1]
    with torch.no_grad(), OLMoEHooks(layers, args.layer, prompt_length, None) as source_hooks:
        print("running source prefill")
        model(**source, use_cache=False, return_dict=True)
        source_trace = source_hooks.trace()
    target_length = target["input_ids"].shape[1]
    with torch.no_grad(), OLMoEHooks(layers, args.layer, target_length, None) as target_hooks:
        print("running target prefill")
        model(**target, use_cache=False, return_dict=True)
        target_trace = target_hooks.trace()
    print("target prefill complete")

    weight = layers[args.layer].mlp.gate.weight
    basis = blind_basis(weight)
    raw_direction = (
        target_trace.shared_state.float().cpu() - source_trace.shared_state.float().cpu()
    )
    static_master = basis @ (basis.T @ raw_direction)
    dangerous, boundaries = dangerous_basis(model, source, layers, args.layer, args.horizon, basis)
    coordinates = basis.T @ raw_direction
    drpi_master = basis @ (coordinates - dangerous @ (dangerous.T @ coordinates))
    reference = source_trace.shared_state.float().cpu()
    directions = {
        "static_blind": rms_scale(static_master, reference, args.alpha),
        "drpi": rms_scale(drpi_master, reference, args.alpha),
    }
    target_ids = tokenizer(str(pair["target_value"]), add_special_tokens=False)["input_ids"]
    if not target_ids:
        raise ValueError("target_value tokenized to no tokens")
    target_token = int(target_ids[0])
    print("running greedy generation")

    print(f"Target token: {target_token}")
    started = time.perf_counter()
    baseline_ids, baseline_logits, baseline_trace = greedy_generate(
        model, tokenizer, source, layers, args.layer, None, args.max_new_tokens
    )
    baseline_text = tokenizer.decode(baseline_ids[0, prompt_length:], skip_special_tokens=True)
    baseline_logprob = target_logprob(baseline_logits, target_token)
    report_arms: dict[str, Any] = {
        "baseline": {
            "generated_text": baseline_text,
            "generated_token_ids": baseline_ids[0, prompt_length:].tolist(),
            "target_first_token_logprob": baseline_logprob,
            "clean_continuation_nll": clean_continuation_nll(model, baseline_ids, prompt_length),
            **repetition_metrics(baseline_ids[:, prompt_length:]),
        }
    }
    print("baseline generation complete")

    for name, master_direction in directions.items():
        print("generatin for direction:", name)
        full_ids, prefill_logits, edited_trace = greedy_generate(
            model,
            tokenizer,
            source,
            layers,
            args.layer,
            master_direction.to(device=device),
            args.max_new_tokens,
        )
        print("generation complete for direction:", name)
        comparison = route_comparison(baseline_trace, edited_trace, args.layer)
        t1_passed = comparison["injection_topk_exact"] and (
            comparison["injection_logit_max_abs_delta"] <= args.t1_atol
            or comparison["injection_logit_max_relative_delta"] <= args.t1_rtol
        )
        report_arms[name] = {
            "generated_text": tokenizer.decode(
                full_ids[0, prompt_length:], skip_special_tokens=True
            ),
            "generated_token_ids": full_ids[0, prompt_length:].tolist(),
            "target_first_token_logprob": target_logprob(prefill_logits, target_token),
            "target_first_token_logprob_delta": target_logprob(prefill_logits, target_token)
            - baseline_logprob,
            "clean_continuation_nll": clean_continuation_nll(model, full_ids, prompt_length),
            "direction_rms_over_source_state_rms": rms(master_direction) / rms(reference),
            "t1_passed": t1_passed,
            "prefill_path": comparison,
            **repetition_metrics(full_ids[:, prompt_length:]),
        }

    report = {
        "schema_version": "minimal-drpi-generation-path-check-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "completed"
        if all(report_arms[name]["t1_passed"] for name in directions)
        else "invalid_t1",
        "model_id": args.model,
        "model_revision": args.revision,
        "router_weight_sha256": tensor_hash(weight),
        "device": str(device),
        "requested_dtype": args.dtype,
        "actual_dtype": str(next(model.parameters()).dtype),
        "pair_id": args.pair_id,
        "source": pair["source"],
        "target": pair["target"],
        "injection_layer": args.layer,
        "horizon": args.horizon,
        "alpha": args.alpha,
        "seed": args.seed,
        "dangerous_rank": int(dangerous.shape[1]),
        "dangerous_boundaries": boundaries,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": (
            "Compare static_blind and drpi only after their target_first_token_logprob_delta "
            "is sufficiently close. Free-generation paths are not token-aligned evidence. "
            "clean_continuation_nll and repetition are proxies, not a human-quality verdict."
        ),
        "arms": report_arms,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
