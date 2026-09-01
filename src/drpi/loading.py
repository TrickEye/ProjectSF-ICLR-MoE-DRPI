"""Explicit model loading and cache checks for local and CUDA backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from drpi.config import torch_dtype


@dataclass(frozen=True)
class CacheReport:
    complete: bool
    index_path: str | None
    required_shards: tuple[str, ...]
    missing_shards: tuple[str, ...]


def check_weight_cache(model_id: str, revision: str) -> CacheReport:
    """Check every shard named by the cached safetensors index."""
    index = try_to_load_from_cache(
        model_id, "model.safetensors.index.json", revision=revision
    )
    if not isinstance(index, str):
        return CacheReport(False, None, (), ("model.safetensors.index.json",))
    with Path(index).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    shards = tuple(sorted(set(payload.get("weight_map", {}).values())))
    missing = []
    for shard in shards:
        resolved = try_to_load_from_cache(model_id, shard, revision=revision)
        if not isinstance(resolved, str) or not Path(resolved).is_file():
            missing.append(shard)
    return CacheReport(not missing, index, shards, tuple(missing))


def resolve_device(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this process")
    elif name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
    elif name != "cpu":
        raise ValueError(f"unsupported device: {name}")
    return torch.device(name)


def load_model_and_tokenizer(model_config: dict[str, Any]):
    """Load a pinned model without implicit dtype or quantization fallback."""
    model_id = str(model_config["id"])
    revision = str(model_config["revision"])
    device = resolve_device(str(model_config["device"]))
    dtype = torch_dtype(str(model_config["dtype"]))
    quantization = str(model_config.get("quantization", "none"))
    local_files_only = bool(model_config.get("local_files_only", True))

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=local_files_only
    )
    kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if quantization == "none":
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.to(device)
    elif quantization == "int8":
        if device.type != "cuda":
            raise RuntimeError("int8 loading is only enabled for the validated CUDA path")
        from transformers import BitsAndBytesConfig

        # Router gates remain unquantized so static null-space checks use real gate weights.
        skip_modules = [f"model.layers.{layer}.mlp.gate" for layer in range(256)]
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_skip_modules=skip_modules
        )
        kwargs["device_map"] = {"": device.index or 0}
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    else:
        raise ValueError(f"unsupported quantization mode: {quantization}")
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer

