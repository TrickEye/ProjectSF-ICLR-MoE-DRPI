import json

import pytest

from drpi.datasets import PromptRecord, serialize_records, stable_split
from drpi.records import JsonlWriter


def test_split_and_serialization_are_deterministic():
    assert stable_split("same", 9) == stable_split("same", 9)
    record = PromptRecord("p1", "hello", "train")
    assert serialize_records([record]) == serialize_records([record])


def test_append_only_writer_validates_and_resumes(tmp_path):
    writer = JsonlWriter(tmp_path / "run.jsonl")
    with pytest.raises(ValueError, match="missing required"):
        writer.append({"prompt_id": "p"})
    record = {
        "schema_version": "drpi-run-v1",
        "generated_at": "now",
        "model_id": "m",
        "model_revision": "r",
        "router_weight_sha256": "h",
        "code_commit": None,
        "device": "cpu",
        "dtype": "float32",
        "quantization": "none",
        "hook_path": "x",
        "prompt_id": "p",
        "split": "test",
        "seed": 1,
        "injection_layer": 0,
        "token_position": 1,
        "alpha": 0.1,
        "horizon": 1,
    }
    writer.append(record)
    assert writer.completed_keys(["prompt_id", "alpha"]) == {("p", 0.1)}
    assert json.loads((tmp_path / "run.jsonl").read_text())["prompt_id"] == "p"
