#!/usr/bin/env python3
"""Generate leakage survival curves from immutable raw JSONL records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/raw/leakage.jsonl")
    parser.add_argument("--out", default="results/summary/static_blind_leakage.png")
    args = parser.parse_args()
    grouped: dict[tuple[float, int], list[float]] = defaultdict(list)
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[(float(row["alpha"]), int(row["horizon"]))].append(
                    float(bool(row["topk_set_exact"]))
                )
    if not grouped:
        raise ValueError("no leakage records")
    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for alpha in sorted({key[0] for key in grouped}):
        horizons = sorted(key[1] for key in grouped if key[0] == alpha)
        means = [float(np.mean(grouped[(alpha, horizon)])) for horizon in horizons]
        axis.plot(horizons, means, marker="o", label=f"alpha={alpha:g}")
    axis.set(xlabel="Layer distance", ylabel="Top-k set survival", ylim=(-0.02, 1.02))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()

