#!/usr/bin/env python3
"""Plot effect-matched static-blind versus DRPI route divergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/summary/effect_matching.json")
    parser.add_argument("--out", default="results/summary/effect_matched_path_divergence.png")
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    reference, candidate = report["reference"], report["candidate"]
    rows = report["matched_prompts"]
    if not rows:
        raise ValueError("effect-matching report contains no prompts")
    x = [1.0 - float(row[f"{reference}_topk_jaccard"]) for row in rows]
    y = [1.0 - float(row[f"{candidate}_topk_jaccard"]) for row in rows]
    figure, axis = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    axis.scatter(x, y, s=18, alpha=0.65)
    limit = max([0.01, *x, *y])
    axis.plot([0, limit], [0, limit], color="black", linewidth=1, linestyle="--")
    axis.set(
        xlabel=f"{reference} path divergence",
        ylabel=f"{candidate} path divergence",
        xlim=(-0.01, limit * 1.05),
        ylim=(-0.01, limit * 1.05),
    )
    axis.grid(alpha=0.2)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
