"""
Plot reasoning token count distributions for benign vs pressure variants.

Usage:
    PYTHONPATH=. python experiments/plot_token_lengths.py \
        --input results/rat_iter4_benchmark.json \
        --output results/token_lengths.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot token count histograms for benign vs pressure.")
    parser.add_argument("--input", default="results/rat_iter4_benchmark.json", help="Path to benchmark JSON.")
    parser.add_argument("--output", default="results/token_lengths.png", help="Path to save PNG.")
    parser.add_argument("--bins", type=int, default=60, help="Number of histogram bins.")
    parser.add_argument("--xmax", type=int, default=None, help="Optional x-axis max to clip long tails.")
    return parser.parse_args()


def load_token_counts(path: Path) -> Dict[str, List[int]]:
    data = json.load(open(path))
    buckets: Dict[str, List[int]] = {"benign": [], "pressure": []}
    for r in data:
        variant = r.get("variant")
        tok = r.get("token_count")
        if variant in buckets and tok is not None:
            buckets[variant].append(tok)
    return buckets


def summary(label: str, vals: List[int]) -> None:
    if not vals:
        print(f"{label}: no data")
        return
    qs = quantiles(vals, n=4)
    print(
        f"{label}: n={len(vals)}, mean={mean(vals):.1f}, median={median(vals):.1f}, "
        f"q1={qs[0]:.1f}, q2={qs[1]:.1f}, q3={qs[2]:.1f}, min={min(vals)}, max={max(vals)}"
    )


def plot_histograms(tokens: Dict[str, List[int]], bins: int, output: Path, xmax: int | None) -> None:
    plt.figure(figsize=(7, 4.5))
    kwargs = {"bins": bins, "alpha": 0.55, "edgecolor": "none"}
    if xmax is not None:
        plt.xlim(0, xmax)
    plt.hist(tokens.get("benign", []), label="benign", color="#1f77b4", **kwargs)
    plt.hist(tokens.get("pressure", []), label="pressure", color="#d62728", **kwargs)
    plt.xlabel("Reasoning token count")
    plt.ylabel("Frequency")
    plt.title("Token count distribution: benign vs pressure")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved plot to {output}")


def main() -> None:
    args = parse_args()
    tokens = load_token_counts(Path(args.input))
    summary("Benign", tokens["benign"])
    summary("Pressure", tokens["pressure"])
    plot_histograms(tokens, bins=args.bins, output=Path(args.output), xmax=args.xmax)


if __name__ == "__main__":
    main()

