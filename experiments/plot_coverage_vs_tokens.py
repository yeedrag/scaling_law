"""
Plot pressure coverage (and legibility) vs benign token count from
rat_iter4_benchmark.json, plus basic correlations and binned stats.

Usage:
    PYTHONPATH=. python experiments/plot_coverage_vs_tokens.py \
        --input results/rat_iter4_benchmark.json \
        --coverage-output results/coverage_vs_tokens_pressure.png \
        --legibility-output results/legibility_vs_tokens_pressure.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pressure coverage/legibility vs benign token count and print stats.")
    parser.add_argument("--input", default="results/rat_iter4_benchmark.json", help="Path to benchmark JSON.")
    parser.add_argument("--coverage-output", default="results/coverage_vs_tokens_pressure.png", help="Path to save coverage PNG.")
    parser.add_argument("--legibility-output", default="results/legibility_vs_tokens_pressure.png", help="Path to save legibility PNG.")
    parser.add_argument(
        "--bins",
        default="0,150,300,450,600,1000,1500",
        help="Comma-separated benign token count bucket edges for binned stats.",
    )
    return parser.parse_args()


def parse_bins(spec: str) -> List[int]:
    try:
        bins = [int(x) for x in spec.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise SystemExit(f"Invalid --bins spec: {spec}") from exc
    if len(bins) < 2:
        raise SystemExit("Need at least two bin edges")
    return sorted(bins)


def load_pairs(path: Path) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    data = json.load(open(path))
    benign_map = {}
    pressure_map = {}
    for r in data:
        q = r.get("question")
        if not q:
            continue
        if r.get("variant") == "benign":
            benign_map[q] = r
        elif r.get("variant") == "pressure":
            pressure_map[q] = r

    coverage_pairs: list[tuple[int, int]] = []
    legibility_pairs: list[tuple[int, int]] = []
    for q, b in benign_map.items():
        p = pressure_map.get(q)
        if not p:
            continue
        tok = b.get("token_count")
        if tok is None:
            continue
        cov = p.get("coverage")
        leg = p.get("legibility")
        if cov is not None:
            coverage_pairs.append((tok, cov))
        if leg is not None:
            legibility_pairs.append((tok, leg))
    return coverage_pairs, legibility_pairs


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (vx * vy) ** 0.5


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    # Simple average-rank Spearman; fall back to pearson on rank lists.
    def ranks(vals: Sequence[float]) -> List[float]:
        sorted_pairs = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
                j += 1
            avg_rank = (i + j) / 2 + 1  # 1-based
            for k in range(i, j + 1):
                ranks[sorted_pairs[k][0]] = avg_rank
            i = j + 1
        return ranks

    if len(xs) != len(ys):
        return float("nan")
    return pearson(ranks(xs), ranks(ys))


def bin_stats(pairs: Iterable[tuple[int, int]], bins: Sequence[int]) -> list[dict]:
    """Return list of bucket stats: edge_low, edge_high, count, mean."""
    results = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        bucket = [y for x, y in pairs if lo <= x < hi]
        results.append(
            {
                "range": f"[{lo}, {hi})",
                "count": len(bucket),
                "mean": round(mean(bucket), 3) if bucket else None,
            }
        )
    # last bin inclusive of upper edge
    last_lo = bins[-1]
    bucket = [y for x, y in pairs if x >= last_lo]
    results.append(
        {
            "range": f"[{last_lo}, inf)",
            "count": len(bucket),
            "mean": round(mean(bucket), 3) if bucket else None,
        }
    )
    return results


def report(name: str, pairs: list[tuple[int, int]], bins: Sequence[int]) -> None:
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    print(f"\n{name}: n={len(pairs)}, mean_x={mean(xs):.1f}, mean_y={mean(ys):.3f}")
    print(f"  Pearson:  {pearson(xs, ys):.4f}")
    print(f"  Spearman: {spearman(xs, ys):.4f}")
    print("  Bins (mean of y per bin):")
    for row in bin_stats(pairs, bins):
        print(f"    {row['range']}: count={row['count']}, mean={row['mean']}")


def plot_pairs(pairs: list[tuple[int, int]], ylabel: str, title: str, output: Path) -> None:
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, alpha=0.4, s=14)
    plt.xlabel("Benign token count (reasoning)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved plot to {output}")


def main() -> None:
    args = parse_args()
    bins = parse_bins(args.bins)
    coverage_pairs, legibility_pairs = load_pairs(Path(args.input))

    if coverage_pairs:
        report("Coverage vs benign tokens", coverage_pairs, bins)
        plot_pairs(
            coverage_pairs,
            ylabel="Pressure coverage score",
            title="Pressure coverage vs benign token count",
            output=Path(args.coverage_output),
        )
    else:
        print("No coverage data found.")

    if legibility_pairs:
        report("Legibility vs benign tokens", legibility_pairs, bins)
        plot_pairs(
            legibility_pairs,
            ylabel="Pressure legibility score",
            title="Pressure legibility vs benign token count",
            output=Path(args.legibility_output),
        )
    else:
        print("No legibility data found.")


if __name__ == "__main__":
    main()


