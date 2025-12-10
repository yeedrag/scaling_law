"""
Bin questions by benign average reasoning length and plot mean accuracy per bin.

Usage:
    PYTHONPATH=. python experiments/plot_accuracy_by_length.py \
        --baseline results/baseline_accuracy_samples.jsonl \
        --bins 0,150,300,450,600,1000,2000 \
        --output results/accuracy_by_length.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot accuracy by benign avg token length bins.")
    p.add_argument("--baseline", default="results/baseline_accuracy_samples.jsonl", help="Path to baseline accuracy JSONL.")
    p.add_argument("--bins", default="0,150,300,450,600,1000,2000", help="Comma-separated bin edges (last bin is [last, inf)).")
    p.add_argument("--output", default="results/accuracy_by_length.png", help="Output PNG path.")
    return p.parse_args()


def load_baseline(path: Path) -> List[Dict]:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            recs.append(json.loads(line))
    return recs


def parse_bins(spec: str) -> List[int]:
    try:
        bins = [int(x) for x in spec.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise SystemExit(f"Invalid --bins spec: {spec}") from exc
    if len(bins) < 2:
        raise SystemExit("Need at least two bin edges")
    return sorted(bins)


def main() -> None:
    args = parse_args()
    bins = parse_bins(args.bins)
    data = load_baseline(Path(args.baseline))

    # Prepare buckets
    bucket_stats = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        bucket_stats.append({"range": f"[{lo},{hi})", "accs": []})
    bucket_stats.append({"range": f"[{bins[-1]},inf)", "accs": []})

    for rec in data:
        tok = rec.get("avg_token_count")
        acc = rec.get("accuracy")
        if tok is None or acc is None:
            continue
        placed = False
        for b, (lo, hi) in zip(bucket_stats[:-1], zip(bins[:-1], bins[1:])):
            if lo <= tok < hi:
                b["accs"].append(acc)
                placed = True
                break
        if not placed:
            bucket_stats[-1]["accs"].append(acc)

    # Compute means
    labels = []
    means = []
    counts = []
    for b in bucket_stats:
        labels.append(b["range"])
        if b["accs"]:
            means.append(mean(b["accs"]))
            counts.append(len(b["accs"]))
        else:
            means.append(0.0)
            counts.append(0)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar(labels, means, color="#1f77b4", alpha=0.8)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean accuracy")
    plt.xlabel("Benign avg token count bin")
    plt.ylim(0, 1.05)
    plt.title("Baseline accuracy by reasoning length bin")
    # annotate counts
    for i, (m, c) in enumerate(zip(means, counts)):
        plt.text(i, m + 0.02, str(c), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    for b, m, c in zip(labels, means, counts):
        print(f"{b}: n={c}, mean_acc={m:.3f}")


if __name__ == "__main__":
    main()

