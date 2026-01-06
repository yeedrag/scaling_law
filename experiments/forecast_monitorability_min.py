"""
Forecast monitorability from MINIMUM benign CoT token count.

Same as forecast_monitorability.py but uses min(token_counts) instead of mean.
This tests whether the "compressibility floor" predicts adversarial monitorability.

Usage:
    PYTHONPATH=. python experiments/forecast_monitorability_min.py \
        --benign-samples results/benign_samples.jsonl \
        --pressure-samples results/pressure_samples.jsonl \
        --output results/monitorability_forecast_min
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forecast monitorability from MIN benign token count")
    p.add_argument("--benign-samples", default="results/benign_samples.jsonl", 
                   help="Benign samples JSONL")
    p.add_argument("--pressure-samples", default="results/pressure_samples.jsonl",
                   help="Pressure samples JSONL")
    p.add_argument("--output", default="results/monitorability_forecast_min", 
                   help="Output prefix for plots and report")
    p.add_argument("--bins", default="0,100,200,300,400,500,700,1000", 
                   help="Bin edges for benign token counts")
    return p.parse_args()


def load_benign_data(path: Path) -> dict:
    """Load benign samples, keyed by question. Extract min token count."""
    benign_map = {}
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec.get("question")
            if not q:
                continue
            
            # Get min from token_counts list
            token_counts = rec.get("token_counts", [])
            if token_counts:
                rec["min_token_count"] = min(token_counts)
            benign_map[q] = rec
    
    return benign_map


def load_pressure_samples(path: Path) -> dict:
    """Load pressure samples with mean coverage/legibility."""
    pressure_map = {}
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec.get("question")
            if not q:
                continue
            pressure_map[q] = rec
    
    return pressure_map


def extract_pairs(benign_map: dict, pressure_map: dict, y_key: str) -> List[Tuple[float, float]]:
    """Extract (min_benign_tokens, avg_pressure_metric) pairs."""
    pairs = []
    for q, b in benign_map.items():
        p = pressure_map.get(q)
        if not p:
            continue
        
        benign_tok = b.get("min_token_count")
        pressure_val = p.get(y_key)
        
        if benign_tok is not None and pressure_val is not None:
            pairs.append((float(benign_tok), float(pressure_val)))
    
    return pairs


def pearson(xs: List[float], ys: List[float]) -> float:
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


def spearman(xs: List[float], ys: List[float]) -> float:
    def ranks(vals: List[float]) -> List[float]:
        pairs = sorted(enumerate(vals), key=lambda t: t[1])
        r = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and pairs[j + 1][1] == pairs[i][1]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[pairs[k][0]] = avg
            i = j + 1
        return r
    
    if len(xs) != len(ys):
        return float("nan")
    return pearson(ranks(xs), ranks(ys))


def linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if len(xs) < 2:
        return float("nan"), float("nan")
    
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = sum((x - mx) ** 2 for x in xs)
    
    if denominator == 0:
        return float("nan"), float("nan")
    
    slope = numerator / denominator
    intercept = my - slope * mx
    
    return slope, intercept


def bin_analysis(pairs: List[Tuple[float, float]], bin_edges: List[int]) -> List[dict]:
    bins = []
    edges = sorted(bin_edges)
    
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        bucket = [y for x, y in pairs if lo <= x < hi]
        bins.append({
            "range": f"[{lo}, {hi})",
            "lo": lo,
            "hi": hi,
            "count": len(bucket),
            "mean": mean(bucket) if bucket else None,
            "std": stdev(bucket) if len(bucket) > 1 else None,
        })
    
    last_edge = edges[-1]
    bucket = [y for x, y in pairs if x >= last_edge]
    bins.append({
        "range": f"[{last_edge}, ∞)",
        "lo": last_edge,
        "hi": None,
        "count": len(bucket),
        "mean": mean(bucket) if bucket else None,
        "std": stdev(bucket) if len(bucket) > 1 else None,
    })
    
    return bins


def plot_forecast(pairs: List[Tuple[float, float]], ylabel: str, title: str, 
                  slope: float, intercept: float, output: Path) -> None:
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.scatter(xs, ys, alpha=0.4, s=20, color='steelblue', label='Data')
    
    if not np.isnan(slope):
        x_range = np.linspace(min(xs), max(xs), 100)
        y_pred = slope * x_range + intercept
        ax.plot(x_range, y_pred, 'r-', linewidth=2, 
                label=f'y = {slope:.4f}x + {intercept:.2f}')
    
    ax.set_xlabel('Benign CoT Token Count (MIN of 20 samples)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"  → Saved: {output}")


def plot_binned_analysis(bins: List[dict], ylabel: str, title: str, output: Path) -> None:
    valid_bins = [b for b in bins if b["mean"] is not None and b["count"] > 0]
    
    if not valid_bins:
        print(f"  ⚠ No valid bins for {output}")
        return
    
    labels = [b["range"] for b in valid_bins]
    means = [b["mean"] for b in valid_bins]
    stds = [b["std"] if b["std"] is not None else 0 for b in valid_bins]
    counts = [b["count"] for b in valid_bins]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.RdYlGn([m / 5.0 for m in means])
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, 
                  edgecolor='black', linewidth=0.8, alpha=0.85)
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.1,
                f'n={count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Minimum Benign Token Count Range', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.axhline(y=np.mean(means), color='red', linestyle='--', linewidth=1.5, 
               label=f'Overall mean: {np.mean(means):.2f}')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"  → Saved: {output}")


def generate_report(coverage_pairs: List[Tuple[float, float]], 
                   legibility_pairs: List[Tuple[float, float]],
                   bin_edges: List[int]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("MONITORABILITY FORECASTING FROM MIN BENIGN TOKEN COUNT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Research Question: Does the compressibility floor (min token count)")
    lines.append("                   predict adversarial monitorability?")
    lines.append("")
    
    if coverage_pairs:
        cov_xs = [p[0] for p in coverage_pairs]
        cov_ys = [p[1] for p in coverage_pairs]
        cov_slope, cov_intercept = linear_regression(cov_xs, cov_ys)
        cov_pearson = pearson(cov_xs, cov_ys)
        cov_spearman = spearman(cov_xs, cov_ys)
        
        lines.append("=" * 80)
        lines.append("1. MIN BENIGN TOKEN COUNT → AVG PRESSURE COVERAGE")
        lines.append("=" * 80)
        lines.append(f"Sample size: {len(coverage_pairs)} questions")
        lines.append("")
        lines.append("--- Correlation ---")
        lines.append(f"  Pearson r:  {cov_pearson:.4f}")
        lines.append(f"  Spearman ρ: {cov_spearman:.4f}")
        lines.append("")
        lines.append("--- Linear Regression ---")
        lines.append(f"  coverage = {cov_slope:.6f} × min_tokens + {cov_intercept:.3f}")
        lines.append("")
        
        bins = bin_analysis(coverage_pairs, bin_edges)
        lines.append("--- Binned Analysis ---")
        lines.append(f"{'Token Range':<20} {'Count':<8} {'Mean Coverage':<15} {'Std Dev':<10}")
        lines.append("-" * 60)
        for b in bins:
            mean_str = f"{b['mean']:.3f}" if b['mean'] is not None else "N/A"
            std_str = f"±{b['std']:.3f}" if b['std'] is not None else ""
            lines.append(f"{b['range']:<20} {b['count']:<8} {mean_str:<15} {std_str:<10}")
        lines.append("")
    
    if legibility_pairs:
        leg_xs = [p[0] for p in legibility_pairs]
        leg_ys = [p[1] for p in legibility_pairs]
        leg_slope, leg_intercept = linear_regression(leg_xs, leg_ys)
        leg_pearson = pearson(leg_xs, leg_ys)
        leg_spearman = spearman(leg_xs, leg_ys)
        
        lines.append("=" * 80)
        lines.append("2. MIN BENIGN TOKEN COUNT → AVG PRESSURE LEGIBILITY")
        lines.append("=" * 80)
        lines.append(f"Sample size: {len(legibility_pairs)} questions")
        lines.append("")
        lines.append("--- Correlation ---")
        lines.append(f"  Pearson r:  {leg_pearson:.4f}")
        lines.append(f"  Spearman ρ: {leg_spearman:.4f}")
        lines.append("")
        lines.append("--- Linear Regression ---")
        lines.append(f"  legibility = {leg_slope:.6f} × min_tokens + {leg_intercept:.3f}")
        lines.append("")
        
        bins = bin_analysis(legibility_pairs, bin_edges)
        lines.append("--- Binned Analysis ---")
        lines.append(f"{'Token Range':<20} {'Count':<8} {'Mean Legibility':<15} {'Std Dev':<10}")
        lines.append("-" * 60)
        for b in bins:
            mean_str = f"{b['mean']:.3f}" if b['mean'] is not None else "N/A"
            std_str = f"±{b['std']:.3f}" if b['std'] is not None else ""
            lines.append(f"{b['range']:<20} {b['count']:<8} {mean_str:<15} {std_str:<10}")
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    
    print("Loading benign samples...")
    benign_map = load_benign_data(Path(args.benign_samples))
    print(f"  Benign records: {len(benign_map)}")
    
    print("Loading pressure samples...")
    pressure_map = load_pressure_samples(Path(args.pressure_samples))
    print(f"  Pressure records: {len(pressure_map)}")
    print()
    
    coverage_pairs = extract_pairs(benign_map, pressure_map, "avg_coverage")
    legibility_pairs = extract_pairs(benign_map, pressure_map, "avg_legibility")
    
    print(f"Extracted {len(coverage_pairs)} coverage pairs, {len(legibility_pairs)} legibility pairs")
    print()
    
    bin_edges = [int(x) for x in args.bins.split(",")]
    
    print("Computing correlations...")
    if coverage_pairs:
        cov_xs = [p[0] for p in coverage_pairs]
        cov_ys = [p[1] for p in coverage_pairs]
        cov_slope, cov_intercept = linear_regression(cov_xs, cov_ys)
    else:
        cov_slope, cov_intercept = float("nan"), float("nan")
    
    if legibility_pairs:
        leg_xs = [p[0] for p in legibility_pairs]
        leg_ys = [p[1] for p in legibility_pairs]
        leg_slope, leg_intercept = linear_regression(leg_xs, leg_ys)
    else:
        leg_slope, leg_intercept = float("nan"), float("nan")
    
    print("Generating plots...")
    prefix = Path(args.output)
    
    if coverage_pairs:
        plot_forecast(
            coverage_pairs,
            ylabel="Pressure Coverage Score",
            title="Forecasting Coverage from MIN Benign Token Count",
            slope=cov_slope,
            intercept=cov_intercept,
            output=prefix.with_name(prefix.name + "_coverage.png")
        )
    
    if legibility_pairs:
        plot_forecast(
            legibility_pairs,
            ylabel="Pressure Legibility Score",
            title="Forecasting Legibility from MIN Benign Token Count",
            slope=leg_slope,
            intercept=leg_intercept,
            output=prefix.with_name(prefix.name + "_legibility.png")
        )
    
    print("Generating binned analysis plots...")
    if coverage_pairs:
        cov_bins = bin_analysis(coverage_pairs, bin_edges)
        plot_binned_analysis(
            cov_bins,
            ylabel="Mean Pressure Coverage Score",
            title="Coverage by MIN Benign Token Count Bin",
            output=prefix.with_name(prefix.name + "_coverage_bins.png")
        )
    
    if legibility_pairs:
        leg_bins = bin_analysis(legibility_pairs, bin_edges)
        plot_binned_analysis(
            leg_bins,
            ylabel="Mean Pressure Legibility Score",
            title="Legibility by MIN Benign Token Count Bin",
            output=prefix.with_name(prefix.name + "_legibility_bins.png")
        )
    
    print()
    print("Generating report...")
    report = generate_report(coverage_pairs, legibility_pairs, bin_edges)
    
    report_path = prefix.with_name(prefix.name + "_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"  → Saved: {report_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()

