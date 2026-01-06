"""
Forecast monitorability (coverage/legibility) from benign CoT characteristics.

Research Question: Can we predict how monitorable a pressure response will be
based on properties of the benign CoT trace?

Data Sources:
  - Benign token counts: from benign_samples.jsonl (mean of 20 samples)
  - Pressure coverage/legibility: from pressure_samples.jsonl (mean of 20 samples)
  
This gives robust estimates on both sides for strong forecasting.

Analysis:
  1. Correlation between benign token count and pressure coverage/legibility
  2. Regression coefficients for forecasting
  3. Binned analysis to show trends
  4. Scatter plots with trend lines

Usage:
    PYTHONPATH=. python experiments/forecast_monitorability.py \
        --benign-samples results/benign_samples.jsonl \
        --pressure-samples results/pressure_samples.jsonl \
        --output results/monitorability_forecast
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
    p = argparse.ArgumentParser(description="Forecast monitorability from benign CoT characteristics")
    p.add_argument("--benign-samples", default="results/benign_samples.jsonl", 
                   help="Benign samples JSONL (for robust mean token counts)")
    p.add_argument("--pressure-samples", default="results/pressure_samples.jsonl",
                   help="Pressure samples JSONL (for robust mean coverage/legibility)")
    p.add_argument("--output", default="results/monitorability_forecast", 
                   help="Output prefix for plots and report")
    p.add_argument("--bins", default="0,150,300,450,600,1000,1500", 
                   help="Bin edges for benign token counts")
    return p.parse_args()


def load_benign_data(path: Path) -> dict:
    """Load benign samples (mean token counts from 20 samples), keyed by question."""
    benign_map = {}
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec.get("question")
            if not q:
                continue
            
            # This has avg_token_count from 20 samples
            benign_map[q] = rec
    
    return benign_map


def load_pressure_samples(path: Path) -> dict:
    """Load pressure samples with mean coverage/legibility (20 samples per question)."""
    pressure_map = {}
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec.get("question")
            if not q:
                continue
            
            # This has avg_coverage, avg_legibility from 20 samples
            pressure_map[q] = rec
    
    return pressure_map


def extract_pairs(benign_map: dict, pressure_map: dict, y_key: str) -> List[Tuple[float, float]]:
    """Extract (avg_benign_tokens, avg_pressure_metric) pairs.
    
    Args:
        benign_map: Dict of benign_samples records with avg_token_count
        pressure_map: Dict of pressure_samples records with avg_coverage, avg_legibility
        y_key: Either "avg_coverage" or "avg_legibility"
    """
    pairs = []
    for q, b in benign_map.items():
        p = pressure_map.get(q)
        if not p:
            continue
        
        benign_tok = b.get("avg_token_count")
        pressure_val = p.get(y_key)
        
        if benign_tok is not None and pressure_val is not None:
            pairs.append((float(benign_tok), float(pressure_val)))
    
    return pairs


def pearson(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation."""
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
    """Compute Spearman rank correlation."""
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
    """Simple linear regression: y = slope * x + intercept."""
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
    """Bin pairs by x value and compute statistics for each bin."""
    bins = []
    edges = sorted(bin_edges)
    
    # Create bins [lo, hi)
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
    
    # Last bin [last_edge, inf)
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
    """Create scatter plot with regression line."""
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Scatter plot
    ax.scatter(xs, ys, alpha=0.4, s=20, color='steelblue', label='Data')
    
    # Regression line
    if not np.isnan(slope):
        x_range = np.linspace(min(xs), max(xs), 100)
        y_pred = slope * x_range + intercept
        ax.plot(x_range, y_pred, 'r-', linewidth=2, 
                label=f'y = {slope:.4f}x + {intercept:.2f}')
    
    ax.set_xlabel('Benign CoT Token Count (mean of 20 samples)', fontsize=11)
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
    """Create bar chart of binned analysis with error bars."""
    # Filter bins with data
    valid_bins = [b for b in bins if b["mean"] is not None and b["count"] > 0]
    
    if not valid_bins:
        print(f"  ⚠ No valid bins for {output}")
        return
    
    labels = [b["range"] for b in valid_bins]
    means = [b["mean"] for b in valid_bins]
    stds = [b["std"] if b["std"] is not None else 0 for b in valid_bins]
    counts = [b["count"] for b in valid_bins]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color gradient based on mean value (darker = lower)
    colors = plt.cm.RdYlGn([m / 5.0 for m in means])  # Scale 0-5 to 0-1
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, 
                  edgecolor='black', linewidth=0.8, alpha=0.85)
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.1,
                f'n={count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Benign Token Count Range', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylim(0, 5.5)  # Coverage/legibility scale 0-5
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
    """Generate text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("MONITORABILITY FORECASTING ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Research Question: Can benign CoT trace length predict")
    lines.append("                   how monitorable the pressure response will be?")
    lines.append("")
    lines.append("Data: Mean of 20 benign samples → Mean of 20 pressure samples")
    lines.append("      (Robust estimates on both X and Y axes)")
    lines.append("")
    
    # Coverage analysis
    if coverage_pairs:
        cov_xs = [p[0] for p in coverage_pairs]
        cov_ys = [p[1] for p in coverage_pairs]
        cov_slope, cov_intercept = linear_regression(cov_xs, cov_ys)
        cov_pearson = pearson(cov_xs, cov_ys)
        cov_spearman = spearman(cov_xs, cov_ys)
        
        lines.append("=" * 80)
        lines.append("1. AVG BENIGN TOKEN COUNT → AVG PRESSURE COVERAGE")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Sample size: {len(coverage_pairs)} question pairs")
        lines.append(f"             ({len(coverage_pairs) * 20} benign + {len(coverage_pairs) * 20} pressure samples)")
        lines.append("")
        lines.append("--- Correlation ---")
        lines.append(f"  Pearson correlation:  {cov_pearson:.4f}")
        lines.append(f"  Spearman correlation: {cov_spearman:.4f}")
        lines.append("")
        lines.append("--- Linear Regression ---")
        lines.append(f"  coverage = {cov_slope:.6f} × benign_tokens + {cov_intercept:.3f}")
        lines.append("")
        
        if cov_pearson < -0.3:
            lines.append("  ✓ STRONG NEGATIVE CORRELATION: Longer benign traces → lower coverage")
        elif cov_pearson < -0.1:
            lines.append("  → MODERATE NEGATIVE CORRELATION: Some predictive power")
        else:
            lines.append("  ✗ WEAK CORRELATION: Benign length may not predict coverage well")
        lines.append("")
        
        # Binned analysis
        bins = bin_analysis(coverage_pairs, bin_edges)
        lines.append("--- Binned Analysis ---")
        lines.append(f"{'Token Range':<20} {'Count':<8} {'Mean Coverage':<15} {'Std Dev':<10}")
        lines.append("-" * 60)
        for b in bins:
            mean_str = f"{b['mean']:.3f}" if b['mean'] is not None else "N/A"
            std_str = f"±{b['std']:.3f}" if b['std'] is not None else ""
            lines.append(f"{b['range']:<20} {b['count']:<8} {mean_str:<15} {std_str:<10}")
        lines.append("")
    
    # Legibility analysis
    if legibility_pairs:
        leg_xs = [p[0] for p in legibility_pairs]
        leg_ys = [p[1] for p in legibility_pairs]
        leg_slope, leg_intercept = linear_regression(leg_xs, leg_ys)
        leg_pearson = pearson(leg_xs, leg_ys)
        leg_spearman = spearman(leg_xs, leg_ys)
        
        lines.append("=" * 80)
        lines.append("2. AVG BENIGN TOKEN COUNT → AVG PRESSURE LEGIBILITY")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Sample size: {len(legibility_pairs)} question pairs")
        lines.append(f"             ({len(legibility_pairs) * 20} benign + {len(legibility_pairs) * 20} pressure samples)")
        lines.append("")
        lines.append("--- Correlation ---")
        lines.append(f"  Pearson correlation:  {leg_pearson:.4f}")
        lines.append(f"  Spearman correlation: {leg_spearman:.4f}")
        lines.append("")
        lines.append("--- Linear Regression ---")
        lines.append(f"  legibility = {leg_slope:.6f} × benign_tokens + {leg_intercept:.3f}")
        lines.append("")
        
        if leg_pearson < -0.3:
            lines.append("  ✓ STRONG NEGATIVE CORRELATION: Longer benign traces → lower legibility")
        elif leg_pearson < -0.1:
            lines.append("  → MODERATE NEGATIVE CORRELATION: Some predictive power")
        else:
            lines.append("  ✗ WEAK CORRELATION: Benign length may not predict legibility well")
        lines.append("")
        
        # Binned analysis
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
    lines.append("3. KEY FINDINGS")
    lines.append("=" * 80)
    lines.append("")
    
    if coverage_pairs:
        cov_xs = [p[0] for p in coverage_pairs]
        cov_ys = [p[1] for p in coverage_pairs]
        cov_pearson = pearson(cov_xs, cov_ys)
        
        if cov_pearson < -0.2:
            lines.append("✓ Benign trace length IS a predictor of monitorability loss")
            lines.append("  → Longer benign reasoning → lower coverage under pressure")
        else:
            lines.append("✗ Benign trace length is NOT a strong predictor")
            lines.append("  → Monitorability loss may depend on other factors")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    
    print("Loading benign samples (mean token counts from 20 samples each)...")
    benign_map = load_benign_data(Path(args.benign_samples))
    print(f"  Benign records: {len(benign_map)}")
    
    print("Loading pressure samples (mean coverage/legibility from 20 samples each)...")
    pressure_map = load_pressure_samples(Path(args.pressure_samples))
    print(f"  Pressure records: {len(pressure_map)}")
    print()
    
    # Extract paired data
    coverage_pairs = extract_pairs(benign_map, pressure_map, "avg_coverage")
    legibility_pairs = extract_pairs(benign_map, pressure_map, "avg_legibility")
    
    print(f"Extracted {len(coverage_pairs)} coverage pairs, {len(legibility_pairs)} legibility pairs")
    print()
    
    # Parse bins
    bin_edges = [int(x) for x in args.bins.split(",")]
    
    # Regression
    print("Computing correlations and regressions...")
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
    
    # Generate plots
    print("Generating plots...")
    prefix = Path(args.output)
    
    if coverage_pairs:
        plot_forecast(
            coverage_pairs,
            ylabel="Pressure Coverage Score",
            title="Forecasting Coverage from Benign CoT Length",
            slope=cov_slope,
            intercept=cov_intercept,
            output=prefix.with_name(prefix.name + "_coverage.png")
        )
    
    if legibility_pairs:
        plot_forecast(
            legibility_pairs,
            ylabel="Pressure Legibility Score",
            title="Forecasting Legibility from Benign CoT Length",
            slope=leg_slope,
            intercept=leg_intercept,
            output=prefix.with_name(prefix.name + "_legibility.png")
        )
    
    # Generate binned analysis plots
    print("Generating binned analysis plots...")
    if coverage_pairs:
        cov_bins = bin_analysis(coverage_pairs, bin_edges)
        plot_binned_analysis(
            cov_bins,
            ylabel="Mean Pressure Coverage Score",
            title="Coverage by Benign Token Count Bin\n(Lower = Less Monitorable)",
            output=prefix.with_name(prefix.name + "_coverage_bins.png")
        )
    
    if legibility_pairs:
        leg_bins = bin_analysis(legibility_pairs, bin_edges)
        plot_binned_analysis(
            leg_bins,
            ylabel="Mean Pressure Legibility Score",
            title="Legibility by Benign Token Count Bin",
            output=prefix.with_name(prefix.name + "_legibility_bins.png")
        )
    
    # Generate report
    print()
    print("Generating analysis report...")
    report = generate_report(coverage_pairs, legibility_pairs, bin_edges)
    
    # Save report
    report_path = prefix.with_name(prefix.name + "_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"  → Saved: {report_path}")
    print()
    
    # Print to console
    print(report)


if __name__ == "__main__":
    main()

