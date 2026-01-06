"""
Full correlation analysis for monitorability forecasting paper.

Measures:
1. Mean benign length → mean adversarial coverage/legibility
2. Min benign length → mean adversarial coverage/legibility  
3. Mean adversarial length → mean adversarial coverage/legibility
4. Mean benign length → mean adversarial length

Outputs: Scatter plots with regression lines + summary table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_benign_data(path: Path) -> dict:
    """Load benign samples, keyed by question."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec.get("question")
            if not q:
                continue
            token_counts = rec.get("token_counts", [])
            data[q] = {
                "L_mean": rec.get("avg_token_count", mean(token_counts) if token_counts else 0),
                "L_min": min(token_counts) if token_counts else 0,
                "token_counts": token_counts,
            }
    return data


def load_pressure_data(path: Path) -> dict:
    """Load pressure samples, keyed by question."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec.get("question")
            if not q:
                continue
            
            # Get token counts from samples
            samples = rec.get("samples", [])
            token_counts = [s.get("token_count", 0) for s in samples]
            
            data[q] = {
                "C_adv": rec.get("avg_coverage", 0),
                "G_adv": rec.get("avg_legibility", 0),
                "L_adv_mean": rec.get("avg_token_count", mean(token_counts) if token_counts else 0),
                "token_counts": token_counts,
            }
    return data


def compute_correlation(xs: List[float], ys: List[float]) -> dict:
    """Compute correlation statistics."""
    xs = np.array(xs)
    ys = np.array(ys)
    
    pearson_r, pearson_p = stats.pearsonr(xs, ys)
    spearman_r, spearman_p = stats.spearmanr(xs, ys)
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    
    return {
        "n": len(xs),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err,
    }


def plot_correlation(xs: List[float], ys: List[float], 
                     xlabel: str, ylabel: str, title: str,
                     stats_dict: dict, output: Path) -> None:
    """Create scatter plot with regression line."""
    xs = np.array(xs)
    ys = np.array(ys)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Scatter
    ax.scatter(xs, ys, alpha=0.35, s=18, color='#2E86AB', edgecolors='none')
    
    # Regression line
    x_range = np.linspace(xs.min(), xs.max(), 100)
    y_pred = stats_dict["slope"] * x_range + stats_dict["intercept"]
    ax.plot(x_range, y_pred, 'r-', linewidth=2, 
            label=f'y = {stats_dict["slope"]:.4f}x + {stats_dict["intercept"]:.2f}')
    
    # Stats annotation
    stats_text = (
        f'n = {stats_dict["n"]}\n'
        f'Pearson r = {stats_dict["pearson_r"]:.3f} (p = {stats_dict["pearson_p"]:.2e})\n'
        f'Spearman ρ = {stats_dict["spearman_r"]:.3f} (p = {stats_dict["spearman_p"]:.2e})\n'
        f'R² = {stats_dict["r_squared"]:.4f}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"  → Saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benign-samples", default="results/benign_samples.jsonl")
    parser.add_argument("--pressure-samples", default="results/pressure_samples.jsonl")
    parser.add_argument("--output-dir", default="results/correlation_analysis")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    benign = load_benign_data(Path(args.benign_samples))
    pressure = load_pressure_data(Path(args.pressure_samples))
    
    # Find common questions
    common_qs = set(benign.keys()) & set(pressure.keys())
    print(f"  Common questions: {len(common_qs)}")
    
    # Extract all variables
    L_mean = []      # Mean benign length
    L_min = []       # Min benign length
    L_adv_mean = []  # Mean adversarial length
    C_adv = []       # Mean adversarial coverage
    G_adv = []       # Mean adversarial legibility
    
    for q in common_qs:
        L_mean.append(benign[q]["L_mean"])
        L_min.append(benign[q]["L_min"])
        L_adv_mean.append(pressure[q]["L_adv_mean"])
        C_adv.append(pressure[q]["C_adv"])
        G_adv.append(pressure[q]["G_adv"])
    
    # Define all correlation pairs
    analyses = [
        # Primary analyses (1-2)
        ("1a", L_mean, C_adv, "Mean Benign Length (tokens)", "Mean Adversarial Coverage",
         r"$L_{\mathrm{mean}}$ vs $C_{\mathrm{adv}}$", "L_mean_vs_C_adv"),
        ("1b", L_mean, G_adv, "Mean Benign Length (tokens)", "Mean Adversarial Legibility",
         r"$L_{\mathrm{mean}}$ vs $G_{\mathrm{adv}}$", "L_mean_vs_G_adv"),
        ("2a", L_min, C_adv, "Min Benign Length (tokens)", "Mean Adversarial Coverage",
         r"$L_{\mathrm{min}}$ vs $C_{\mathrm{adv}}$", "L_min_vs_C_adv"),
        ("2b", L_min, G_adv, "Min Benign Length (tokens)", "Mean Adversarial Legibility",
         r"$L_{\mathrm{min}}$ vs $G_{\mathrm{adv}}$", "L_min_vs_G_adv"),
        # Additional analyses (3-4)
        ("3a", L_adv_mean, C_adv, "Mean Adversarial Length (tokens)", "Mean Adversarial Coverage",
         r"$L^{\mathrm{adv}}_{\mathrm{mean}}$ vs $C_{\mathrm{adv}}$", "L_adv_vs_C_adv"),
        ("3b", L_adv_mean, G_adv, "Mean Adversarial Length (tokens)", "Mean Adversarial Legibility",
         r"$L^{\mathrm{adv}}_{\mathrm{mean}}$ vs $G_{\mathrm{adv}}$", "L_adv_vs_G_adv"),
        ("4", L_mean, L_adv_mean, "Mean Benign Length (tokens)", "Mean Adversarial Length (tokens)",
         r"$L_{\mathrm{mean}}$ vs $L^{\mathrm{adv}}_{\mathrm{mean}}$", "L_mean_vs_L_adv"),
    ]
    
    # Compute and store results
    results = []
    
    print("\nComputing correlations and generating plots...")
    print("=" * 80)
    
    for label, xs, ys, xlabel, ylabel, title, filename in analyses:
        stats_dict = compute_correlation(xs, ys)
        results.append((label, title, stats_dict))
        
        # Generate plot
        plot_correlation(xs, ys, xlabel, ylabel, title, stats_dict, 
                        out_dir / f"{filename}.png")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("CORRELATION SUMMARY TABLE")
    print("=" * 100)
    print(f"{'#':<4} {'Relationship':<45} {'Pearson r':>12} {'Spearman ρ':>12} {'p-value':>12} {'R²':>10}")
    print("-" * 100)
    
    for label, title, s in results:
        # Clean up LaTeX for terminal display
        clean_title = title.replace(r"$", "").replace(r"\mathrm{", "").replace("}", "").replace("_", "_")
        print(f"{label:<4} {clean_title:<45} {s['pearson_r']:>12.4f} {s['spearman_r']:>12.4f} {s['p_value']:>12.2e} {s['r_squared']:>10.4f}")
    
    print("=" * 100)
    
    # Save LaTeX table
    latex_table = r"""
\begin{table}[h]
\centering
\caption{Correlation analysis between benign trace characteristics and adversarial monitorability. 
All correlations computed across $n=""" + str(len(common_qs)) + r"""$ questions, each with 20 benign and 20 adversarial rollouts.}
\label{tab:correlations}
\begin{tabular}{clcccc}
\toprule
\# & Relationship & Pearson $r$ & Spearman $\rho$ & $p$-value & $R^2$ \\
\midrule
"""
    
    for label, title, s in results:
        sig = "***" if s['p_value'] < 0.001 else ("**" if s['p_value'] < 0.01 else ("*" if s['p_value'] < 0.05 else ""))
        latex_table += f"{label} & {title} & {s['pearson_r']:.3f}{sig} & {s['spearman_r']:.3f} & {s['p_value']:.2e} & {s['r_squared']:.4f} \\\\\n"
    
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    latex_path = out_dir / "correlation_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"\n  → LaTeX table saved: {latex_path}")
    
    # Save detailed report
    report_path = out_dir / "full_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FULL CORRELATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data: {len(common_qs)} questions × 20 samples each\n")
        f.write(f"Benign samples: {args.benign_samples}\n")
        f.write(f"Pressure samples: {args.pressure_samples}\n\n")
        
        for label, title, s in results:
            f.write(f"\n{label}. {title}\n")
            f.write("-" * 60 + "\n")
            f.write(f"  n = {s['n']}\n")
            f.write(f"  Pearson r = {s['pearson_r']:.4f} (p = {s['pearson_p']:.2e})\n")
            f.write(f"  Spearman ρ = {s['spearman_r']:.4f} (p = {s['spearman_p']:.2e})\n")
            f.write(f"  Regression: y = {s['slope']:.6f}x + {s['intercept']:.4f}\n")
            f.write(f"  R² = {s['r_squared']:.4f}\n")
            
            if s['p_value'] < 0.001:
                f.write("  → HIGHLY SIGNIFICANT (p < 0.001)\n")
            elif s['p_value'] < 0.05:
                f.write("  → SIGNIFICANT (p < 0.05)\n")
            else:
                f.write("  → NOT SIGNIFICANT\n")
    
    print(f"  → Full report saved: {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()




