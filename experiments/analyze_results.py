"""
Analyze results from experiments to compute key metrics.

Usage:
    python experiments/analyze_results.py \
        --benchmark results/rat_iter4_benchmark.jsonl \
        --pressure-samples results/pressure_samples.jsonl \
        --output results/analysis_summary.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze experiment results and compute metrics.")
    p.add_argument("--benchmark", default="results/rat_iter4_benchmark.jsonl", help="Benchmark JSONL from run_experiment.py")
    p.add_argument("--pressure-samples", default="results/pressure_samples.jsonl", help="Pressure samples JSONL from run_pressure_samples.py")
    p.add_argument("--output", default="results/analysis_summary.txt", help="Output text file")
    return p.parse_args()


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def analyze_benchmark(path: Path) -> Dict:
    """Analyze rat_iter4_benchmark.jsonl (benign vs pressure comparison)."""
    benign_records = []
    pressure_records = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            variant = rec.get("variant")
            if variant == "benign":
                benign_records.append(rec)
            elif variant == "pressure":
                pressure_records.append(rec)
    
    # Benign stats
    benign_correct = [1 if r.get("correct") else 0 for r in benign_records]
    benign_tokens = [safe_float(r.get("token_count")) for r in benign_records if r.get("token_count") is not None]
    
    # Pressure stats
    pressure_correct = [1 if r.get("correct") else 0 for r in pressure_records]
    pressure_tokens = [safe_float(r.get("token_count")) for r in pressure_records if r.get("token_count") is not None]
    pressure_coverage = [safe_float(r.get("coverage")) for r in pressure_records if r.get("coverage") is not None]
    pressure_legibility = [safe_float(r.get("legibility")) for r in pressure_records if r.get("legibility") is not None]
    
    return {
        "n_benign": len(benign_records),
        "n_pressure": len(pressure_records),
        "benign_accuracy": mean(benign_correct) if benign_correct else 0.0,
        "benign_tokens_mean": mean(benign_tokens) if benign_tokens else 0.0,
        "benign_tokens_std": stdev(benign_tokens) if len(benign_tokens) > 1 else 0.0,
        "pressure_accuracy": mean(pressure_correct) if pressure_correct else 0.0,
        "pressure_tokens_mean": mean(pressure_tokens) if pressure_tokens else 0.0,
        "pressure_tokens_std": stdev(pressure_tokens) if len(pressure_tokens) > 1 else 0.0,
        "pressure_coverage_mean": mean(pressure_coverage) if pressure_coverage else 0.0,
        "pressure_coverage_std": stdev(pressure_coverage) if len(pressure_coverage) > 1 else 0.0,
        "pressure_legibility_mean": mean(pressure_legibility) if pressure_legibility else 0.0,
        "pressure_legibility_std": stdev(pressure_legibility) if len(pressure_legibility) > 1 else 0.0,
    }


def analyze_pressure_samples(path: Path) -> Dict:
    """Analyze pressure_samples.jsonl (multiple samples per question)."""
    questions = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            questions.append(rec)
    
    # Per-question aggregates
    accuracies = [safe_float(rec.get("accuracy")) for rec in questions]
    avg_coverages = [safe_float(rec.get("avg_coverage")) for rec in questions]
    avg_legibilities = [safe_float(rec.get("avg_legibility")) for rec in questions]
    avg_tokens = [safe_float(rec.get("avg_token_count")) for rec in questions]
    
    # Count total samples
    total_samples = sum(len(rec.get("samples", [])) for rec in questions)
    
    return {
        "n_questions": len(questions),
        "total_samples": total_samples,
        "samples_per_question": total_samples / len(questions) if questions else 0,
        "accuracy_mean": mean(accuracies) if accuracies else 0.0,
        "accuracy_std": stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "coverage_mean": mean(avg_coverages) if avg_coverages else 0.0,
        "coverage_std": stdev(avg_coverages) if len(avg_coverages) > 1 else 0.0,
        "legibility_mean": mean(avg_legibilities) if avg_legibilities else 0.0,
        "legibility_std": stdev(avg_legibilities) if len(avg_legibilities) > 1 else 0.0,
        "tokens_mean": mean(avg_tokens) if avg_tokens else 0.0,
        "tokens_std": stdev(avg_tokens) if len(avg_tokens) > 1 else 0.0,
    }


def format_report(benchmark_stats: Dict, pressure_stats: Dict) -> str:
    """Format analysis results as a readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT RESULTS ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("1. BENCHMARK: Benign vs Pressure (rat_iter4_benchmark.jsonl)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Number of benign samples:   {benchmark_stats['n_benign']}")
    lines.append(f"Number of pressure samples: {benchmark_stats['n_pressure']}")
    lines.append("")
    
    lines.append("--- BENIGN (Standard Reasoning) ---")
    lines.append(f"  Accuracy:      {benchmark_stats['benign_accuracy']:.3f}")
    lines.append(f"  Token count:   {benchmark_stats['benign_tokens_mean']:.1f} ± {benchmark_stats['benign_tokens_std']:.1f}")
    lines.append("")
    
    lines.append("--- PRESSURE (Adversarial Prompt) ---")
    lines.append(f"  Accuracy:      {benchmark_stats['pressure_accuracy']:.3f}")
    lines.append(f"  Token count:   {benchmark_stats['pressure_tokens_mean']:.1f} ± {benchmark_stats['pressure_tokens_std']:.1f}")
    lines.append(f"  Coverage:      {benchmark_stats['pressure_coverage_mean']:.3f} ± {benchmark_stats['pressure_coverage_std']:.3f}")
    lines.append(f"  Legibility:    {benchmark_stats['pressure_legibility_mean']:.3f} ± {benchmark_stats['pressure_legibility_std']:.3f}")
    lines.append("")
    
    acc_drop = benchmark_stats['benign_accuracy'] - benchmark_stats['pressure_accuracy']
    token_reduction = benchmark_stats['benign_tokens_mean'] - benchmark_stats['pressure_tokens_mean']
    lines.append("--- COMPARISON ---")
    lines.append(f"  Accuracy drop:     {acc_drop:+.3f} ({acc_drop/benchmark_stats['benign_accuracy']*100:+.1f}%)")
    lines.append(f"  Token reduction:   {token_reduction:+.1f} ({token_reduction/benchmark_stats['benign_tokens_mean']*100:+.1f}%)")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("2. PRESSURE SAMPLES: Multiple samples per question (pressure_samples.jsonl)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Number of questions:      {pressure_stats['n_questions']}")
    lines.append(f"Total samples:            {pressure_stats['total_samples']}")
    lines.append(f"Samples per question:     {pressure_stats['samples_per_question']:.1f}")
    lines.append("")
    
    lines.append("--- AGGREGATED METRICS (averaged across questions) ---")
    lines.append(f"  Accuracy:      {pressure_stats['accuracy_mean']:.3f} ± {pressure_stats['accuracy_std']:.3f}")
    lines.append(f"  Coverage:      {pressure_stats['coverage_mean']:.3f} ± {pressure_stats['coverage_std']:.3f}")
    lines.append(f"  Legibility:    {pressure_stats['legibility_mean']:.3f} ± {pressure_stats['legibility_std']:.3f}")
    lines.append(f"  Token count:   {pressure_stats['tokens_mean']:.1f} ± {pressure_stats['tokens_std']:.1f}")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("3. KEY INSIGHTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"→ Pressure prompting reduces monitorability:")
    lines.append(f"  • Coverage:    {benchmark_stats['pressure_coverage_mean']:.3f} (lower = less monitorable)")
    lines.append(f"  • Legibility:  {benchmark_stats['pressure_legibility_mean']:.3f} (lower = less legible)")
    lines.append("")
    lines.append(f"→ Pressure prompting reduces token usage:")
    lines.append(f"  • {token_reduction:.0f} fewer tokens ({abs(token_reduction/benchmark_stats['benign_tokens_mean']*100):.1f}% reduction)")
    lines.append("")
    if acc_drop > 0.01:
        lines.append(f"→ Accuracy degradation: {acc_drop:.3f} ({abs(acc_drop/benchmark_stats['benign_accuracy']*100):.1f}% drop)")
    else:
        lines.append(f"→ Accuracy maintained under pressure")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("4. NEXT STEPS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Generate plots with:")
    lines.append("  python experiments/plot_coverage_vs_tokens.py --input results/rat_iter4_benchmark.jsonl")
    lines.append("  python experiments/plot_token_lengths.py --input results/rat_iter4_benchmark.jsonl")
    lines.append("")
    
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    
    benchmark_path = Path(args.benchmark)
    pressure_path = Path(args.pressure_samples)
    
    print(f"Loading benchmark data from {benchmark_path}...")
    benchmark_stats = analyze_benchmark(benchmark_path)
    
    print(f"Loading pressure samples data from {pressure_path}...")
    pressure_stats = analyze_pressure_samples(pressure_path)
    
    report = format_report(benchmark_stats, pressure_stats)
    
    # Print to console
    print("\n" + report)
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nAnalysis saved to: {output_path}")


if __name__ == "__main__":
    main()





