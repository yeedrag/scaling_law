"""
Plot min_token_correct vs coverage/legibility and benign token count.

Usage:
    PYTHONPATH=. python experiments/plot_min_tokens_vs_scores.py \
        --min-tokens results/min_tokens_samples.jsonl \
        --benchmark results/rat_iter4_benchmark.json \
        --out-prefix results/min_tokens_corr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot min tokens vs coverage/legibility.")
    p.add_argument("--min-tokens", default="results/min_tokens_samples.jsonl")
    p.add_argument("--benchmark", default="results/rat_iter4_benchmark.json")
    p.add_argument("--out-prefix", default="results/min_tokens_corr")
    return p.parse_args()


def load_min_tokens(path: Path) -> Dict[str, int]:
    data: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            mt = rec.get("min_token_correct")
            q = rec.get("question")
            if q is not None and mt is not None:
                data[q] = mt
    return data


def load_benchmark(path: Path) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    benign: Dict[str, dict] = {}
    pressure: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    for rec in arr:
        q = rec.get("question")
        if not q:
            continue
        variant = rec.get("variant")
        if variant == "benign":
            benign[q] = rec
        elif variant == "pressure":
            pressure[q] = rec
    return benign, pressure


def scatter(xs: List[float], ys: List[float], xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, alpha=0.5, s=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    min_tokens = load_min_tokens(Path(args.min_tokens))
    benign, pressure = load_benchmark(Path(args.benchmark))

    cov_x, cov_y = [], []
    leg_x, leg_y = [], []
    benign_tok_x, benign_tok_y = [], []

    for q, mt in min_tokens.items():
        p_rec = pressure.get(q)
        b_rec = benign.get(q)
        if p_rec:
            cov = p_rec.get("coverage")
            leg = p_rec.get("legibility")
            if cov is not None:
                cov_x.append(mt)
                cov_y.append(cov)
            if leg is not None:
                leg_x.append(mt)
                leg_y.append(leg)
        if b_rec:
            tok = b_rec.get("token_count")
            if tok is not None:
                benign_tok_x.append(mt)
                benign_tok_y.append(tok)

    prefix = Path(args.out_prefix)
    scatter(cov_x, cov_y, "min_token_correct", "pressure coverage", "Min tokens vs coverage", prefix.with_name(prefix.name + "_cov.png"))
    scatter(leg_x, leg_y, "min_token_correct", "pressure legibility", "Min tokens vs legibility", prefix.with_name(prefix.name + "_leg.png"))
    scatter(benign_tok_x, benign_tok_y, "min_token_correct", "benign token_count", "Min tokens vs benign token_count", prefix.with_name(prefix.name + "_benign_tokens.png"))


if __name__ == "__main__":
    main()

