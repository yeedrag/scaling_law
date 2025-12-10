"""
Plot baseline average length/accuracy vs pressure coverage/legibility.

Inputs:
  --baseline results/baseline_accuracy_samples.jsonl
      (fields: question, avg_token_count, accuracy)
  --benchmark results/rat_iter4_benchmark.json
      (pressure records with coverage/legibility)

Outputs:
  PNGs under --out-prefix:
    *_len_cov.png  : avg_token_count vs pressure coverage
    *_len_leg.png  : avg_token_count vs pressure legibility
    *_acc_cov.png  : baseline accuracy vs pressure coverage
    *_acc_leg.png  : baseline accuracy vs pressure legibility
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot baseline length/accuracy vs pressure coverage/legibility.")
    p.add_argument("--baseline", default="results/baseline_accuracy_samples.jsonl")
    p.add_argument("--benchmark", default="results/rat_iter4_benchmark.json")
    p.add_argument("--out-prefix", default="results/baseline_vs_pressure")
    return p.parse_args()


def load_baseline(path: Path) -> Dict[str, dict]:
    data: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec.get("question")
            if not q:
                continue
            avg_tok = rec.get("avg_token_count")
            acc = rec.get("accuracy")
            data[q] = {"avg_token_count": avg_tok, "accuracy": acc}
    return data


def load_pressure(path: Path) -> Dict[str, dict]:
    pressure: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    for rec in arr:
        if rec.get("variant") != "pressure":
            continue
        q = rec.get("question")
        if not q:
            continue
        pressure[q] = rec
    return pressure


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
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[pairs[k][0]] = avg_rank
            i = j + 1
        return r

    if len(xs) != len(ys):
        return float("nan")
    return pearson(ranks(xs), ranks(ys))


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
    baseline = load_baseline(Path(args.baseline))
    pressure = load_pressure(Path(args.benchmark))

    len_cov_x, len_cov_y = [], []
    len_leg_x, len_leg_y = [], []
    acc_cov_x, acc_cov_y = [], []
    acc_leg_x, acc_leg_y = [], []

    for q, b in baseline.items():
        p = pressure.get(q)
        if not p:
            continue
        cov = p.get("coverage")
        leg = p.get("legibility")
        avg_tok = b.get("avg_token_count")
        acc = b.get("accuracy")
        if cov is not None and avg_tok is not None:
            len_cov_x.append(avg_tok)
            len_cov_y.append(cov)
        if leg is not None and avg_tok is not None:
            len_leg_x.append(avg_tok)
            len_leg_y.append(leg)
        if cov is not None and acc is not None:
            acc_cov_x.append(acc)
            acc_cov_y.append(cov)
        if leg is not None and acc is not None:
            acc_leg_x.append(acc)
            acc_leg_y.append(leg)

    prefix = Path(args.out_prefix)
    scatter(len_cov_x, len_cov_y, "Avg token count (benign)", "Pressure coverage", "Avg length vs coverage", prefix.with_name(prefix.name + "_len_cov.png"))
    scatter(len_leg_x, len_leg_y, "Avg token count (benign)", "Pressure legibility", "Avg length vs legibility", prefix.with_name(prefix.name + "_len_leg.png"))
    scatter(acc_cov_x, acc_cov_y, "Baseline accuracy", "Pressure coverage", "Baseline accuracy vs coverage", prefix.with_name(prefix.name + "_acc_cov.png"))
    scatter(acc_leg_x, acc_leg_y, "Baseline accuracy", "Pressure legibility", "Baseline accuracy vs legibility", prefix.with_name(prefix.name + "_acc_leg.png"))

    def report(label, xs, ys):
        print(f"{label}: n={len(xs)}, pearson={pearson(xs, ys):.4f}, spearman={spearman(xs, ys):.4f}")

    report("len vs coverage", len_cov_x, len_cov_y)
    report("len vs legibility", len_leg_x, len_leg_y)
    report("acc vs coverage", acc_cov_x, acc_cov_y)
    report("acc vs legibility", acc_leg_x, acc_leg_y)


if __name__ == "__main__":
    main()

