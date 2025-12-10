"""
Correlate min-token-needed with coverage/legibility from rat_iter4_benchmark.json.

Inputs:
  --min-tokens results/min_tokens_samples.jsonl        (from sample_min_tokens.py)
  --benchmark  results/rat_iter4_benchmark.json        (pressure/benign evals)

Outputs:
  Prints Pearson/Spearman correlations for:
    min_token_correct vs coverage (pressure only)
    min_token_correct vs legibility (pressure only)
    min_token_correct vs benign token_count (optional)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate min tokens with coverage/legibility.")
    p.add_argument("--min-tokens", default="results/min_tokens_samples.jsonl")
    p.add_argument("--benchmark", default="results/rat_iter4_benchmark.json")
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


def correlate(name: str, xs: List[float], ys: List[float]) -> None:
    print(f"{name}: n={len(xs)}")
    print(f"  Pearson:  {pearson(xs, ys):.4f}")
    print(f"  Spearman: {spearman(xs, ys):.4f}")


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

    correlate("min_tokens vs pressure coverage", cov_x, cov_y)
    correlate("min_tokens vs pressure legibility", leg_x, leg_y)
    correlate("min_tokens vs benign token_count", benign_tok_x, benign_tok_y)


if __name__ == "__main__":
    main()

