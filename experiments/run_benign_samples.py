"""
Run benign prompt multiple times per question to collect token counts.

For each question:
  - Generate `--samples` benign responses.
  - Extract answer and check correctness via simple string matching.
  - Aggregate: accuracy, mean token_count.

NO OpenAI judging needed - benign samples have perfect coverage/legibility by definition.

Outputs: JSONL where each line is a question with its samples and aggregates.

Usage:
    PYTHONPATH=. python experiments/run_benign_samples.py \
        --config config.yaml \
        --samples 20 \
        --limit 200 \
        --output results/benign_samples.jsonl \
        --chunk-size 128 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Set

from tqdm import tqdm

from src.actor import ReasoningActor, extract_answer
from src.config import load_config
from src.utils import set_global_seed
from experiments.min_token_search import answers_match
from experiments import rat2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benign multi-sample runner (no OpenAI judging).")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    p.add_argument("--limit", type=int, default=None, help="Limit questions.")
    p.add_argument("--samples", type=int, default=20, help="Samples per question.")
    p.add_argument("--output", default="results/benign_samples.jsonl", help="Output JSONL path.")
    p.add_argument("--seed", type=int, default=42, help="Seed; -1 to disable seeding.")
    p.add_argument("--chunk-size", type=int, default=128, help="Process questions in chunks to allow incremental writes.")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    cfg = load_config(args.config)
    
    # Set seed
    if args.seed >= 0:
        set_global_seed(args.seed)
        logging.info("Set global seed to %s", args.seed)
    
    # Load questions
    questions_path = getattr(cfg, "questions_path", None)
    if not questions_path:
        raise SystemExit("config.experiment.questions_path is required.")
    problems = rat2.load_filtered_problems(questions_path, args.limit)
    logging.info("Loaded %s problems from %s", len(problems), questions_path)
    
    # Initialize actor (benign)
    actor = ReasoningActor(
        model_id=cfg.qwen_model_id,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        max_model_len=cfg.qwen_max_model_len,
        tensor_parallel_size=None,
        max_num_seqs=getattr(cfg, "actor_batch_size", None) or None,
        seed=None if args.seed < 0 else args.seed,
    )
    
    # Output
    out_path = Path(args.output)
    ensure_dir(out_path)
    
    # Resume support: load seen questions
    seen: Set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    q = rec.get("question")
                    if q:
                        seen.add(q)
                except json.JSONDecodeError:
                    continue
        logging.info("Resuming; found %s completed questions", len(seen))
    
    # Filter pending problems
    pending = [(i, p) for i, p in enumerate(problems) if p.question not in seen]
    if not pending:
        logging.info("All questions already completed!")
        return
    
    logging.info("Processing %s pending questions", len(pending))
    
    # Process in chunks
    for start in range(0, len(pending), args.chunk_size):
        chunk = pending[start : start + args.chunk_size]
        prompts: List[str] = []
        owners: List[int] = []
        
        for idx, p in chunk:
            # Use benign prompt (standard actor format with chat template)
            formatted = actor._apply_chat_template(p.question)
            for _ in range(args.samples):
                prompts.append(formatted)
                owners.append(idx)
        
        logging.info("Chunk %s/%s: generating %s prompts for %s questions", 
                     start // args.chunk_size + 1, 
                     (len(pending) + args.chunk_size - 1) // args.chunk_size, 
                     len(prompts), 
                     len(chunk))
        
        outputs = actor.llm.generate(prompts, sampling_params=actor.sampling_params, use_tqdm=True)
        
        # Group samples by question
        per_q_samples: Dict[int, List[dict]] = defaultdict(list)
        for out, idx in zip(outputs, owners):
            comp = out.outputs[0]
            text = comp.text.strip()
            full = f"<think>\n{text}"
            ans = extract_answer(full)
            p = problems[idx]
            
            # Simple correctness check via string matching (no OpenAI needed)
            correct = answers_match(ans, p.answer) if ans else False
            
            per_q_samples[idx].append({
                "answer": ans,
                "token_count": len(comp.token_ids),
                "correct": correct,
            })
        
        # Aggregate and write to file
        with out_path.open("a", encoding="utf-8") as f:
            for idx, samples in per_q_samples.items():
                p = problems[idx]
                
                # Compute aggregates
                correct_count = sum(1 for s in samples if s.get("correct"))
                accuracy = correct_count / len(samples) if samples else 0.0
                avg_token_count = mean([s.get("token_count", 0) for s in samples]) if samples else 0.0
                
                record = {
                    "question": p.question,
                    "expected": p.answer,
                    "n_samples": len(samples),
                    "accuracy": accuracy,
                    "avg_token_count": avg_token_count,
                    # Don't store full samples to save space - just aggregates
                    "token_counts": [s["token_count"] for s in samples],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logging.info("Wrote chunk %s questions to %s", len(per_q_samples), out_path)
    
    logging.info("Done! Results saved to %s", out_path)


if __name__ == "__main__":
    main()
