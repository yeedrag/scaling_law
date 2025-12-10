"""
Sample baseline (benign) prompt multiple times per question with no token
budget restriction and report accuracy.

Usage:
    PYTHONPATH=. python experiments/sample_accuracy_baseline.py \
        --config config.yaml \
        --samples 20 \
        --limit 200 \
        --output results/baseline_accuracy_samples.jsonl \
        --seed -1

Notes:
    - Uses BASELINE_ACTOR_PROMPT.
    - No pressure prefix, no token budget note.
    - If --seed >= 0 we set the global seed and pass it to the actor; otherwise
      sampling is stochastic.
    - Batches across questions using actor_batch_size from config or --batch-size.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from experiments.run_experiment import BASELINE_ACTOR_PROMPT
from experiments.min_token_search import extract_answer_loose, answers_match
from src.actor import ReasoningActor
from src.config import load_config
from src.utils import set_global_seed
from experiments import rat2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample baseline prompt and report accuracy per question.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    p.add_argument("--samples", type=int, default=10, help="Samples per question.")
    p.add_argument("--limit", type=int, default=None, help="Limit number of questions.")
    p.add_argument("--output", default="results/baseline_accuracy_samples.jsonl", help="Output JSONL path.")
    p.add_argument("--batch-size", type=int, default=None, help="Prompt batch size for vLLM (falls back to config.actor_batch_size).")
    p.add_argument("--seed", type=int, default=-1, help="Seed; -1 to disable seeding.")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_prompt(tokenizer, question: str) -> str:
    user = BASELINE_ACTOR_PROMPT.format(question=question)
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.seed >= 0:
        set_global_seed(args.seed)

    model_id = cfg.qwen_model_id or "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    temperature = cfg.temperature or 0.6
    max_tokens = cfg.max_tokens or 2048
    max_model_len = cfg.qwen_max_model_len or 4096
    actor_batch_size = args.batch_size or getattr(cfg, "actor_batch_size", None) or None

    actor = ReasoningActor(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        tensor_parallel_size=None,
        max_num_seqs=actor_batch_size,
        seed=None if args.seed < 0 else args.seed,
    )

    questions_path = cfg.questions_path
    if not questions_path:
        raise SystemExit("config.experiment.questions_path is required.")
    problems = rat2.load_filtered_problems(questions_path, args.limit)
    logging.info("Loaded %s problems from %s", len(problems), questions_path)

    # Flatten prompts across questions for batching
    all_prompts: List[str] = []
    owners: List[int] = []
    for qid, p in enumerate(problems):
        prompt = format_prompt(actor.tokenizer, p.question)
        for _ in range(args.samples):
            all_prompts.append(prompt)
            owners.append(qid)

    per_q_samples: List[List[Dict[str, Any]]] = [[] for _ in problems]
    batch_size = actor_batch_size or len(all_prompts)
    logging.info("Total prompts: %s, batch_size: %s", len(all_prompts), batch_size)

    for start in tqdm(range(0, len(all_prompts), batch_size), desc="Generating"):
        end = start + batch_size
        batch_prompts = all_prompts[start:end]
        batch_owners = owners[start:end]
        outputs = actor.llm.generate(batch_prompts, sampling_params=actor.sampling_params, use_tqdm=False)
        for out, owner in zip(outputs, batch_owners):
            comp = out.outputs[0]
            text = comp.text.strip()
            ans = extract_answer_loose(text)
            tok = len(comp.token_ids)
            per_q_samples[owner].append(
                {
                    "answer": ans,
                    "token_count": tok,
                    "correct": answers_match(problems[owner].answer, ans),
                }
            )

    results: List[Dict[str, Any]] = []
    for qid, p in enumerate(problems):
        samples_list = per_q_samples[qid]
        total = len(samples_list)
        correct = sum(1 for s in samples_list if s["correct"])
        acc = correct / total if total else 0.0
        results.append(
            {
                "question": p.question,
                "expected": p.answer,
                "samples": samples_list,
                "accuracy": acc,
                "correct": correct,
                "total": total,
            }
        )

    out_path = Path(args.output)
    ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logging.info("Wrote %s records to %s", len(results), out_path)


if __name__ == "__main__":
    main()

