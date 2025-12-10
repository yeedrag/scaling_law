"""
Run the Iter-4 pressure prompt multiple times per question and aggregate metrics,
writing incrementally so runs can be resumed.

For each question:
  - Generate `--samples` pressure responses (no benign).
  - Evaluate correctness, coverage, legibility for each sample.
  - Aggregate averages per question (accuracy, avg coverage/legibility, mean token_count).

Outputs: JSONL where each line is a question with its samples and aggregates.

Usage:
    PYTHONPATH=. python experiments/run_pressure_samples.py \
        --config config.yaml \
        --samples 20 \
        --limit 200 \
        --output results/pressure_samples.jsonl \
        --autorater-model gpt-5.1 \
        --judge-model gpt-5-nano \
        --openai-concurrency 32 \
        --chunk-size 16 \
        --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

from openai import AsyncOpenAI
from tqdm import tqdm

from src.actor import ReasoningActor, extract_answer, extract_think
from src.config import load_config
from src.utils import set_global_seed
from experiments.run_experiment import (
    PRESSURE_PREFIX,
    PRESSURE_PROMPT,
    safe_int,
    openai_call,
    extract_answer_with_judge,
    judge_correctness,
    score_monitorability,
)
from experiments.min_token_search import answers_match
from experiments import rat2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pressure-only multi-sample runner.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    p.add_argument("--limit", type=int, default=None, help="Limit questions.")
    p.add_argument("--samples", type=int, default=20, help="Samples per question.")
    p.add_argument("--autorater-model", default="gpt-5.1", help="Model for coverage/legibility scoring.")
    p.add_argument("--judge-model", default="gpt-5-nano", help="Model for correctness judging.")
    p.add_argument("--openai-concurrency", type=int, default=32, help="Max concurrent OpenAI calls.")
    p.add_argument("--output", default="results/pressure_samples.jsonl", help="Output JSONL path.")
    p.add_argument("--seed", type=int, default=42, help="Seed; -1 to disable seeding.")
    p.add_argument("--chunk-size", type=int, default=16, help="Process questions in chunks to allow incremental writes.")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_with_prefix(actor: ReasoningActor, question: str, prefix: str) -> str:
    messages = [
        {"role": "user", "content": PRESSURE_PROMPT.format(question=question)},
        {"role": "assistant", "content": f"<think>\n{prefix}\n"},
    ]
    return actor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


@dataclass
class SampleRecord:
    question: str
    expected: str
    reasoning: str
    answer: str | None
    token_count: int
    correct: bool | None
    legibility: int | None
    coverage: int | None
    justification: str


async def evaluate_sample(client: AsyncOpenAI, sem: asyncio.Semaphore, autorater_model: str, judge_model: str, sample: dict) -> dict:
    # sample contains question, expected, reasoning, answer, token_count
    ans = sample.get("answer")
    if ans is None:
        ans = await extract_answer_with_judge(client, judge_model, sample.get("reasoning", "") or sample.get("full_text", ""), sem)
    ok = await judge_correctness(client, judge_model, sample.get("expected"), ans, sem)
    sc = await score_monitorability(client, autorater_model, sample["question"], sample["reasoning"] or "", ans or "", sem)
    return {**sample, "answer": ans, "correct": ok, **sc}


async def main_async(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.seed >= 0:
        set_global_seed(args.seed)

    # vLLM actor
    actor = ReasoningActor(
        model_id=cfg.qwen_model_id,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        max_model_len=cfg.qwen_max_model_len,
        tensor_parallel_size=None,
        max_num_seqs=getattr(cfg, "actor_batch_size", None) or None,
        seed=None if args.seed < 0 else args.seed,
    )

    # Data
    questions_path = getattr(cfg, "questions_path", None)
    if not questions_path:
        raise SystemExit("config.experiment.questions_path is required.")
    problems = rat2.load_filtered_problems(questions_path, args.limit)
    logging.info("Loaded %s problems from %s", len(problems), questions_path)

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
        logging.info("Nothing to do; all questions already processed.")
        return

    async_client = AsyncOpenAI(timeout=60)
    sem = asyncio.Semaphore(args.openai_concurrency)

    # Process in chunks
    for start in range(0, len(pending), args.chunk_size):
        chunk = pending[start : start + args.chunk_size]
        prompts: List[str] = []
        owners: List[int] = []
        idx_to_question: Dict[int, str] = {}
        for idx, p in chunk:
            formatted = format_with_prefix(actor, p.question, PRESSURE_PREFIX)
            for _ in range(args.samples):
                prompts.append(formatted)
                owners.append(idx)
            idx_to_question[idx] = p.question

        logging.info("Chunk %s/%s: generating %s prompts for %s questions", start // args.chunk_size + 1, (len(pending)+args.chunk_size-1)//args.chunk_size, len(prompts), len(chunk))
        outputs = actor.llm.generate(prompts, sampling_params=actor.sampling_params, use_tqdm=False)

        per_q_raw: Dict[int, List[dict]] = defaultdict(list)
        for out, idx in zip(outputs, owners):
            comp = out.outputs[0]
            text = comp.text.strip()
            full = f"<think>\n{text}"
            reasoning = extract_think(full) or full
            ans = extract_answer(full)
            p = problems[idx]
            per_q_raw[idx].append(
                {
                    "question": p.question,
                    "expected": p.answer,
                    "reasoning": reasoning,
                    "full_text": full,
                    "answer": ans,
                    "token_count": len(comp.token_ids),
                }
            )

        # Evaluate this chunk
        tasks = []
        for idx, samples in per_q_raw.items():
            for s in samples:
                tasks.append(evaluate_sample(async_client, sem, args.autorater_model, args.judge_model, s))

        results_eval: List[dict] = []
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Scoring chunk {start//args.chunk_size+1}", unit="sample"):
            results_eval.append(await fut)

        # Group and append to file
        per_q_final: Dict[str, List[dict]] = defaultdict(list)
        for r in results_eval:
            per_q_final[r["question"]].append(r)

        with out_path.open("a", encoding="utf-8") as f:
            for idx, p in chunk:
                samples = per_q_final.get(p.question, [])
                acc = sum(1 for s in samples if s.get("correct")) / len(samples) if samples else 0.0
                avg_cov = mean([safe_int(s.get("coverage"), 0) for s in samples]) if samples else 0.0
                avg_leg = mean([safe_int(s.get("legibility"), 0) for s in samples]) if samples else 0.0
                avg_tok = mean([s.get("token_count", 0) for s in samples]) if samples else 0.0
                record = {
                    "question": p.question,
                    "expected": p.answer,
                    "samples": samples,
                    "avg_coverage": avg_cov,
                    "avg_legibility": avg_leg,
                    "accuracy": acc,
                    "avg_token_count": avg_tok,
                    "num_samples": len(samples),
                }
                f.write(json.dumps(record) + "\n")
        logging.info("Wrote chunk %s questions to %s", len(chunk), out_path)

    logging.info("Done. Output: %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

