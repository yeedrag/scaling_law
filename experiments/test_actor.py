from __future__ import annotations

import argparse
import logging
from statistics import mean
from time import perf_counter

from src.actor import ReasoningActor
from src.config import ExperimentConfig, load_config
from src.dataset import load_chemistry_problems
from src.prompts import BASELINE_ACTOR_PROMPT
from src.utils import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick latency tester for ReasoningActor.")
    parser.add_argument("--config", default="config.yaml", help="Path to experiment config.")
    parser.add_argument(
        "--question",
        default=None,
        help="Custom question to probe the actor. Defaults to first chemistry problem.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of forward passes to time.")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Tensor parallel size for vLLM.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    cfg: ExperimentConfig = load_config(args.config)
    set_global_seed(cfg.random_seed)

    if args.question:
        question = args.question
        logging.info("Using custom question: %s", question)
    else:
        problems = load_chemistry_problems(
            dataset_name=cfg.dataset_name,
            split=cfg.dataset_split,
            limit=1,
            shuffle_seed=cfg.random_seed,
        )
        if not problems:
            raise RuntimeError("No problems loaded; provide --question to test manually.")
        question = problems[0].question
        logging.info("Using first problem from dataset %s.", cfg.dataset_name)

    actor = ReasoningActor(
        model_id=cfg.qwen_model_id,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        reasoning_parser=cfg.qwen_reasoning_parser,
        max_model_len=cfg.qwen_max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=cfg.random_seed,
    )

    prompt = BASELINE_ACTOR_PROMPT.format(question=question)
    latencies: list[float] = []
    token_counts: list[int] = []

    for run_idx in range(1, args.runs + 1):
        start = perf_counter()
        response = actor.generate(prompt)
        elapsed = perf_counter() - start
        latencies.append(elapsed)
        token_counts.append(response.token_count)

        logging.info(
            "Run %d | %.2fs | %d tokens | answer=%s",
            run_idx,
            elapsed,
            response.token_count,
            response.final_answer,
        )

    if latencies:
        logging.info(
            "Latency stats over %d runs: avg=%.2fs | min=%.2fs | max=%.2fs",
            len(latencies),
            mean(latencies),
            min(latencies),
            max(latencies),
        )
        logging.info(
            "Avg token count: %.1f",
            mean(token_counts),
        )


if __name__ == "__main__":
    main()


