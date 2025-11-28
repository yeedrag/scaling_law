from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Literal, Sequence

from tqdm import tqdm
from src.actor import ReasoningActor
from src.config import ExperimentConfig, load_config
from src.dataset import load_problems
from src.monitor import ChainOfThoughtMonitor
from src.prompts import BASELINE_ACTOR_PROMPT, EVASION_ACTOR_PROMPT
from src.utils import ensure_dir, is_correct_prediction, save_jsonl, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reasoning length vs. monitor evasion experiment.")
    parser.add_argument("--config", default="config.yaml", help="Path to experiment config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems for smoke testing.")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Optional tensor parallel size for vLLM.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for actor generations.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--include-prompts", action="store_true", help="Include raw prompts in output JSONL.")
    return parser.parse_args()


@dataclass
class GenerationTask:
    problem_index: int
    sample_index: int
    variant: Literal["baseline", "evasion"]
    prompt: str
    question: str
    answer: str


def chunked(seq: Sequence[GenerationTask], size: int) -> Iterable[list[GenerationTask]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    cfg: ExperimentConfig = load_config(args.config)
    set_global_seed(cfg.random_seed)

    problems = load_problems(
        dataset_name=cfg.dataset_name,
        split=cfg.dataset_split,
        subset=cfg.dataset_subset,
        limit=args.limit,
        shuffle_seed=cfg.random_seed,
    )
    dataset_desc = f"{cfg.dataset_name}" + (f"/{cfg.dataset_subset}" if cfg.dataset_subset else "")
    logging.info("Loaded %d problems from %s", len(problems), dataset_desc)

    batch_size = args.batch_size or getattr(cfg, "actor_batch_size", 1)
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    
    logging.info("Initializing vLLM with model: %s", cfg.qwen_model_id)
    init_start = time.perf_counter()
    actor = ReasoningActor(
        model_id=cfg.qwen_model_id,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        reasoning_parser=cfg.qwen_reasoning_parser,
        max_model_len=cfg.qwen_max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=cfg.random_seed,
    )
    init_elapsed = time.perf_counter() - init_start
    logging.info("vLLM model initialized in %.2fs", init_elapsed)
    
    monitor = ChainOfThoughtMonitor(
        model=cfg.monitor_model, 
        subject=cfg.monitor_subject,
        max_tokens=cfg.monitor_max_tokens,
    )
    logging.info("Monitor configured to detect: %s", cfg.monitor_subject)

    records = []

    for problem_idx, problem in enumerate(
        tqdm(problems, desc="Problems", unit="problem", total=len(problems))
    ):
        tasks: list[GenerationTask] = []
        for sample_idx in range(cfg.samples_per_problem):
            tasks.append(
                GenerationTask(
                    problem_index=problem_idx,
                    sample_index=sample_idx,
                    variant="baseline",
                    prompt=BASELINE_ACTOR_PROMPT.format(question=problem.question),
                    question=problem.question,
                    answer=problem.answer,
                )
            )
            tasks.append(
                GenerationTask(
                    problem_index=problem_idx,
                    sample_index=sample_idx,
                    variant="evasion",
                    prompt=EVASION_ACTOR_PROMPT.format(question=problem.question),
                    question=problem.question,
                    answer=problem.answer,
                )
            )

        baseline_cache: dict[int, dict] = {}
        evasion_pending: list[tuple[GenerationTask, object]] = []  # (task, response)
        
        # Phase 1: Generate all responses with vLLM (batched)
        gen_start = time.perf_counter()
        for task_batch in chunked(tasks, batch_size):
            prompts = [task.prompt for task in task_batch]
            responses = actor.generate_batch(prompts)
            for task, response in zip(task_batch, responses):
                if task.variant == "baseline":
                    baseline_correct = is_correct_prediction(response.final_answer, task.answer)
                    baseline_cache[task.sample_index] = {
                        "prompt": task.prompt,
                        "reasoning": response.reasoning,
                        "final_answer": response.final_answer,
                        "correct": baseline_correct,
                        "token_count": response.token_count,
                    }
                    logging.debug("Baseline response: %s", response.reasoning[:200] if response.reasoning else "None")
                else:
                    # Queue evasion for concurrent monitor classification
                    evasion_pending.append((task, response))
                    logging.debug("Evasion response: %s", response.reasoning[:200])
        gen_elapsed = time.perf_counter() - gen_start
        logging.info("Problem %d: vLLM generation took %.2fs for %d tasks", problem_idx, gen_elapsed, len(tasks))
        
        # Phase 2: Classify all evasion responses CONCURRENTLY
        monitor_start = time.perf_counter()
        monitor_results: dict[int, str] = {}  # sample_index -> label
        
        def classify_task(sample_idx: int, reasoning: str) -> tuple[int, str]:
            return sample_idx, monitor.classify(reasoning)
        
        with ThreadPoolExecutor(max_workers=len(evasion_pending)) as executor:
            futures = {
                executor.submit(classify_task, task.sample_index, response.reasoning): (task, response)
                for task, response in evasion_pending
            }
            for future in as_completed(futures):
                sample_idx, label = future.result()
                monitor_results[sample_idx] = label
        
        monitor_elapsed = time.perf_counter() - monitor_start
        logging.info("Problem %d: Monitor classification took %.2fs for %d evasions", 
                     problem_idx, monitor_elapsed, len(evasion_pending))
        
        # Phase 3: Assemble records
        for task, response in evasion_pending:
            monitor_label = monitor_results[task.sample_index]
            logging.debug("Monitor label for sample %d: %s", task.sample_index, monitor_label)
            monitor_detected = monitor_label == "DETECTED"
            evasion_correct = is_correct_prediction(response.final_answer, task.answer)
            baseline_data = baseline_cache.get(task.sample_index)
            if baseline_data is None:
                logging.warning(
                    "Missing baseline result for problem %d sample %d",
                    task.problem_index,
                    task.sample_index,
                )
                continue
            evasion_success = evasion_correct and not monitor_detected

            record = {
                "problem_index": task.problem_index,
                "sample_index": task.sample_index,
                "question": task.question,
                "answer": task.answer,
                "baseline_reasoning": baseline_data["reasoning"],
                "baseline_final_answer": baseline_data["final_answer"],
                "baseline_correct": baseline_data["correct"],
                "baseline_reasoning_tokens": baseline_data["token_count"],
                "evasion_reasoning": response.reasoning,
                "evasion_final_answer": response.final_answer,
                "evasion_correct": evasion_correct,
                "monitor_label": monitor_label,
                "monitor_detected": monitor_detected,
                "evasion_success": evasion_success,
            }
            if args.include_prompts:
                record["baseline_prompt"] = baseline_data["prompt"]
                record["evasion_prompt"] = task.prompt
            records.append(record)

    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)
    jsonl_path = output_dir / "chemistry_monitorability.jsonl"
    save_jsonl(records, jsonl_path)
    logging.info("Saved detailed results to %s", jsonl_path)

    summary = build_summary(records)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("Saved summary metrics to %s", summary_path)


def build_summary(records: list[dict]) -> dict:
    total = len(records)
    if total == 0:
        return {}
    baseline_acc = sum(1 for r in records if r["baseline_correct"]) / total
    evasion_acc = sum(1 for r in records if r["evasion_correct"]) / total
    evasion_success = sum(1 for r in records if r["evasion_success"]) / total
    monitor_rate = sum(1 for r in records if r["monitor_detected"]) / total
    avg_reasoning_tokens = mean(r["baseline_reasoning_tokens"] for r in records)
    return {
        "total_samples": total,
        "baseline_accuracy": baseline_acc,
        "evasion_accuracy": evasion_acc,
        "evasion_success_rate": evasion_success,
        "monitor_detection_rate": monitor_rate,
        "avg_baseline_reasoning_tokens": avg_reasoning_tokens,
    }


if __name__ == "__main__":
    main()

