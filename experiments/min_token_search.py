"""
Search for the minimum reasoning token budget that maintains accuracy.

For each question:
- Sample `--samples` generations at the original max_tokens (upper bound).
- Binary search max_tokens in [--min-tokens, upper] to find the smallest
  budget whose accuracy stays above a target (absolute or relative).
- Writes JSONL with per-question history of (max_tokens, accuracy).

This reuses RAT2 helpers for answer extraction/judging and the iteration-4
pressure prompt/prefix.

Example:
    PYTHONPATH=. python experiments/min_token_search.py \
        --config config.yaml \
        --samples 3 \
        --target-accuracy 0.78 \
        --relative-drop 0.02 \
        --output results/min_token_search.jsonl
"""
from __future__ import annotations
import logging
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

from tqdm import tqdm
from vllm import SamplingParams

from experiments import rat2
from experiments.run_experiment import (
    BASELINE_ACTOR_PROMPT,
)
from src.actor import ReasoningActor, extract_answer, extract_think
from src.config import load_config
from src.utils import set_global_seed


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class BudgetEval:
    max_tokens: int
    correct: int
    total: int
    accuracy: float
    token_counts: List[int]


@dataclass
class QuestionResult:
    question: str
    expected: str
    baseline_accuracy: float
    threshold: float
    min_tokens: Optional[int]
    history: List[BudgetEval]


@dataclass
class QState:
    question: str
    expected: str
    lo: int
    hi: int
    baseline: BudgetEval
    threshold: float
    best: Optional[BudgetEval]
    history: List[BudgetEval]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binary search minimal token budget that preserves accuracy.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    p.add_argument("--limit", type=int, default=None, help="Limit questions for smoke testing.")
    p.add_argument("--samples", type=int, default=3, help="Number of generations per token budget.")
    p.add_argument("--min-tokens", type=int, default=16, help="Lower bound for token budget search.")
    p.add_argument("--max-tokens", type=int, default=None, help="Upper bound; defaults to config.experiment.max_tokens or 2048.")
    p.add_argument("--target-accuracy", type=float, default=0.78, help="Absolute accuracy floor to accept a budget.")
    p.add_argument("--relative-drop", type=float, default=0.02, help="Allowed drop vs baseline accuracy.")
    p.add_argument("--output", default="results/min_token_search.jsonl", help="Path to output JSONL.")
    p.add_argument("--seed", type=int, default=42, help="Global seed for reproducibility.")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_sampling(base: SamplingParams, max_tokens: int) -> SamplingParams:
    """Clone base sampling params with a different max_tokens."""
    return SamplingParams(
        temperature=base.temperature,
        max_tokens=max_tokens,
        top_p=base.top_p,
        top_k=base.top_k,
        seed=base.seed,
        n=1,
        repetition_penalty=base.repetition_penalty,
        stop=base.stop,
    )


def format_prompt_with_budget(tokenizer, question: str, token_budget: int) -> str:
    """Apply chat template with prefix injected and a budget notice.

    Uses the benign (baseline) prompt; we only vary max_tokens.
    """
    budget_note = f"[Budget: at most {token_budget} tokens inside <think>; keep within this limit.]"
    user_text = BASELINE_ACTOR_PROMPT.format(question=f"{question}\n\n{budget_note}")
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ""},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def extract_answer_loose(text: str) -> Optional[str]:
    """Try strict tag parse, then a loose fallback for missing closing tags."""
    ans = extract_answer(text)
    if ans:
        return ans
    if "<answer>" in text:
        after = text.split("<answer>", 1)[1]
        # stop at first tag-ish boundary if present
        for stop_tok in ["</", "<think", "<answer"]:
            if stop_tok in after:
                after = after.split(stop_tok, 1)[0]
        candidate = after.strip()
        if candidate:
            return candidate.splitlines()[0].strip()
    return None


def summarize_outputs(outputs, prefix: str) -> List[dict]:
    records = []
    for out in outputs:
        comp = out.outputs[0]
        text = comp.text.strip()
        full = f"<think>\n{prefix}\n{text}"
        reasoning = extract_think(full) or full
        ans = extract_answer_loose(full)
        token_count = len(comp.token_ids)
        records.append({"answer": ans, "reasoning": reasoning, "full_text": full, "token_count": token_count})
    return records


def run_batched_generations(actor: ReasoningActor, batch: List[dict]) -> Dict[int, List[dict]]:
    """
    Run vLLM generations grouped by shared max_tokens.
    batch: list of {qid, question, max_tokens, samples}
    Returns: mapping qid -> list of record dicts (one per sample)
    """
    by_max: Dict[int, List[dict]] = {}
    for job in batch:
        by_max.setdefault(job["max_tokens"], []).append(job)

    results: Dict[int, List[dict]] = {}
    for max_tok, jobs in by_max.items():
        logger.info("Batch generate @%s tokens for %s qids, total prompts=%s", max_tok, len(jobs), sum(j["samples"] for j in jobs))
        sampling = build_sampling(actor.sampling_params, max_tok)
        prompts = []
        owners = []
        for job in jobs:
            for _ in range(job["samples"]):
                prompts.append(format_prompt_with_budget(actor.tokenizer, job["question"], max_tok))
                owners.append(job["qid"])
        outputs = actor.llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
        records = summarize_outputs(outputs, "")
        logger.info("Finished batch @%s tokens: %s prompts", max_tok, len(records))
        # bucket back to qids
        idx = 0
        for owner in owners:
            results.setdefault(owner, []).append(records[idx])
            idx += 1
    return results


def _to_number(text: str) -> Optional[float]:
    """Best-effort numeric parsing for simple integers/floats/fractions."""
    if text is None:
        return None
    cleaned = text.strip().lower()
    # remove simple currency symbols and trailing punctuation
    cleaned = cleaned.replace("$", "").replace("usd", "").replace("€", "").replace("£", "")
    cleaned = cleaned.rstrip("., ")
    # handle simple fractions like "3/4"
    if "/" in cleaned and all(part.strip().replace("-", "").isdigit() for part in cleaned.split("/", 1)):
        num, den = cleaned.split("/", 1)
        try:
            return float(num) / float(den)
        except ZeroDivisionError:
            return None
    # remove commas
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def answers_match(expected: str, candidate: str | None) -> bool:
    if candidate is None:
        return False
    e = expected.strip()
    c = candidate.strip()
    if e == c:
        return True
    # Numeric fallback
    en = _to_number(e)
    cn = _to_number(c)
    if en is not None and cn is not None:
        return abs(en - cn) < 1e-6
    return False


def evaluate_batch(actor: ReasoningActor, tasks: List[dict], expected_map: Dict[int, str]) -> Dict[int, BudgetEval]:
    """
    tasks: list of {qid, question, max_tokens, samples}
    expected_map: qid -> expected answer
    Returns: qid -> BudgetEval
    """
    records = run_batched_generations(actor, tasks)
    evals: Dict[int, BudgetEval] = {}
    for qid, recs in records.items():
        correct = sum(1 for r in recs if answers_match(expected_map[qid], r["answer"]))
        acc = correct / len(recs) if recs else 0.0
        evals[qid] = BudgetEval(
            max_tokens=tasks[0]["max_tokens"] if tasks else 0,  # overwritten below
            correct=correct,
            total=len(recs),
            accuracy=acc,
            token_counts=[r["token_count"] for r in recs],
        )
    # update max_tokens per qid from task info
    for t in tasks:
        if t["qid"] in evals:
            evals[t["qid"]].max_tokens = t["max_tokens"]
    return evals


def run_search(
    actor: ReasoningActor,
    problems: List[Any],
    min_tokens: int,
    max_tokens: int,
    samples: int,
    target_accuracy: float,
    relative_drop: float,
) -> List[QuestionResult]:
    # Assign qids
    states: Dict[int, QState] = {}
    expected_map: Dict[int, str] = {}

    # Baseline batch at upper bound
    logger.info("Starting baseline batch @%s tokens for %s problems", max_tokens, len(problems))
    baseline_tasks = []
    for qid, p in enumerate(problems):
        baseline_tasks.append({"qid": qid, "question": p.question, "max_tokens": max_tokens, "samples": samples})
        expected_map[qid] = p.answer
    baseline_evals = evaluate_batch(actor, baseline_tasks, expected_map)
    logger.info("Completed baseline batch")

    for qid, p in enumerate(problems):
        base_eval = baseline_evals[qid]
        threshold = max(target_accuracy, base_eval.accuracy - relative_drop)
        states[qid] = QState(
            question=p.question,
            expected=p.answer,
            lo=min_tokens,
            hi=max_tokens,
            baseline=base_eval,
            threshold=threshold,
            best=None,
            history=[base_eval],
        )

    # Binary search batched by shared mid
    pending = {qid for qid in states}
    while pending:
        # build tasks grouped by their current mid
        tasks_by_mid: Dict[int, List[int]] = {}
        for qid in list(pending):
            s = states[qid]
            if s.lo > s.hi:
                pending.discard(qid)
                continue
            mid = (s.lo + s.hi) // 2
            if mid == s.baseline.max_tokens:
                s.hi = mid - 1
                if s.lo > s.hi:
                    pending.discard(qid)
                continue
            tasks_by_mid.setdefault(mid, []).append(qid)

        if not tasks_by_mid:
            break

        # execute batches per mid (shared max_tokens)
        logger.info("Mid-level batches to run: %s", {m: len(qids) for m, qids in tasks_by_mid.items()})
        for mid, qids in tasks_by_mid.items():
            batch = [{"qid": qid, "question": states[qid].question, "max_tokens": mid, "samples": samples} for qid in qids]
            evals = evaluate_batch(actor, batch, expected_map)
            for qid in qids:
                s = states[qid]
                result = evals[qid]
                s.history.append(result)
                if result.accuracy >= s.threshold:
                    s.best = result
                    s.hi = mid - 1
                else:
                    s.lo = mid + 1
                if s.lo > s.hi:
                    pending.discard(qid)

    # Build outputs
    outputs: List[QuestionResult] = []
    for qid, s in states.items():
        outputs.append(
            QuestionResult(
                question=s.question,
                expected=s.expected,
                baseline_accuracy=s.baseline.accuracy,
                threshold=s.threshold,
                min_tokens=s.best.max_tokens if s.best else None,
                history=s.history,
            )
        )
    return outputs


def dump_jsonl(path: Path, records: Iterable[QuestionResult]) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "question": r.question,
                        "expected": r.expected,
                        "baseline_accuracy": r.baseline_accuracy,
                        "threshold": r.threshold,
                        "min_tokens": r.min_tokens,
                        "history": [
                            {
                                "max_tokens": h.max_tokens,
                                "correct": h.correct,
                                "total": h.total,
                                "accuracy": h.accuracy,
                                "token_counts": h.token_counts,
                            }
                            for h in r.history
                        ],
                    }
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_global_seed(args.seed)

    max_tokens = args.max_tokens or cfg.max_tokens or 2048
    temperature = cfg.temperature or 0.6
    model_id = cfg.qwen_model_id or "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    max_model_len = cfg.qwen_max_model_len or 4096
    tensor_parallel = None  # keep simple; adjust if needed

    actor = ReasoningActor(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel,
        seed=args.seed,
    )

    questions_path = cfg.questions_path
    if not questions_path:
        raise SystemExit("config.experiment.questions_path is required.")
    problems = rat2.load_filtered_problems(questions_path, args.limit)
    logger.info("Loaded %s problems from %s", len(problems), questions_path)

    results = run_search(
        actor=actor,
        problems=problems,
        min_tokens=args.min_tokens,
        max_tokens=max_tokens,
        samples=args.samples,
        target_accuracy=args.target_accuracy,
        relative_drop=args.relative_drop,
    )

    dump_jsonl(Path(args.output), results)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()

