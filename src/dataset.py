from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from datasets import Dataset, load_dataset


@dataclass
class Problem:
    question: str
    answer: str
    metadata: dict


# Backwards compatibility alias
ChemistryProblem = Problem


# Dataset-specific loaders
DATASET_LOADERS = {}


def register_loader(name: str):
    """Decorator to register a dataset-specific loader."""
    def decorator(func):
        DATASET_LOADERS[name] = func
        return func
    return decorator


@register_loader("rl-actors/Chemistry-GPQA")
def _load_chemistry_gpqa(row: dict) -> Optional[Problem]:
    """Load Chemistry-GPQA dataset (hard chemistry problems)."""
    question = str(row.get("question") or row.get("problem") or "").strip()
    answer = str(row.get("answer") or row.get("solution") or "").strip()
    if not question or not answer:
        return None
    metadata = {k: v for k, v in row.items() if k not in {"question", "problem", "answer", "solution"}}
    return Problem(question=question, answer=answer, metadata=metadata)


@register_loader("openai/gsm8k")
def _load_gsm8k(row: dict) -> Optional[Problem]:
    """Load GSM8k dataset (grade school math)."""
    question = str(row.get("question", "")).strip()
    # GSM8k answer format: "reasoning #### final_answer"
    full_answer = str(row.get("answer", "")).strip()
    # Extract just the final numeric answer after ####
    if "####" in full_answer:
        answer = full_answer.split("####")[-1].strip()
    else:
        answer = full_answer
    if not question or not answer:
        return None
    metadata = {"full_solution": full_answer}
    return Problem(question=question, answer=answer, metadata=metadata)


@register_loader("cais/mmlu")
def _load_mmlu(row: dict) -> Optional[Problem]:
    """Load MMLU dataset (multiple choice)."""
    question = str(row.get("question", "")).strip()
    choices = row.get("choices", [])
    answer_idx = row.get("answer", 0)
    
    if not question or not choices:
        return None
    
    # Format as multiple choice question
    choice_labels = ["A", "B", "C", "D"]
    formatted_choices = "\n".join(
        f"{choice_labels[i]}. {choice}" 
        for i, choice in enumerate(choices[:4])
    )
    full_question = f"{question}\n\n{formatted_choices}"
    
    # Answer is the letter
    answer = choice_labels[answer_idx] if isinstance(answer_idx, int) else str(answer_idx)
    
    metadata = {
        "subject": row.get("subject", "unknown"),
        "choices": choices,
        "answer_idx": answer_idx,
    }
    return Problem(question=full_question, answer=answer, metadata=metadata)


def _load_generic(row: dict) -> Optional[Problem]:
    """Generic loader for unknown datasets."""
    question = str(row.get("question") or row.get("problem") or row.get("prompt") or row.get("input") or "").strip()
    answer = str(row.get("answer") or row.get("solution") or row.get("output") or row.get("target") or "").strip()
    if not question or not answer:
        return None
    metadata = {k: v for k, v in row.items() if k not in {"question", "problem", "prompt", "input", "answer", "solution", "output", "target"}}
    return Problem(question=question, answer=answer, metadata=metadata)


def load_problems(
    dataset_name: str,
    split: str,
    subset: Optional[str] = None,
    limit: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
) -> List[Problem]:
    """Load problems from Hugging Face dataset and normalize fields.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "openai/gsm8k", "cais/mmlu")
        split: Dataset split (e.g., "train", "test")
        subset: Optional subset/config name (e.g., "high_school_chemistry" for MMLU)
        limit: Max number of problems to load
        shuffle_seed: Random seed for shuffling
    
    Returns:
        List of Problem objects
    """
    # Load dataset (with optional subset)
    if subset:
        hf_dataset: Dataset = load_dataset(dataset_name, subset, split=split)
    else:
        hf_dataset: Dataset = load_dataset(dataset_name, split=split)
    
    if shuffle_seed is not None:
        hf_dataset = hf_dataset.shuffle(seed=shuffle_seed)
    if limit is not None:
        hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

    # Get the appropriate loader
    loader = DATASET_LOADERS.get(dataset_name, _load_generic)
    
    problems: List[Problem] = []
    for row in hf_dataset:
        problem = loader(row)
        if problem:
            problems.append(problem)
    
    return problems


# Backwards compatibility
def load_chemistry_problems(
    dataset_name: str,
    split: str,
    limit: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
) -> List[Problem]:
    """Backwards-compatible function name."""
    return load_problems(dataset_name, split, subset=None, limit=limit, shuffle_seed=shuffle_seed)

