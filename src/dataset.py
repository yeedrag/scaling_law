from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from datasets import Dataset, load_dataset


@dataclass
class ChemistryProblem:
    question: str
    answer: str
    metadata: dict


def load_chemistry_problems(
    dataset_name: str,
    split: str,
    limit: int | None = None,
    shuffle_seed: int | None = None,
) -> List[ChemistryProblem]:
    """Load problems from Hugging Face dataset and normalize fields."""
    hf_dataset: Dataset = load_dataset(dataset_name, split=split)
    if shuffle_seed is not None:
        hf_dataset = hf_dataset.shuffle(seed=shuffle_seed)
    if limit is not None:
        hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

    problems: List[ChemistryProblem] = []
    for row in hf_dataset:
        question = str(row.get("question") or row.get("problem") or row.get("prompt") or "").strip()
        answer = str(row.get("answer") or row.get("solution") or "").strip()
        metadata = {k: v for k, v in row.items() if k not in {"question", "problem", "prompt", "answer", "solution"}}
        if not question or not answer:
            continue
        problems.append(ChemistryProblem(question=question, answer=answer, metadata=metadata))
    return problems

