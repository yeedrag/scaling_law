from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_answer(answer: str | None) -> str:
    if not answer:
        return ""
    text = answer.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def is_correct_prediction(prediction: str | None, reference: str | None) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_jsonl(records: list[Dict[str, Any]], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

