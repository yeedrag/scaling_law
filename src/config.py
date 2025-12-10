import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclasses.dataclass
class ExperimentConfig:
    # Core fields (provide safe defaults for tools that don't need them)
    dataset_name: str = ""
    dataset_split: str = ""
    temperature: float = 1.0
    max_tokens: int = 4096
    monitor_model: str = ""
    monitor_max_tokens: int = 0
    qwen_model_id: str = ""
    qwen_max_model_len: int = 4096
    output_dir: str = "results"
    random_seed: int = 42
    # Optional fields with defaults
    dataset_subset: Optional[str] = None  # For MMLU subjects like "high_school_chemistry"
    monitor_subject: str = "chemistry"  # Subject the monitor looks for (e.g., "chemistry", "biology", "math")
    samples_per_problem: int = 2
    baseline_max_retries: int = 2
    qwen_reasoning_parser: Optional[str] = None
    actor_batch_size: int = 1
    questions_path: Optional[str] = None  # Path to questions JSON (list of {question, answer})


def _expand_env(value: Any) -> Any:
    """Recursively expand environment variables inside config strings."""
    if isinstance(value, str):
        return Path(value).expanduser().__str__() if value.startswith(("~", "/")) else value
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    experiment = raw.get("experiment", {})
    experiment = _expand_env(experiment)
    return ExperimentConfig(**experiment)

