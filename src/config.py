import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclasses.dataclass
class ExperimentConfig:
    dataset_name: str
    dataset_split: str
    samples_per_problem: int
    temperature: float
    max_tokens: int
    baseline_max_retries: int
    evasion_prompt: str
    monitor_model: str
    monitor_max_tokens: int
    qwen_model_id: str
    qwen_reasoning_parser: str
    qwen_max_model_len: int
    output_dir: str
    random_seed: int
    actor_batch_size: int = 1


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

