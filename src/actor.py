from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from vllm import LLM, SamplingParams


FINAL_ANSWER_REGEX = re.compile(r"final answer\s*[:\-]\s*(.+)", re.IGNORECASE)


@dataclass
class ActorResponse:
    text: str
    reasoning: str
    final_answer: Optional[str]
    token_count: int


class ReasoningActor:
    """Wrapper around vLLM for Qwen3 32B thinking model."""

    def __init__(
        self,
        model_id: str,
        temperature: float,
        max_tokens: int,
        reasoning_parser: str | None = None,
        max_model_len: int | None = None,
        tensor_parallel_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        llm_kwargs = {
            "model": model_id,
            "trust_remote_code": True,
        }
        if reasoning_parser:
            llm_kwargs["reasoning_parser"] = reasoning_parser
        if max_model_len:
            llm_kwargs["max_model_len"] = max_model_len
        if tensor_parallel_size:
            llm_kwargs["tensor_parallel_size"] = tensor_parallel_size

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            top_k=-1,
            seed=seed,
            n=1,
            repetition_penalty=1.0,
        )

    def generate(self, prompt: str) -> ActorResponse:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: List[str]) -> List[ActorResponse]:
        logging.debug("Generating batch of %d prompts", len(prompts))
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)
        responses: List[ActorResponse] = []
        for request_output in outputs:
            completion = request_output.outputs[0]
            responses.append(self._build_response(completion))
        return responses

    @staticmethod
    def _build_response(completion) -> ActorResponse:
        text = completion.text.strip()
        final_answer = extract_final_answer(text)
        reasoning = text
        token_count = len(completion.token_ids)
        return ActorResponse(
            text=text,
            reasoning=reasoning,
            final_answer=final_answer,
            token_count=token_count,
        )


def extract_final_answer(text: str) -> Optional[str]:
    match = FINAL_ANSWER_REGEX.search(text)
    if match:
        return match.group(1).strip()
    return None

