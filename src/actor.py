from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


@dataclass
class ActorResponse:
    text: str
    reasoning: str
    final_answer: Optional[str]
    token_count: int


class ReasoningActor:
    """Wrapper around vLLM for reasoning models with chat template support."""

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            top_k=-1,
            seed=seed,
            n=1,
            repetition_penalty=1.0,
            stop=["</answer>"],  # Stop after answer tag
        )

    def _apply_chat_template(self, user_message: str) -> str:
        """Apply the model's chat template to format the prompt correctly."""
        messages = [{"role": "user", "content": user_message}]
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    def generate(self, prompt: str) -> ActorResponse:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: List[str]) -> List[ActorResponse]:
        logging.debug("Generating batch of %d prompts", len(prompts))
        # Apply chat template to each prompt
        formatted_prompts = [self._apply_chat_template(p) for p in prompts]
        outputs = self.llm.generate(formatted_prompts, sampling_params=self.sampling_params, use_tqdm=False)
        responses: List[ActorResponse] = []
        for request_output in outputs:
            completion = request_output.outputs[0]
            responses.append(self._build_response(completion))
        return responses

    @staticmethod
    def _build_response(completion) -> ActorResponse:
        text = completion.text.strip()
        
        # Extract reasoning from <think>...</think> tags
        reasoning = extract_think(text)
        if reasoning:
            logging.debug("Extracted reasoning from <think> tags: %d chars", len(reasoning))
        else:
            reasoning = text
            logging.debug("No <think> tags found, using full text: %d chars", len(reasoning))
        
        # Extract final answer from <answer>...</answer> tags
        final_answer = extract_answer(text)
        if final_answer:
            logging.debug("Extracted answer from <answer> tags: %s", final_answer[:50])
        
        token_count = len(completion.token_ids)
        return ActorResponse(
            text=text,
            reasoning=reasoning,
            final_answer=final_answer,
            token_count=token_count,
        )


def extract_think(text: str) -> Optional[str]:
    """Extract thinking content (everything before </think> tag)."""
    if '</think>' in text.lower():
        idx = text.lower().find('</think>')
        return text[:idx].strip() or None
    return None


def extract_answer(text: str) -> Optional[str]:
    """Extract content between <answer> and </answer> tags (last instance)."""
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None

