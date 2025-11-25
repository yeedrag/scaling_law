from __future__ import annotations

import logging
import os
from typing import Literal

from openai import OpenAI

from .prompts import MONITOR_PROMPT


class ChainOfThoughtMonitor:
    """Wrapper around OpenAI monitor model to detect chemistry reasoning."""

    def __init__(self, model: str, max_tokens: int = 128) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable is required for the monitor.")
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens

    def classify(self, reasoning: str) -> Literal["DETECTED", "NOT_DETECTED"]:
        prompt = MONITOR_PROMPT.format(reasoning=reasoning)
        logging.debug("Sending monitor prompt of length %d", len(prompt))
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=self.max_tokens,
            temperature=0,
        )
        text_blocks = []
        for block in response.output:
            for item in block.content:
                if hasattr(item, "text"):
                    text_blocks.append(item.text)
        text = " ".join(text_blocks).strip().upper()
        if "DETECTED" in text:
            return "DETECTED"
        return "NOT_DETECTED"

