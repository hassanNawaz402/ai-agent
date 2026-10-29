
from __future__ import annotations

import logging
import os

import ollama

from app.llm.client import Message


log = logging.getLogger(__name__)


class OllamaQwenClient:
    """Local Qwen inference through the Ollama API."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 300.0,
    ) -> None:
        self.model_name = model_name
        self.client = ollama.Client(
            host=base_url,
            timeout=timeout_seconds,
        )

        log.info("initialized Ollama client for model: %s", model_name)

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": message["role"], "content": message["content"]}
                for message in messages
            ],
            options={
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
            keep_alive="24h",
        )

        
        log.info(
            "OLLAMA_TIMING total=%.2fs load=%.2fs "
            "prompt_eval=%.2fs prompt_tokens=%s "
            "generation=%.2fs output_tokens=%s",
            getattr(response, "total_duration", 0) / 1e9,
            getattr(response, "load_duration", 0) / 1e9,
            getattr(response, "prompt_eval_duration", 0) / 1e9,
            getattr(response, "prompt_eval_count", 0),
            getattr(response, "eval_duration", 0) / 1e9,
            getattr(response, "eval_count", 0),
        )

        return response.message.content

