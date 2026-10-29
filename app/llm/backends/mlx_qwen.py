from __future__ import annotations

import logging
from typing import Any

from app.llm.client import Message


log = logging.getLogger(__name__)


class MLXQwenClient:
    """Local Qwen inference via Apple MLX. Mac + Apple Silicon only."""

    def __init__(self, model_name: str) -> None:
        from mlx_lm import load  # lazy import so Linux boxes don't blow up

        log.info("loading MLX model: %s (first run downloads from HuggingFace)", model_name)
        self.model_name = model_name
        self._model, self._tokenizer = load(model_name)

    
    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> str:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampler = make_sampler(temp=temperature)

        return mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_output_tokens,
            sampler=sampler,
            verbose=False,
        )

