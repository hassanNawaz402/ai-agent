from __future__ import annotations

import logging
import os

from app.llm.client import Message


log = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI (ChatGPT) backend. Also used for connectivity checks at startup."""

    def __init__(
        self,
        model_name: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        from openai import OpenAI  # lazy import

        key = os.environ.get(api_key_env, "")
        if not key:
            raise RuntimeError(f"missing env var: {api_key_env}")
        self.model_name = model_name
        self._client = OpenAI(api_key=key, base_url=base_url, timeout=timeout_seconds)

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def ping(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception as exc:
            log.warning("openai connectivity check failed: %s", exc)
            return False
