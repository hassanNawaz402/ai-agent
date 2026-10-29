from __future__ import annotations

from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str
    content: str


class LLMClient(Protocol):
    """Minimal LLM interface. All backends — local Qwen, OpenAI, Anthropic,
    mock — implement these two methods and nothing else."""

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> str: ...


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def model_name(self) -> str: ...
    @property
    def dim(self) -> int: ...
