from __future__ import annotations

import logging
import re

from app.llm.client import Message


log = logging.getLogger(__name__)


class MockQwenClient:
    """Deterministic LLM for smoke tests. Returns short canned replies.

    - Topic classification prompts: returns a JSON list of topics it sees mentioned.
    - Summarization prompts: returns the first 100 words of the user content.
    - Everything else: returns a short grounded-looking reply.
    """

    def __init__(self, model_name: str = "mock-qwen") -> None:
        self.model_name = model_name

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> str:
        user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        system = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )

        if "Return ONLY a JSON array" in system or "topic classifier" in system.lower():
            return self._classify(system, user)

        if "summarize" in system.lower() or "summary" in system.lower():
            return self._summarize(user)

        return (
            "Thank you for your query. Based on the provided context, "
            "here is a brief answer. (Mock backend — replace with a real "
            "model in llm_configs.yml for production responses.)"
        )

    def _classify(self, system: str, user: str) -> str:
        topics = re.findall(r"- ([a-zA-Z0-9 _\-]+)", system)
        matched = [t for t in topics if t.lower() in user.lower()]
        import json
        return json.dumps(matched)

    def _summarize(self, user: str) -> str:
        words = user.split()
        return " ".join(words[:100])
