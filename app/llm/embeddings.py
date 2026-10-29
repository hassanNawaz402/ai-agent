from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import load_config


log = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        log.info("loading embedding model: %s", model_name)
        self._model_name = model_name
        self._model: Any = SentenceTransformer(model_name)
        self._dim: int = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim


class MockEmbedder:
    """Deterministic 384-dim embedder for pipeline smoke tests without downloading models."""

    _DIM = 384

    def __init__(self, model_name: str = "mock-embed-384") -> None:
        self._model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        import math

        out: list[list[float]] = []
        for t in texts:
            # Deterministic pseudo-embedding seeded by text hash; normalized.
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "big")
            rng = _lcg(seed)
            raw = [next(rng) for _ in range(self._DIM)]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            out.append([x / norm for x in raw])
        return out

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._DIM


def _lcg(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        # map to [-1, 1)
        yield (state / 0x3FFFFFFF) - 1.0


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformerEmbedder | MockEmbedder:
    cfg = load_config()
    model_name = cfg.active_profile.embedding_model
    if model_name == "mock":
        return MockEmbedder()
    return SentenceTransformerEmbedder(model_name)
