from __future__ import annotations

import re
from dataclasses import dataclass


_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    token_count: int


def _approx_tokens(text: str) -> int:
    # ~0.75 words per token for English; use word count as a stable proxy
    # so we don't need a model tokenizer here.
    return max(1, len(_WORD_RE.findall(text)))


def chunk_text(
    text: str, chunk_size_tokens: int = 512, overlap_tokens: int = 64
) -> list[Chunk]:
    if not text.strip():
        return []
    words = _WORD_RE.findall(text)
    if not words:
        return []
    # Treat each word as ~1 token for chunking.
    step = max(1, chunk_size_tokens - overlap_tokens)
    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < len(words):
        window = words[i : i + chunk_size_tokens]
        if not window:
            break
        chunk_str = " ".join(window)
        chunks.append(
            Chunk(index=idx, text=chunk_str, token_count=_approx_tokens(chunk_str))
        )
        idx += 1
        i += step
    return chunks
