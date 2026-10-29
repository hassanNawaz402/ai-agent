from __future__ import annotations

import logging


log = logging.getLogger(__name__)


_SUMMARIZE_CHUNK = (
    "You are an email summarizer. "
    "Read the email chunk and produce a concise summary that lists the "
    "customer's questions and requests. "
    "Keep it under 80 words. No preamble, no quotes, no commentary."
)

_REDUCE = (
    "You are an email summarizer. "
    "Below are partial summaries of different sections of the same email. "
    "Merge them into a single summary that enumerates all the customer's "
    "questions and requests, in order. "
    "Keep it under {max_words} words. No preamble, no commentary."
)


def _split_by_words(text: str, chunk_size_words: int) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size_words:
        return [text]
    chunks: list[str] = []
    for i in range(0, len(words), chunk_size_words):
        chunks.append(" ".join(words[i : i + chunk_size_words]))
    return chunks


def _summarize_once(system: str, user_text: str, max_tokens: int) -> str:
    from app.llm.factory import get_llm_client  # lazy
    client = get_llm_client()
    return client.generate(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=0.1,
        max_output_tokens=max_tokens,
    ).strip()


def summarize_email(text: str) -> str:
    from app.config import load_config  # lazy
    cfg = load_config()
    max_words = cfg.email.max_summary_words
    chunk_size = cfg.email.chunk_size_words

    chunks = _split_by_words(text, chunk_size)
    log.info("email summarizer: %d chunk(s), max %d words final", len(chunks), max_words)

    if len(chunks) == 1:
        return _summarize_once(
            f"{_SUMMARIZE_CHUNK}\nKeep it under {max_words} words.",
            chunks[0],
            max_tokens=max_words * 2,
        )

    partial = [_summarize_once(_SUMMARIZE_CHUNK, c, max_tokens=200) for c in chunks]
    joined = "\n\n".join(f"[Part {i+1}]\n{p}" for i, p in enumerate(partial))
    return _summarize_once(
        _REDUCE.format(max_words=max_words),
        joined,
        max_tokens=max_words * 2,
    )
