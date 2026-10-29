from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.config import load_config
from app.db.engine import session_scope
from app.db.models import Chunk, Embedding
from app.llm.embeddings import get_embedder


log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    source_path: str
    content: str
    score: float


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    total_started = time.monotonic()
    stage_started = time.monotonic()
    cfg = load_config()
    log.info(
        "RAG_TIMING config_load=%.2fs",
        time.monotonic() - stage_started,
    )
    k = top_k if top_k is not None else cfg.rag.top_k
    threshold = cfg.rag.min_similarity

    
    stage_started = time.monotonic()
    embedder = get_embedder()
    qvec = embedder.embed([query])[0]
    model_name = embedder.model_name
    log.info(
        "RAG_TIMING query_embedding=%.2fs",
        time.monotonic() - stage_started,
    )

    stage_started = time.monotonic()
    scored: list[RetrievedChunk] = []
    with session_scope() as sess:
        rows = sess.execute(
            select(Embedding).where(Embedding.model == model_name).options(
                joinedload(Embedding.chunk).joinedload(Chunk.document)
            )
        ).scalars().all()

        log.info(
            "RAG_TIMING database_load=%.2fs embeddings=%d",
            time.monotonic() - stage_started,
            len(rows),
        )

        stage_started = time.monotonic()
        for emb in rows:
            score = _cosine(qvec, emb.vector_json)
            if score < threshold:
                continue
            ch = emb.chunk
            scored.append(
                RetrievedChunk(
                    chunk_id=ch.id,
                    document_id=ch.document_id,
                    source_path=ch.document.source_path,
                    content=ch.content,
                    score=score,
                )
            )

        log.info(
            "RAG_TIMING similarity_scoring=%.2fs",
            time.monotonic() - stage_started,
        )

    scored.sort(key=lambda r: r.score, reverse=True)
    result = scored[:k]
    log.info(
        "RAG_TIMING search_total=%.2fs",
        time.monotonic() - total_started,
    )
    return result
