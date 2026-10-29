from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from app.config import PROJECT_ROOT, load_config
from app.db.engine import session_scope
from app.db.models import Chunk, Document, Embedding
from app.llm.embeddings import get_embedder
from app.parsers import SUPPORTED_EXTS, parse
from app.rag.chunker import chunk_text
from app.utils.hashing import sha256_file


log = logging.getLogger(__name__)

DOCS_DIR = PROJECT_ROOT / "llm_docs"


def _iter_source_files() -> list[Path]:
    if not DOCS_DIR.exists():
        return []
    out: list[Path] = []
    for p in sorted(DOCS_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith("."):
            out.append(p)
    return out


def _upsert_document(path: Path) -> tuple[int, bool]:
    """Return (document_id, changed). `changed` is True if we (re)indexed."""
    cfg = load_config()
    rel = str(path.relative_to(PROJECT_ROOT))
    file_hash = sha256_file(path)
    stat = path.stat()
    mtime = datetime.utcfromtimestamp(stat.st_mtime)

    with session_scope() as sess:
        doc = sess.scalar(select(Document).where(Document.source_path == rel))
        if doc and doc.content_hash == file_hash:
            return doc.id, False

        if doc is None:
            doc = Document(
                source_path=rel,
                content_hash=file_hash,
                byte_size=stat.st_size,
                modified_at=mtime,
                status="indexing",
            )
            sess.add(doc)
            sess.flush()
        else:
            # Changed file: wipe old chunks + embeddings via cascade.
            doc.content_hash = file_hash
            doc.byte_size = stat.st_size
            doc.modified_at = mtime
            doc.status = "indexing"
            doc.chunks.clear()
            sess.flush()

        doc_id = doc.id
        text = parse(path)
        chunks = chunk_text(
            text,
            chunk_size_tokens=cfg.rag.chunk_size_tokens,
            overlap_tokens=cfg.rag.chunk_overlap_tokens,
        )
        if not chunks:
            doc.status = "empty"
            sess.commit()
            return doc_id, True

        embedder = get_embedder()
        vectors = embedder.embed([c.text for c in chunks])
        for chunk, vec in zip(chunks, vectors):
            row = Chunk(
                document_id=doc_id,
                chunk_index=chunk.index,
                content=chunk.text,
                token_count=chunk.token_count,
            )
            sess.add(row)
            sess.flush()
            sess.add(
                Embedding(
                    chunk_id=row.id,
                    model=embedder.model_name,
                    dim=embedder.dim,
                    vector_json=vec,
                )
            )
        doc.status = "indexed"
        sess.commit()
        return doc_id, True


def _purge_removed_files(current_rel_paths: set[str]) -> int:
    with session_scope() as sess:
        existing = sess.scalars(select(Document)).all()
        removed = 0
        for d in existing:
            if d.source_path not in current_rel_paths:
                sess.delete(d)
                removed += 1
        sess.commit()
        return removed


def scan_and_index() -> dict[str, int]:
    """Scan llm_docs/, upsert new/changed files, purge removed files.
    Returns counts for logging."""
    files = _iter_source_files()
    log.info("indexer: scanning %d file(s) in %s", len(files), DOCS_DIR)
    rel_set: set[str] = set()
    added = updated = unchanged = 0
    for p in files:
        rel = str(p.relative_to(PROJECT_ROOT))
        rel_set.add(rel)
        try:
            _, changed = _upsert_document(p)
            if changed:
                updated += 1
            else:
                unchanged += 1
        except Exception as exc:
            log.exception("indexer: failed on %s: %s", p, exc)
    removed = _purge_removed_files(rel_set)
    stats = {
        "scanned": len(files),
        "upserted": updated,
        "unchanged": unchanged,
        "removed": removed,
    }
    log.info("indexer: %s", stats)
    return stats
