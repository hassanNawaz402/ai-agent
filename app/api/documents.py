from __future__ import annotations

from fastapi import APIRouter

from app.rag.indexer import scan_and_index


router = APIRouter(prefix="/documents")


@router.post("/reindex")
def reindex() -> dict[str, int]:
    return scan_and_index()
