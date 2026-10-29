from __future__ import annotations

from fastapi import APIRouter

from app.db.engine import check_connection


router = APIRouter()


@router.get("/health")
def health() -> dict[str, object]:
    db_ok = check_connection()
    status = "ok" if db_ok else "degraded"
    return {"status": status, "db": db_ok}
