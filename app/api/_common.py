from __future__ import annotations

import time
from typing import Any

from app.db.engine import session_scope
from app.db.models import RequestLog


def log_request(
    route: str,
    session_id: str | None,
    topics: list[str] | None,
    refused: bool,
    reason: str | None,
    started_at: float,
) -> None:
    latency_ms = int((time.monotonic() - started_at) * 1000)
    with session_scope() as sess:
        sess.add(
            RequestLog(
                route=route,
                session_id=session_id,
                topics_json=topics,
                refused=1 if refused else 0,
                reason=reason,
                latency_ms=latency_ms,
            )
        )
        sess.commit()


def make_response(
    reply: str, topics: list[str], refused: bool, reason: str | None
) -> dict[str, Any]:
    return {
        "reply": reply,
        "topics_matched": topics,
        "refused": refused,
        "reason": reason,
    }
