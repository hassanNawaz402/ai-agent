from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api._common import log_request, make_response
from app.config import load_config
from app.llm.factory import get_llm_client
from app.llm.prompts import build_rag_messages
from app.rag.retriever import search
from app.summarize.email_summarizer import summarize_email
from app.topics.classifier import classify_topics


router = APIRouter()
log = logging.getLogger(__name__)


class EmailRequest(BaseModel):
    text: str = Field(min_length=1)
    session_id: str | None = None  # ignored for email; kept for API symmetry


@router.post("/email")
def email(req: EmailRequest) -> dict[str, Any]:
    cfg = load_config()
    started = time.monotonic()

    summary = summarize_email(req.text)
    log.info("email summary (%d words): %s", len(summary.split()), summary[:200])

    topics = classify_topics(summary)
    if not topics:
        reason = (
            "This email covers topics outside what I can help with "
            f"({', '.join(cfg.extract_topics)})."
        )
        log_request("email", None, [], True, reason, started)
        return make_response("", [], True, reason)

    hits = search(summary)
    context_chunks = [h.content for h in hits]
    messages = build_rag_messages(
        f"Please reply to this customer email. "
        f"Here is a concise summary of their questions:\n\n{summary}",
        context_chunks,
    )
    client = get_llm_client()
    reply = client.generate(
        messages,
        temperature=cfg.generation.temperature,
        max_output_tokens=cfg.generation.max_output_tokens,
    )
    log_request("email", None, topics, False, None, started)
    return make_response(reply, topics, False, None)
