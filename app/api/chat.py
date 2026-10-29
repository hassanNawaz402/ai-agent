from __future__ import annotations

import logging
import re
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert

from app.api._common import log_request, make_response
from app.config import load_config
from app.db.engine import session_scope
from app.db.models import ChatSession
from app.llm.factory import get_llm_client
from app.llm.prompts import build_rag_messages
from app.rag.retriever import search


router = APIRouter()
log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    text: str = Field(min_length=1)
    session_id: str | None = None


QUOTE_START_PHRASES = (
    "i am looking for a quote",
    "i'm looking for a quote",
    "i need a quote",
    "i need a removals quote",
    "how much to move",
    "i need a man and van",
)

_NUMBER_WORDS = {
    "zero": 0, "no": 0, "one": 1, "a": 1, "an": 1,
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_ITEM_NAMES = (
    "sofas", "sofa", "settees", "settee", "armchairs", "armchair",
    "tables", "table", "chairs", "chair", "beds", "bed",
    "wardrobes", "wardrobe", "desks", "desk", "boxes", "box",
    "bags", "bag", "suitcases", "suitcase", "drawers", "chest of drawers",
    "bookcases", "bookcase", "fridges", "fridge", "freezers", "freezer",
    "washing machines", "washing machine", "tumble dryers", "tumble dryer",
    "televisions", "television", "TVs", "TV",
)
_ITEM_PATTERN = "|".join(
    re.escape(item) for item in sorted(_ITEM_NAMES, key=len, reverse=True)
)


def _load_memory(
    session_id: str | None,
    turns: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not session_id:
        return [], {}

    with session_scope() as sess:
        row = sess.get(ChatSession, session_id)
        if row is None:
            return [], {}

        history = row.history_json or []
        facts = row.facts_json or {}

    # Keep the configured number of prior user/assistant turns.
    return history[-(max(turns, 1) * 2):], facts


def _update_facts(
    facts: dict[str, Any],
    user_text: str,
) -> dict[str, Any]:
    """Extract a small set of explicit moving facts without an LLM call."""
    updated = dict(facts)
    text = " ".join(user_text.split())
    lower = text.lower()

    # Explicit property type, including straightforward corrections.
    property_match = re.search(
        r"\b(?:i live in|i'm in|i am in|it's a|it is a|we live in|"
        r"the property is a)\s+(flat|apartment|house|bungalow|maisonette)\b",
        lower,
    )
    if property_match:
        updated["property_type"] = property_match.group(1)

    # Explicit access details.
    if re.search(r"\b(no lift|without a lift|there is no lift)\b", lower):
        updated["lift"] = False
    elif re.search(r"\b(?:there is|we have|has|with) (?:a )?lift\b", lower):
        updated["lift"] = True

    if re.search(r"\b(?:no stairs|no steps)\b", lower):
        updated["stairs_or_steps"] = "none"
    else:
        access_match = re.search(
            r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"\s+(flights? of stairs|steps|stairs)\b",
            lower,
        )
        if access_match:
            updated["stairs_or_steps"] = (
                f"{access_match.group(1)} {access_match.group(2)}"
            )

    # Explicit postcodes with a stated role.
    postcode = r"[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}"
    for role, pattern in (
        ("pickup_postcode", rf"\b(?:pickup|pick-up|collection)\s+(?:postcode|post code)\s*(?:is|:)?\s*({postcode})"),
        ("destination_postcode", rf"\b(?:destination|delivery|drop-off)\s+(?:postcode|post code)\s*(?:is|:)?\s*({postcode})"),
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            updated[role] = " ".join(match.group(1).upper().split())

    # Explicit item counts; latest matching count replaces an earlier count.
    count_pattern = re.compile(
        rf"\b(?P<count>\d+|zero|no|one|a|an|two|three|four|five|six|seven|eight|nine|ten)"
        rf"\s+(?P<modifier>(?:(?:leather|fabric|large|small|medium|wooden|"
        rf"glass|double|single|king-size|king size)\s+){{0,3}})"
        rf"(?P<item>{_ITEM_PATTERN})\b",
        re.IGNORECASE,
    )
    for match in count_pattern.finditer(text):
        raw_count = match.group("count").lower()
        count = int(raw_count) if raw_count.isdigit() else _NUMBER_WORDS.get(raw_count)
        if count is None:
            continue

        item = match.group("item").lower()
        item_key = item if item.endswith("s") else f"{item}s"
        updated[f"item_count:{item_key}"] = count

        modifier = " ".join(match.group("modifier").lower().split())
        if modifier:
            updated[f"item_details:{item_key}"] = modifier

    # Explicit negatives must be remembered as answers, not treated as missing.
    for item in ("bags", "boxes"):
        if re.search(rf"\b(?:no|none|zero)\s+{item}\b", lower):
            updated[f"item_count:{item}"] = 0

    return updated


def _save_memory(
    session_id: str | None,
    history: list[dict[str, str]],
    facts: dict[str, Any],
) -> None:
    if not session_id:
        return

    # Keep a bounded transcript; structured facts remain stored separately.
    stored_history = history[-24:]
    table = ChatSession.__table__
    values = {
        "session_id": session_id,
        "history_json": stored_history,
        "facts_json": facts,
    }
    statement = mysql_insert(table).values(**values)
    statement = statement.on_duplicate_key_update(
        history_json=statement.inserted.history_json,
        facts_json=statement.inserted.facts_json,
        updated_at=__import__("sqlalchemy").func.current_timestamp(),
    )

    with session_scope() as sess:
        sess.execute(statement)
        sess.commit()


@router.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    cfg = load_config()
    started = time.monotonic()
    history, facts = _load_memory(req.session_id, cfg.chat.session_turns)
    updated_facts = _update_facts(facts, req.text)

    normalized_text = " ".join(req.text.lower().split()).strip(" .!?")

    # Keep the existing fast quote-start response, but persist it too.
    if normalized_text in QUOTE_START_PHRASES:
        reply = (
            "Sure, happy to help with a quote! Could you please share "
            "both postcodes - the pickup postcode and the delivery postcode?"
        )
        updated_history = history + [
            {"role": "user", "content": req.text},
            {"role": "assistant", "content": reply},
        ]
        _save_memory(req.session_id, updated_history, updated_facts)
        log_request("chat", req.session_id, [], False, None, started)
        return make_response(reply, [], False, None)

    stage_started = time.monotonic()
    hits = search(req.text)
    context_chunks = [hit.content for hit in hits]
    log.info(
        "CHAT_TIMING retrieval=%.2fs hits=%d",
        time.monotonic() - stage_started,
        len(hits),
    )

    stage_started = time.monotonic()
    messages = build_rag_messages(
        req.text,
        context_chunks,
        history,
        updated_facts,
    )
    log.info(
        "CHAT_TIMING prompt_preparation=%.2fs messages=%d",
        time.monotonic() - stage_started,
        len(messages),
    )

    stage_started = time.monotonic()
    client = get_llm_client()
    reply = client.generate(
        messages,
        temperature=cfg.generation.temperature,
        max_output_tokens=cfg.generation.max_output_tokens,
    )
    log.info(
        "CHAT_TIMING generation=%.2fs",
        time.monotonic() - stage_started,
    )

    updated_history = history + [
        {"role": "user", "content": req.text},
        {"role": "assistant", "content": reply},
    ]
    _save_memory(req.session_id, updated_history, updated_facts)

    log_request("chat", req.session_id, [], False, None, started)
    log.info("CHAT_TIMING total=%.2fs", time.monotonic() - started)
    return make_response(reply, [], False, None)
