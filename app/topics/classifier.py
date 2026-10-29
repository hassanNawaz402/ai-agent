from __future__ import annotations

import json
import logging
import re


log = logging.getLogger(__name__)



_CLASSIFIER_SYSTEM = (
    "You are a topic classifier for Advantage Removals, a removal services company. "
    "Classify the user's message into one or more of the allowed topics below. "
    "Choose a topic when the message is relevant to it, even if the wording is indirect. "
    "Company information includes questions about the company's services, what it does, "
    "service areas, company details, and customer support. "
    "Booking request includes booking, scheduling, availability, and moving dates. "
    "Payment information includes prices, quotes, fees, invoices, and payment methods. "
    "Only select topics from the supplied allowed-topic list. "
    "If the message is unrelated to all allowed topics, return an empty JSON array. "
    "Return only a JSON array of exact topic names, with no explanation."
)



def _build_prompt(text: str, topics: list[str]) -> list[dict[str, str]]:
    bullets = "\n".join(f"- {t}" for t in topics)
    system = f"{_CLASSIFIER_SYSTEM}\n\nAllowed topics:\n{bullets}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]




def _parse_topics(raw: str, allowed: list[str]) -> list[str]:
    # First try valid JSON, such as:
    # ["company information", "booking request"]
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        data = None

    # Accept a single topic returned as a JSON string.
    if isinstance(data, str):
        data = [data]

    # If JSON parsing failed, try to extract exact allowed
    # topic names from the model's response.
    elif not isinstance(data, list):
        cleaned = raw.strip().strip('"\'` ')

        # First handle a plain, exact topic name.
        allowed_lookup = {topic.casefold(): topic for topic in allowed}
        exact_match = allowed_lookup.get(cleaned.casefold())

        if exact_match is not None:
            data = [exact_match]
        else:
            # Handle responses like:
            # [company information, booking request]
            # Match only configured topic names.
            data = []
            for topic in allowed:
                pattern = (
                    r"(?<![\w])"
                    + re.escape(topic)
                    + r"(?![\w])"
                )
                if re.search(pattern, cleaned, flags=re.IGNORECASE):
                    data.append(topic)

    allowed_lc = {topic.casefold(): topic for topic in allowed}
    result: list[str] = []
    seen: set[str] = set()

    for item in data:
        if not isinstance(item, str):
            continue

        topic = allowed_lc.get(item.strip().casefold())
        if topic is not None and topic not in seen:
            seen.add(topic)
            result.append(topic)

    return result


def classify_topics(text: str) -> list[str]:
    from app.config import load_config  # lazy
    from app.llm.factory import get_llm_client  # lazy

    cfg = load_config()
    topics = cfg.extract_topics
    if not topics:
        return []
    client = get_llm_client()
    msgs = _build_prompt(text, topics)
    raw = client.generate(
        msgs, temperature=0.0, max_output_tokens=128,
    )
    matched = _parse_topics(raw, topics)

    log.info(
        "topic classifier: text=%r raw=%r matched=%s",
        text[:80],
        raw,
        matched,
    )

    return matched
