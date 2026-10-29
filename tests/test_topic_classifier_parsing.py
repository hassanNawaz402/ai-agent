from app.topics.classifier import _parse_topics


ALLOWED = ["company information", "booking request", "payment information"]


def test_parses_clean_json_array():
    assert _parse_topics('["booking request"]', ALLOWED) == ["booking request"]


def test_handles_prose_wrapper():
    raw = 'Here is the answer: ["company information", "payment information"] — done.'
    assert _parse_topics(raw, ALLOWED) == ["company information", "payment information"]


def test_filters_out_unknown_topics():
    raw = '["booking request", "shoe size"]'
    assert _parse_topics(raw, ALLOWED) == ["booking request"]


def test_case_insensitive_match():
    raw = '["BOOKING REQUEST", "Payment Information"]'
    assert _parse_topics(raw, ALLOWED) == ["booking request", "payment information"]


def test_dedupe():
    raw = '["booking request", "booking request"]'
    assert _parse_topics(raw, ALLOWED) == ["booking request"]


def test_no_json_returns_empty():
    assert _parse_topics("sorry, I cannot classify this", ALLOWED) == []


def test_malformed_json_returns_empty():
    assert _parse_topics("[not valid]", ALLOWED) == []
