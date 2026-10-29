from app.llm.backends.mock import MockQwenClient


def test_mock_returns_classification_for_classifier_prompt():
    client = MockQwenClient()
    system = (
        "You are a strict topic classifier. "
        "Return ONLY a JSON array of the allowed topics the message is about.\n\n"
        "Allowed topics:\n- booking request\n- company information\n"
    )
    out = client.generate(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "I want to make a booking request for next week"},
        ]
    )
    assert "booking request" in out


def test_mock_summarizes_by_truncation():
    client = MockQwenClient()
    text = " ".join(f"word{i}" for i in range(300))
    out = client.generate(
        [
            {"role": "system", "content": "summarize this email"},
            {"role": "user", "content": text},
        ]
    )
    assert len(out.split()) <= 100


def test_mock_default_reply():
    client = MockQwenClient()
    out = client.generate(
        [
            {"role": "system", "content": "answer questions"},
            {"role": "user", "content": "hello"},
        ]
    )
    assert "Mock backend" in out or "brief answer" in out
