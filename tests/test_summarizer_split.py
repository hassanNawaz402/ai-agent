from app.summarize.email_summarizer import _split_by_words


def test_short_text_stays_single_chunk():
    assert _split_by_words("one two three", 100) == ["one two three"]


def test_long_text_splits_by_word_count():
    text = " ".join(f"w{i}" for i in range(2500))
    chunks = _split_by_words(text, 1000)
    assert len(chunks) == 3
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w2499"


def test_chunk_boundary_is_clean():
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = _split_by_words(text, 400)
    assert len(chunks) == 3  # 400 + 400 + 200
    assert chunks[0].split()[-1] == "w399"
    assert chunks[1].split()[0] == "w400"
