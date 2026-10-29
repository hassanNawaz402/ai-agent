from app.rag.chunker import chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n") == []


def test_small_text_yields_single_chunk():
    chunks = chunk_text("hello world one two three", chunk_size_tokens=512, overlap_tokens=64)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == "hello world one two three"
    assert chunks[0].token_count == 5


def test_long_text_splits_with_overlap():
    words = [f"w{i}" for i in range(1200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size_tokens=500, overlap_tokens=100)
    assert len(chunks) >= 3
    # chunk_size=500, overlap=100 → step=400, so chunks share 100 words.
    a = chunks[0].text.split()
    b = chunks[1].text.split()
    assert a[-100:] == b[:100]


def test_chunk_indices_are_sequential():
    chunks = chunk_text(" ".join([f"w{i}" for i in range(2000)]))
    assert [c.index for c in chunks] == list(range(len(chunks)))
