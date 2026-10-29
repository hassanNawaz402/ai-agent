from pathlib import Path

from app.utils.hashing import sha256_file, sha256_text


def test_sha256_text_is_stable():
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("world")


def test_sha256_file(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    h1 = sha256_file(p)
    assert len(h1) == 64
    p.write_text("hello world")
    h2 = sha256_file(p)
    assert h1 != h2
