from __future__ import annotations

from pathlib import Path


def parse(path: Path) -> str:
    from pypdf import PdfReader  # lazy import

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)
