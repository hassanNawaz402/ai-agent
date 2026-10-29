from __future__ import annotations

import logging
from pathlib import Path


log = logging.getLogger(__name__)

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv"}


def parse(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        from app.parsers.text import parse as f
    elif ext == ".pdf":
        from app.parsers.pdf import parse as f
    elif ext in (".docx", ".doc"):
        from app.parsers.docx import parse as f
    elif ext in (".xlsx", ".xls", ".csv"):
        from app.parsers.spreadsheet import parse as f
    else:
        raise ValueError(f"unsupported file type: {ext}")
    try:
        return f(path)
    except Exception as exc:
        log.exception("parser failed for %s: %s", path, exc)
        return ""
