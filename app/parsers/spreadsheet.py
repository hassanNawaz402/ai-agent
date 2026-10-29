from __future__ import annotations

from pathlib import Path


def parse(path: Path) -> str:
    import pandas as pd  # lazy import

    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        return df.to_csv(index=False)

    # xls / xlsx — read all sheets, concat with a header per sheet.
    sheets = pd.read_excel(path, sheet_name=None)
    parts: list[str] = []
    for name, df in sheets.items():
        parts.append(f"# Sheet: {name}")
        parts.append(df.to_csv(index=False))
    return "\n".join(parts)
