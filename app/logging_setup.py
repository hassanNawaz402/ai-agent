from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from app.config import PROJECT_ROOT


_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(env: str) -> logging.Logger:
    log_dir = PROJECT_ROOT / "logs" / env
    log_dir.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if env == "dev" else logging.INFO
    formatter = logging.Formatter(_LOG_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_dir / "app.log", when="midnight", backupCount=14, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)
    root.addHandler(console)

    if env == "prod":
        err_handler = logging.handlers.TimedRotatingFileHandler(
            log_dir / "error.log", when="midnight", backupCount=30, encoding="utf-8"
        )
        err_handler.setFormatter(formatter)
        err_handler.setLevel(logging.ERROR)
        root.addHandler(err_handler)

    return root
