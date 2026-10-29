from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from app.api import chat, documents, email, health
from app.config import load_config
from app.db.engine import initialize_database
from app.llm.factory import startup_connectivity_check
from app.logging_setup import configure_logging
from app.rag.indexer import scan_and_index


def create_app() -> FastAPI:
    config = load_config()
    configure_logging(config.active_env)

    log = logging.getLogger("app.main")
    log.info(
        "starting app env=%s provider=%s backend=%s model=%s",
        config.active_env,
        config.active_profile.provider,
        config.active_profile.backend,
        config.active_profile.model,
    )

    initialize_database()
    startup_connectivity_check()

    try:
        scan_and_index()
    except Exception as exc:
        log.exception("initial indexing failed (non-fatal): %s", exc)

    app = FastAPI(title="Advantage Removals LLM API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:8080",
            "http://localhost:8080",
             "https://ai-agent.advantageremovals.co.uk",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(email.router)
    app.include_router(documents.router)
    return app


app = create_app()
