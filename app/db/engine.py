from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import PROJECT_ROOT, AppConfig, load_config


log = logging.getLogger(__name__)
INIT_SQL_PATH = PROJECT_ROOT / "scripts" / "init_db.sql"


def _dsn(cfg: AppConfig, with_db: bool) -> str:
    db = cfg.database
    tail = f"/{db.name}" if with_db else ""
    return (
        f"mysql+pymysql://{db.user}:{db.password}@{db.host}:{db.port}{tail}"
        f"?charset=utf8mb4"
    )


def ensure_database_exists(cfg: AppConfig) -> None:
    """Create the application database if it is missing. Idempotent."""
    db = cfg.database
    conn = pymysql.connect(
        host=db.host, port=db.port, user=db.user, password=db.password,
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db.name}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()
    finally:
        conn.close()


def run_init_sql(engine: Engine) -> None:
    sql = INIT_SQL_PATH.read_text()
    # Split on ';' and execute each non-empty statement.
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    cfg = load_config()
    return create_engine(_dsn(cfg, with_db=True), pool_pre_ping=True, future=True)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def session_scope() -> Session:
    return get_session_factory()()


def initialize_database() -> None:
    """Apply the initialization SQL to the existing application database."""
    cfg = load_config()
    log.info("applying init_db.sql to database '%s'", cfg.database.name)
    run_init_sql(get_engine())


def check_connection() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        log.warning("db connection check failed: %s", exc)
        return False
