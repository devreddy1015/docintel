from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from docintel.config import get_settings

SCHEMA = Path(__file__).with_name("schema.sql")


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


def ensure_database(url: str) -> None:
    """Create the database itself if missing (the server is shared with TenderLens)."""
    info = psycopg.conninfo.conninfo_to_dict(url)
    name = info.pop("dbname")
    with psycopg.connect(
        psycopg.conninfo.make_conninfo(**info, dbname="postgres"), autocommit=True
    ) as c:
        if not c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            c.execute(f'CREATE DATABASE "{name}"')


def migrate(url: str | None = None) -> None:
    url = url or get_settings().database_url
    ensure_database(url)
    with psycopg.connect(url, autocommit=True) as c:
        c.execute(SCHEMA.read_text())


@lru_cache(maxsize=1)
def pool() -> ConnectionPool:
    return ConnectionPool(
        get_settings().database_url, min_size=1, max_size=8, configure=_configure, open=True
    )


@contextmanager
def connection():
    with pool().connection() as conn:
        yield conn
