from argparse import Namespace
import os
import uuid
from collections.abc import Iterator

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("alembic")

from alembic import command
from alembic.config import Config
from rag_api.core.config import get_settings
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url

EXPECTED_TABLES = {"documents", "chunks", "query_logs", "ingestion_jobs"}


def _quote_ident(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


@pytest.fixture
def migrated_database_url() -> Iterator[str]:
    test_database_url = os.getenv("TEST_DATABASE_URL") or get_settings().TEST_DATABASE_URL
    if not test_database_url:
        pytest.skip("TEST_DATABASE_URL is not set.")

    base_url = make_url(test_database_url)
    db_name = f"{base_url.database}_migr_{uuid.uuid4().hex[:8]}"
    fresh_url = base_url.set(database=db_name)
    fresh_url_str = fresh_url.render_as_string(hide_password=False)
    admin_url: URL = base_url.set(database="postgres")

    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE {_quote_ident(db_name)}"))

        alembic_cfg = Config("alembic.ini")
        alembic_cfg.cmd_opts = Namespace(x=[f"dburl={fresh_url_str}"])
        command.upgrade(alembic_cfg, "head")

        yield fresh_url_str
    finally:
        with admin_engine.connect() as conn:
            conn.execute(
                text(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = :db_name AND pid <> pg_backend_pid()
                    """
                ),
                {"db_name": db_name},
            )
            conn.execute(text(f"DROP DATABASE IF EXISTS {_quote_ident(db_name)}"))
        admin_engine.dispose()


def test_pgvector_extension_exists(migrated_database_url: str) -> None:
    engine = create_engine(migrated_database_url)
    try:
        with engine.connect() as conn:
            extension = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one_or_none()
    finally:
        engine.dispose()

    assert extension == "vector"


def test_expected_tables_exist(migrated_database_url: str) -> None:
    engine = create_engine(migrated_database_url)
    try:
        with engine.connect() as conn:
            table_names = set(
                conn.execute(
                    text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                ).scalars()
            )
    finally:
        engine.dispose()

    assert EXPECTED_TABLES.issubset(table_names)
