from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from rag_api.core.config import get_settings
from rag_api.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
database_url = os.getenv("DATABASE_URL") or settings.DATABASE_URL
test_database_url = os.getenv("TEST_DATABASE_URL") or settings.TEST_DATABASE_URL
x_args = context.get_x_argument(as_dictionary=True)
requested_dburl = x_args.get("dburl")


def _resolve_dburl(selector: str | None) -> str:
    if selector is None or selector == "main":
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not set. Provide it in env/.env or pass -x dburl=<url>."
            )
        return database_url

    if selector == "test":
        if not test_database_url:
            raise RuntimeError("TEST_DATABASE_URL is not set. Provide it in env/.env.")
        return test_database_url

    env_selected = os.getenv(selector)
    if env_selected:
        return env_selected

    if "://" in selector:
        return selector

    raise RuntimeError(
        "Unknown -x dburl selector. Use main|test, an env var name, or a full SQLAlchemy URL."
    )


config.set_main_option("sqlalchemy.url", _resolve_dburl(requested_dburl))

target_metadata = Base.metadata if Base is not None else None


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
