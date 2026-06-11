"""Alembic migration environment for Latos.

Uses `latos.persistence.schema.Base.metadata` so autogenerate works.

Database URL is taken from:
  1. `-x url=sqlite:///path/to/data.db` on the alembic command line, OR
  2. `LATOS_DB_URL` environment variable, OR
  3. `sqlite:///./latos.db` as a fallback (mostly for autogenerate dry-runs).
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, event, pool

from latos.persistence.schema import Base

# Alembic Config object — provides access to alembic.ini values.
config = context.config

# Configure Python logging from alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata target for autogenerate.
target_metadata = Base.metadata


def _resolve_db_url() -> str:
    """Pick a SQLite URL from CLI args, env, or default."""
    x_args = context.get_x_argument(as_dictionary=True)
    if "url" in x_args:
        return x_args["url"]
    env_url = os.environ.get("LATOS_DB_URL")
    if env_url:
        return env_url
    return "sqlite:///./latos.db"


def run_migrations_offline() -> None:
    """Generate SQL without connecting to a DB."""
    url = _resolve_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # required for SQLite ALTER TABLE support
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations to a live database."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _resolve_db_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    # Enable SQLite foreign keys via a connection event so the PRAGMA fires
    # for every connection without contaminating the migration transaction.
    @event.listens_for(connectable, "connect")
    def _enable_fk(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
