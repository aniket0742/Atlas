"""Migration runner.

Plain numbered .sql files applied in order, tracked in a schema_migrations
table. No Alembic. The reasoning is in Decision.md (ADR-0006): the schema's
interesting parts are pgvector index definitions with tuning parameters
(m, ef_construction) and, later, generated tsvector columns. Alembic's
autogenerate does not model those, so they would be hand-written in migration
files regardless -- at which point Alembic is contributing a dependency and a
directory of boilerplate rather than leverage.

Each migration runs inside a transaction: Postgres has transactional DDL, so a
migration that fails halfway leaves no partial schema.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from psycopg import AsyncConnection

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    checksum    text NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
)
"""


def discover(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"No migrations directory at {directory}")
    return sorted(directory.glob("[0-9]*.sql"))


async def apply_all(dsn: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every unapplied migration. Returns the versions applied."""
    applied: list[str] = []
    async with await AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute(_BOOTSTRAP)

        cur = await conn.execute("SELECT version, checksum FROM schema_migrations")
        known = {row[0]: row[1] for row in await cur.fetchall()}

        for path in discover(directory):
            version = path.stem
            sql = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()

            if version in known:
                if known[version] != checksum:
                    # An applied migration was edited. Silently ignoring this
                    # means the database and the repository disagree about the
                    # schema, which is worse than refusing to start.
                    raise RuntimeError(
                        f"Migration {version} was modified after it was applied "
                        f"(recorded checksum {known[version][:12]}, file is "
                        f"{checksum[:12]}). Write a new migration instead."
                    )
                continue

            logger.info("applying migration %s", version)
            async with conn.transaction():
                await conn.execute(sql)  # type: ignore[arg-type]
                await conn.execute(
                    "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                    (version, checksum),
                )
            applied.append(version)

    return applied


async def status(dsn: str, directory: Path = MIGRATIONS_DIR) -> list[tuple[str, bool]]:
    async with await AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute(_BOOTSTRAP)
        cur = await conn.execute("SELECT version FROM schema_migrations")
        known = {row[0] for row in await cur.fetchall()}
    return [(p.stem, p.stem in known) for p in discover(directory)]
