"""Async Postgres connection pool.

Atlas talks to Postgres with psycopg 3 and hand-written SQL rather than an ORM.
Rationale in Decision.md (ADR-0005), briefly: the queries that matter here are
vector-distance queries with pgvector operators and, from Phase 2, a hybrid
lexical/vector fusion. Those are not expressible in an ORM without dropping to
raw SQL anyway, and the ORM layer would then be hiding the one thing worth
reading in this codebase.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


async def _configure(conn: AsyncConnection) -> None:
    """Per-connection setup.

    The vector type has to be registered on every connection, not once
    globally: psycopg looks up the type OID per connection, and pgvector's OID
    is not fixed (it is assigned when the extension is created).
    """
    conn.row_factory = dict_row  # type: ignore[assignment]
    await register_vector_async(conn)


class Database:
    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 10) -> None:
        self._pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            configure=_configure,
            # Do not connect at construction: the API process must start even if
            # Postgres is briefly unavailable, and fail requests rather than
            # fail to boot.
            open=False,
            kwargs={"autocommit": True},
        )

    async def open(self) -> None:
        await self._pool.open(wait=True, timeout=10.0)

    async def close(self) -> None:
        await self._pool.close()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncConnection]:
        async with self._pool.connection() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncConnection]:
        """A connection wrapped in an explicit transaction.

        Ingestion needs this: writing a new document version's chunks and
        deleting the previous version's must be atomic, or a concurrent query
        can observe a document with no retrievable content.
        """
        async with self._pool.connection() as conn, conn.transaction():
            yield conn

    async def healthy(self) -> bool:
        try:
            async with self.connection() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            logger.exception("database health check failed")
            return False
