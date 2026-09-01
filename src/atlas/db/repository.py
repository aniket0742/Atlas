"""Data access.

Every function here takes a tenant_id and every query filters on it. That is
deliberate and it is enforced by shape rather than by convention: there is no
"get chunk by id" that omits the tenant. Phase 1 runs with a single tenant, so
none of this is load-bearing yet -- which is exactly why it has to be built now,
while it is free, rather than retrofitted in Phase 5 across a codebase that has
learned to assume a global namespace.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from psycopg import AsyncConnection

from atlas.core import ids
from atlas.core.models import Chunk, ParsedDocument, RetrievedChunk

# ---------------------------------------------------------------------------
# Tenants and sources
# ---------------------------------------------------------------------------


async def ensure_tenant(conn: AsyncConnection, slug: str, name: str | None = None) -> uuid.UUID:
    tenant_id = ids.tenant_id(slug)
    await conn.execute(
        """
        INSERT INTO tenants (id, slug, name) VALUES (%s, %s, %s)
        ON CONFLICT (id) DO NOTHING
        """,
        (tenant_id, slug, name or slug),
    )
    return tenant_id


async def ensure_source(
    conn: AsyncConnection,
    tenant_id: uuid.UUID,
    name: str,
    kind: str = "upload",
    config: dict[str, Any] | None = None,
) -> uuid.UUID:
    source_id = ids.source_id(tenant_id, name)
    await conn.execute(
        """
        INSERT INTO sources (id, tenant_id, kind, name, config)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET config = EXCLUDED.config
        """,
        (source_id, tenant_id, kind, name, json.dumps(config or {})),
    )
    return source_id


async def list_sources(conn: AsyncConnection, tenant_id: uuid.UUID) -> list[dict[str, Any]]:
    cur = await conn.execute(
        """
        SELECT s.id, s.name, s.kind, s.created_at,
               count(d.id) AS document_count
        FROM sources s
        LEFT JOIN documents d ON d.source_id = s.id AND d.tenant_id = s.tenant_id
        WHERE s.tenant_id = %s
        GROUP BY s.id
        ORDER BY s.name
        """,
        (tenant_id,),
    )
    return await cur.fetchall()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


async def begin_document_version(
    conn: AsyncConnection,
    tenant_id: uuid.UUID,
    source_id: uuid.UUID,
    parsed: ParsedDocument,
) -> tuple[uuid.UUID, int, bool]:
    """Reserve a document version for this content.

    Returns (document_id, version, changed). When `changed` is False the stored
    content hash already matches and the caller should skip the rest of the
    pipeline -- that skip is what makes re-ingesting an unchanged corpus cheap.

    Must be called inside a transaction. It takes a row lock on the existing
    document so two concurrent ingests of the same document serialise rather
    than both allocating version N+1.
    """
    document_id = ids.document_id(tenant_id, source_id, parsed.external_id)

    cur = await conn.execute(
        """
        SELECT content_hash, version FROM documents
        WHERE id = %s AND tenant_id = %s
        FOR UPDATE
        """,
        (document_id, tenant_id),
    )
    existing = await cur.fetchone()

    if existing is not None and existing["content_hash"] == parsed.content_hash:
        return document_id, int(existing["version"]), False

    version = int(existing["version"]) + 1 if existing else 1

    await conn.execute(
        """
        INSERT INTO documents (
            id, tenant_id, source_id, external_id, uri, title, mime_type,
            content_hash, version, content, byte_size, metadata, status, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', now())
        ON CONFLICT (id) DO UPDATE SET
            uri          = EXCLUDED.uri,
            title        = EXCLUDED.title,
            mime_type    = EXCLUDED.mime_type,
            content_hash = EXCLUDED.content_hash,
            version      = EXCLUDED.version,
            content      = EXCLUDED.content,
            byte_size    = EXCLUDED.byte_size,
            metadata     = EXCLUDED.metadata,
            status       = 'pending',
            error        = NULL,
            updated_at   = now()
        """,
        (
            document_id,
            tenant_id,
            source_id,
            parsed.external_id,
            parsed.uri,
            parsed.title,
            parsed.mime_type,
            parsed.content_hash,
            version,
            parsed.content,
            parsed.byte_size,
            json.dumps(parsed.metadata),
        ),
    )
    return document_id, version, True


async def write_chunks(
    conn: AsyncConnection,
    tenant_id: uuid.UUID,
    document_id: uuid.UUID,
    version: int,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    model: str,
) -> None:
    """Write a version's chunks and embeddings, replacing any previous version.

    Must run inside a transaction. The delete-then-insert order matters less
    than the atomicity: a reader either sees the whole old version or the whole
    new one, never a document with zero chunks.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")

    # Removing other versions cascades to chunk_embeddings.
    await conn.execute(
        """
        DELETE FROM chunks
        WHERE document_id = %s AND tenant_id = %s AND document_version <> %s
        """,
        (document_id, tenant_id, version),
    )

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk_id = ids.chunk_id(document_id, version, chunk.ordinal)
        await conn.execute(
            """
            INSERT INTO chunks (
                id, tenant_id, document_id, document_version, ordinal, text,
                token_count, char_start, char_end, heading_path, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                text        = EXCLUDED.text,
                token_count = EXCLUDED.token_count,
                char_start  = EXCLUDED.char_start,
                char_end    = EXCLUDED.char_end,
                heading_path = EXCLUDED.heading_path,
                metadata    = EXCLUDED.metadata
            """,
            (
                chunk_id,
                tenant_id,
                document_id,
                version,
                chunk.ordinal,
                chunk.text,
                chunk.token_count,
                chunk.char_start,
                chunk.char_end,
                chunk.heading_path,
                json.dumps(chunk.metadata),
            ),
        )
        await conn.execute(
            """
            INSERT INTO chunk_embeddings (chunk_id, tenant_id, model, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (chunk_id, model) DO UPDATE SET embedding = EXCLUDED.embedding
            """,
            (chunk_id, tenant_id, model, embedding),
        )

    await conn.execute(
        """
        UPDATE documents
        SET status = 'indexed', indexed_at = now(), updated_at = now(), error = NULL
        WHERE id = %s AND tenant_id = %s
        """,
        (document_id, tenant_id),
    )


async def mark_document_failed(
    conn: AsyncConnection, tenant_id: uuid.UUID, document_id: uuid.UUID, error: str
) -> None:
    await conn.execute(
        """
        UPDATE documents SET status = 'failed', error = %s, updated_at = now()
        WHERE id = %s AND tenant_id = %s
        """,
        (error[:2000], document_id, tenant_id),
    )


async def get_document(
    conn: AsyncConnection, tenant_id: uuid.UUID, document_id: uuid.UUID
) -> dict[str, Any] | None:
    cur = await conn.execute(
        """
        SELECT d.id, d.external_id, d.uri, d.title, d.mime_type, d.content_hash,
               d.version, d.byte_size, d.metadata, d.status, d.error,
               d.created_at, d.updated_at, d.indexed_at,
               s.name AS source_name,
               (SELECT count(*) FROM chunks c
                 WHERE c.document_id = d.id AND c.document_version = d.version) AS chunk_count
        FROM documents d
        JOIN sources s ON s.id = d.source_id
        WHERE d.id = %s AND d.tenant_id = %s
        """,
        (document_id, tenant_id),
    )
    return await cur.fetchone()  # type: ignore[return-value]


async def get_document_content(
    conn: AsyncConnection, tenant_id: uuid.UUID, document_id: uuid.UUID
) -> str | None:
    cur = await conn.execute(
        "SELECT content FROM documents WHERE id = %s AND tenant_id = %s",
        (document_id, tenant_id),
    )
    row = await cur.fetchone()
    return row["content"] if row else None


async def list_documents(
    conn: AsyncConnection, tenant_id: uuid.UUID, limit: int = 50, offset: int = 0
) -> list[dict[str, Any]]:
    cur = await conn.execute(
        """
        SELECT d.id, d.external_id, d.title, d.uri, d.status, d.version,
               d.indexed_at, s.name AS source_name
        FROM documents d
        JOIN sources s ON s.id = d.source_id
        WHERE d.tenant_id = %s
        ORDER BY d.updated_at DESC
        LIMIT %s OFFSET %s
        """,
        (tenant_id, limit, offset),
    )
    return await cur.fetchall()  # type: ignore[return-value]


async def corpus_stats(conn: AsyncConnection, tenant_id: uuid.UUID) -> dict[str, Any]:
    cur = await conn.execute(
        """
        SELECT
            (SELECT count(*) FROM documents WHERE tenant_id = %(t)s) AS documents,
            (SELECT count(*) FROM documents WHERE tenant_id = %(t)s AND status = 'indexed')
                AS indexed_documents,
            (SELECT count(*) FROM documents WHERE tenant_id = %(t)s AND status = 'failed')
                AS failed_documents,
            (SELECT count(*) FROM chunks WHERE tenant_id = %(t)s) AS chunks,
            (SELECT count(*) FROM chunk_embeddings WHERE tenant_id = %(t)s) AS embeddings,
            (SELECT count(*) FROM sources WHERE tenant_id = %(t)s) AS sources
        """,
        {"t": tenant_id},
    )
    return await cur.fetchone()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def search_dense(
    conn: AsyncConnection,
    tenant_id: uuid.UUID,
    query_embedding: list[float],
    model: str,
    limit: int,
    *,
    source_ids: list[uuid.UUID] | None = None,
) -> list[RetrievedChunk]:
    """Nearest-neighbour search over chunk embeddings, scoped to one tenant.

    `<=>` is pgvector's cosine *distance*, so it sorts ascending and similarity
    is 1 - distance. Both models used here emit unit-norm vectors, so cosine and
    inner product would rank identically; cosine is used because the HNSW index
    is built with vector_cosine_ops.

    Caveat, recorded in ADR-0003: the HNSW index is searched before the tenant
    and source predicates are applied, so a highly selective filter can return
    fewer than `limit` rows even when more matching rows exist. With one tenant
    this cannot bite. Phase 5 revisits it with measurements.
    """
    filters = ["c.tenant_id = %(tenant)s", "ce.model = %(model)s"]
    params: dict[str, Any] = {
        "tenant": tenant_id,
        "model": model,
        "embedding": query_embedding,
        "limit": limit,
    }
    if source_ids:
        filters.append("d.source_id = ANY(%(source_ids)s)")
        params["source_ids"] = source_ids

    cur = await conn.execute(
        f"""
        SELECT c.id AS chunk_id, c.document_id, c.ordinal, c.text,
               c.char_start, c.char_end, c.heading_path,
               d.external_id AS document_external_id,
               d.title AS document_title, d.uri AS document_uri,
               s.name AS source_name,
               1 - (ce.embedding <=> %(embedding)s::vector) AS similarity
        FROM chunk_embeddings ce
        JOIN chunks c    ON c.id = ce.chunk_id
        JOIN documents d ON d.id = c.document_id AND d.version = c.document_version
        JOIN sources s   ON s.id = d.source_id
        WHERE {" AND ".join(filters)}
        ORDER BY ce.embedding <=> %(embedding)s::vector
        LIMIT %(limit)s
        """,
        params,
    )
    rows = await cur.fetchall()
    return [
        RetrievedChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            document_external_id=row["document_external_id"],
            document_title=row["document_title"],
            document_uri=row["document_uri"],
            source_name=row["source_name"],
            ordinal=row["ordinal"],
            text=row["text"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            heading_path=list(row["heading_path"] or []),
            score=float(row["similarity"]),
            component_scores={"dense": float(row["similarity"])},
        )
        for row in rows
    ]
