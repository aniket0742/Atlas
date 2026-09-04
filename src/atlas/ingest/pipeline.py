"""The ingestion pipeline: parse -> normalise -> chunk -> embed -> index.

Phase 1 runs this synchronously, inside the request. That is a deliberate
starting point, not an oversight: it keeps the pipeline observable and
debuggable while its behaviour is still being established, and every step is
already written so it can be moved behind a queue without changing.

Phase 3 moves the call site to a worker. What makes that a small change rather
than a rewrite is the two properties built in here:

  * Idempotency. Ids are derived from content identity, so running the pipeline
    twice on the same document converges rather than duplicating.
  * Content-hash short-circuit. Re-ingesting unchanged content does no parsing,
    no embedding and no writes, which is what makes a re-crawl affordable.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from atlas.config import Settings
from atlas.core.models import IngestResult, ParsedDocument
from atlas.db import repository as repo
from atlas.db.pool import Database
from atlas.ingest import parsers
from atlas.ingest.chunking import chunk_document
from atlas.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestRequest:
    data: bytes
    external_id: str
    source_name: str = "default"
    filename: str | None = None
    mime_type: str | None = None
    uri: str | None = None
    metadata: dict[str, Any] | None = None


class Ingestor:
    def __init__(self, db: Database, embedder: EmbeddingProvider, settings: Settings) -> None:
        self._db = db
        self._embedder = embedder
        self._settings = settings

    async def ingest(
        self,
        tenant_id: uuid.UUID,
        request: IngestRequest,
        *,
        source_id: uuid.UUID | None = None,
    ) -> IngestResult:
        """Run the full pipeline for one document.

        `source_id` lets a caller that already resolved the source skip the
        lookup -- the worker knows it from the job row. Callers that only have a
        name (the CLI, a direct upload) leave it None and the source is created
        on demand.
        """
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        parsed = parsers.parse(
            request.data,
            external_id=request.external_id,
            filename=request.filename,
            mime_type=request.mime_type,
            uri=request.uri,
            metadata=request.metadata,
        )
        timings["parse_ms"] = (time.perf_counter() - t0) * 1000

        # Reserve the version first, in its own transaction, so the unchanged
        # case does no further work and holds no locks while embedding runs.
        async with self._db.transaction() as conn:
            if source_id is None:
                source_id = await repo.ensure_source(conn, tenant_id, request.source_name)
            document_id, version, changed = await repo.begin_document_version(
                conn, tenant_id, source_id, parsed
            )

        if not changed:
            logger.info(
                "ingest skipped document_id=%s hash unchanged (%s)",
                document_id,
                parsed.content_hash[:12],
            )
            return IngestResult(
                document_id=document_id, version=version, chunk_count=0, changed=False
            )

        try:
            chunks, embeddings, step_timings = await self._build_index_payload(parsed)
            timings.update(step_timings)

            t0 = time.perf_counter()
            async with self._db.transaction() as conn:
                await repo.write_chunks(
                    conn,
                    tenant_id,
                    document_id,
                    version,
                    chunks,
                    embeddings,
                    self._embedder.model_id,
                )
            timings["index_ms"] = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            # Record the failure against the document rather than losing it.
            # The document stays queryable as 'failed' with its error, which is
            # what makes "where did ingestion break" answerable later.
            logger.exception("ingest failed document_id=%s", document_id)
            async with self._db.transaction() as conn:
                await repo.mark_document_failed(conn, tenant_id, document_id, str(exc))
            raise

        logger.info(
            "ingested document_id=%s version=%s chunks=%s timings=%s",
            document_id,
            version,
            len(chunks),
            {k: round(v, 1) for k, v in timings.items()},
        )
        return IngestResult(
            document_id=document_id,
            version=version,
            chunk_count=len(chunks),
            changed=True,
        )

    async def _build_index_payload(
        self, parsed: ParsedDocument
    ) -> tuple[list, list[list[float]], dict[str, float]]:
        import asyncio

        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        chunks = chunk_document(
            parsed.content,
            self._embedder,
            target_tokens=self._settings.chunk_target_tokens,
            overlap_tokens=self._settings.chunk_overlap_tokens,
            min_tokens=self._settings.chunk_min_tokens,
        )
        timings["chunk_ms"] = (time.perf_counter() - t0) * 1000

        if not chunks:
            return [], [], timings

        # ONNX inference is CPU-bound and releases the GIL inside the runtime,
        # but the surrounding Python is not async. Pushing it to the default
        # executor keeps the event loop responsive while a large document is
        # embedding -- without this, one 200-page PDF stalls every other request.
        t0 = time.perf_counter()
        embeddings = await asyncio.to_thread(
            self._embedder.embed_documents, [c.text for c in chunks]
        )
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000

        return chunks, embeddings, timings
