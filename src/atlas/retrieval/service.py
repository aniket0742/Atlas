"""Retrieval.

Phase 1 is dense retrieval only, deliberately. The point of the phase gate is
that hybrid search and reranking get added in Phase 2 *against a measured
baseline*, so that "hybrid retrieval improved Recall@5 by X" is a claim backed
by two numbers rather than an assumption inherited from a blog post.

The one non-obvious piece here is the similarity floor. Retrieval always returns
its top-k; without a floor, a question with no answer in the corpus still yields
five confident-looking chunks, and the model is then asked to answer from
irrelevant evidence. The floor is what turns "no answer exists" into an explicit
refusal instead of a fabrication.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

from atlas.config import Settings
from atlas.core.models import RetrievedChunk
from atlas.db import repository as repo
from atlas.db.pool import Database
from atlas.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    # Everything the search returned, before the similarity floor was applied.
    # Kept so the eval harness can measure how much the floor costs in recall,
    # and so /query can explain an empty result rather than just returning none.
    candidates: list[RetrievedChunk]
    timings_ms: dict[str, float]


class Retriever:
    def __init__(self, db: Database, embedder: EmbeddingProvider, settings: Settings) -> None:
        self._db = db
        self._embedder = embedder
        self._settings = settings

    @property
    def embedder(self) -> EmbeddingProvider:
        """Exposed so a report can record which model produced its numbers."""
        return self._embedder

    async def retrieve(
        self,
        tenant_id: uuid.UUID,
        query: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
        source_ids: list[uuid.UUID] | None = None,
    ) -> RetrievalResult:
        k = top_k or self._settings.retrieval_top_k
        floor = self._settings.min_similarity if min_similarity is None else min_similarity
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        embedding = await asyncio.to_thread(self._embedder.embed_query, query)
        timings["embed_query_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        async with self._db.connection() as conn:
            candidates = await repo.search_dense(
                conn,
                tenant_id,
                embedding,
                model=self._embedder.model_id,
                limit=k,
                source_ids=source_ids,
            )
        timings["search_ms"] = (time.perf_counter() - t0) * 1000

        kept = [c for c in candidates if c.score >= floor]
        if candidates and not kept:
            logger.info(
                "retrieval floor rejected all %s candidates (best=%.3f floor=%.3f)",
                len(candidates),
                candidates[0].score,
                floor,
            )

        return RetrievalResult(chunks=kept, candidates=candidates, timings_ms=timings)
