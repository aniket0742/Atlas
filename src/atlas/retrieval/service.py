"""Retrieval.

Three modes, selectable per request so the eval harness can compare them
directly:

* `dense`   — vector similarity only. The Phase 1 baseline.
* `lexical` — PostgreSQL full-text search only. Mostly a diagnostic: useful for
  attributing a hybrid result to its components, rarely the right production
  setting on its own.
* `hybrid`  — both, fused by reciprocal rank (see `fusion.py`).

Optionally followed by cross-encoder reranking, which reorders a wider candidate
set by scoring each (query, chunk) pair directly instead of comparing two
independently-produced vectors.

## The similarity floor moved

In Phase 1 the floor filtered individual chunks by cosine similarity. That does
not survive fusion: an RRF score is a sum of reciprocal ranks, not a similarity,
so a threshold calibrated on cosine has no meaning against it, and applying one
anyway would silently change refusal behaviour while every retrieval metric
looked fine.

The floor is now a **query-level gate evaluated on the dense candidates before
fusion**: if no dense candidate reaches the floor, the query is treated as
having no evidence and answering refuses. Otherwise the fused ranking is
returned untouched.

This matches what the floor was always for -- deciding whether the corpus can
answer at all, which is a property of the query, not of each chunk. It is also
the only formulation that keeps working when the final ranking is produced by
fusion or by a reranker. See ADR-0019.

In `lexical` mode there is no dense score, so no gate applies; that mode is for
measurement, not for serving.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from atlas.config import Settings
from atlas.core.models import RetrievedChunk
from atlas.db import repository as repo
from atlas.db.pool import Database
from atlas.providers.base import EmbeddingProvider, RerankProvider
from atlas.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

RetrievalMode = Literal["dense", "lexical", "hybrid"]


@dataclass(slots=True)
class RetrievalResult:
    """Evidence for a query, plus everything needed to explain it."""

    # What answering should use: the final ranking, or empty when the floor gate
    # rejected the query.
    chunks: list[RetrievedChunk]
    # The full final ranking regardless of the gate. Retrieval metrics score
    # this, because the gate is an answering policy rather than a property of
    # retrieval.
    candidates: list[RetrievedChunk]
    timings_ms: dict[str, float]
    mode: str = "dense"
    reranked: bool = False
    # Highest dense similarity seen, or None in lexical-only mode. This is what
    # the gate is evaluated against.
    best_dense_score: float | None = None
    per_component: dict[str, int] = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        db: Database,
        embedder: EmbeddingProvider,
        settings: Settings,
        reranker: RerankProvider | None = None,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._settings = settings
        self._reranker = reranker

    @property
    def embedder(self) -> EmbeddingProvider:
        """Exposed so a report can record which model produced its numbers."""
        return self._embedder

    @property
    def reranker(self) -> RerankProvider | None:
        return self._reranker

    async def retrieve(
        self,
        tenant_id: uuid.UUID,
        query: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
        source_ids: list[uuid.UUID] | None = None,
        mode: RetrievalMode | None = None,
        rerank: bool | None = None,
    ) -> RetrievalResult:
        settings = self._settings
        k = top_k or settings.retrieval_top_k
        floor = settings.min_similarity if min_similarity is None else min_similarity
        active_mode: RetrievalMode = mode or settings.retrieval_mode  # type: ignore[assignment]
        use_rerank = settings.rerank_enabled if rerank is None else rerank
        use_rerank = use_rerank and self._reranker is not None

        # Components retrieve deeper than k so fusion and reranking have
        # something to work with. Fusing two lists of length k can only ever
        # reorder those k; the gain comes from a chunk that one component ranked
        # 12th and the other ranked 2nd.
        depth = max(k, settings.retrieval_candidates)
        if use_rerank:
            depth = max(depth, settings.rerank_candidates)

        timings: dict[str, float] = {}
        per_component: dict[str, int] = {}

        dense_results: list[RetrievedChunk] = []
        lexical_results: list[RetrievedChunk] = []

        if active_mode == "dense":
            dense_results = await self._dense(tenant_id, query, depth, source_ids, timings)
        elif active_mode == "lexical":
            lexical_results = await self._lexical(tenant_id, query, depth, source_ids, timings)
        elif active_mode == "hybrid":
            # Independent: one embeds then queries, the other queries directly.
            # Running them concurrently makes hybrid cost roughly the slower of
            # the two rather than their sum.
            dense_results, lexical_results = await asyncio.gather(
                self._dense(tenant_id, query, depth, source_ids, timings),
                self._lexical(tenant_id, query, depth, source_ids, timings),
            )
        else:
            raise ValueError(f"Unknown retrieval mode {active_mode!r}")

        if dense_results:
            per_component["dense"] = len(dense_results)
        if lexical_results:
            per_component["lexical"] = len(lexical_results)

        # The gate is evaluated on dense candidates before any fusion, so it is
        # unaffected by the final ranking method.
        best_dense = max((c.score for c in dense_results), default=None)

        if active_mode == "hybrid":
            t0 = time.perf_counter()
            ranked = reciprocal_rank_fusion(
                {"dense": dense_results, "lexical": lexical_results},
                k=settings.rrf_k,
            )
            timings["fuse_ms"] = (time.perf_counter() - t0) * 1000
        else:
            ranked = list(dense_results or lexical_results)

        if use_rerank and ranked and self._reranker is not None:
            ranked = await self._rerank(query, ranked, settings.rerank_candidates, timings)

        ranked = ranked[:k]

        # Apply the gate.
        if active_mode == "lexical":
            kept = ranked
        elif best_dense is None or best_dense < floor:
            kept = []
            if dense_results:
                logger.info(
                    "similarity gate rejected query: best dense %.3f below floor %.3f",
                    best_dense or 0.0,
                    floor,
                )
        else:
            kept = ranked

        return RetrievalResult(
            chunks=kept,
            candidates=ranked,
            timings_ms=timings,
            mode=active_mode,
            reranked=bool(use_rerank and ranked),
            best_dense_score=best_dense,
            per_component=per_component,
        )

    # -- components --------------------------------------------------------

    async def _dense(
        self,
        tenant_id: uuid.UUID,
        query: str,
        depth: int,
        source_ids: list[uuid.UUID] | None,
        timings: dict[str, float],
    ) -> list[RetrievedChunk]:
        t0 = time.perf_counter()
        embedding = await asyncio.to_thread(self._embedder.embed_query, query)
        timings["embed_query_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        async with self._db.connection() as conn:
            results = await repo.search_dense(
                conn,
                tenant_id,
                embedding,
                model=self._embedder.model_id,
                limit=depth,
                source_ids=source_ids,
            )
        timings["dense_search_ms"] = (time.perf_counter() - t0) * 1000
        return results

    async def _lexical(
        self,
        tenant_id: uuid.UUID,
        query: str,
        depth: int,
        source_ids: list[uuid.UUID] | None,
        timings: dict[str, float],
    ) -> list[RetrievedChunk]:
        t0 = time.perf_counter()
        async with self._db.connection() as conn:
            results = await repo.search_lexical(
                conn, tenant_id, query, limit=depth, source_ids=source_ids
            )
        timings["lexical_search_ms"] = (time.perf_counter() - t0) * 1000
        return results

    async def _rerank(
        self,
        query: str,
        ranked: list[RetrievedChunk],
        candidates: int,
        timings: dict[str, float],
    ) -> list[RetrievedChunk]:
        assert self._reranker is not None
        window = ranked[:candidates]
        t0 = time.perf_counter()
        scores = await asyncio.to_thread(
            self._reranker.rerank, query, [c.text for c in window]
        )
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

        for chunk, score in zip(window, scores, strict=True):
            chunk.component_scores["rerank"] = float(score)
            # The first-stage score is kept under its own name; `score` becomes
            # the reranker's, because that is now what the ordering means.
            chunk.score = float(score)

        window.sort(key=lambda c: (-c.score, str(c.chunk_id)))
        # Anything beyond the rerank window keeps its first-stage order and
        # stays behind the reranked head.
        return window + ranked[candidates:]
