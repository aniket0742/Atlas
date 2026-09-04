"""API request and response models.

Response shapes are deliberately explicit about provenance and cost: every
answer carries its citations, the evidence that was retrieved, per-stage
timings, and token usage. Those are the numbers Phase 6 needs for observability
and Phase 8 needs for evaluation, and returning them from the start means the
API contract does not have to change to expose them later.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestAcceptedResponse(BaseModel):
    """202 response: the document is queued, not yet indexed.

    `document_id` is usable immediately even though no work has happened. It is
    derived from (tenant, source, external_id) rather than assigned by the
    worker, so a caller can hold onto it before the job runs.
    """

    job_id: uuid.UUID
    document_id: uuid.UUID
    external_id: str
    status: str = Field(default="pending", description="Queue state at accept time.")


class JobSummary(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    external_id: str
    filename: str | None
    status: str
    attempts: int
    max_attempts: int
    last_error: str | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None


class QueueStats(BaseModel):
    pending: int
    running: int
    succeeded: int
    dead: int
    oldest_pending_seconds: float = Field(
        description="Age of the oldest waiting job. Depth alone cannot separate "
        "'busy' from 'stuck'; a queue holding steady with work completing is "
        "healthy and one holding steady with nothing progressing is not."
    )


class QueueResponse(BaseModel):
    stats: QueueStats
    jobs: list[JobSummary]


class DocumentSummary(BaseModel):
    id: uuid.UUID
    external_id: str
    title: str | None
    uri: str | None
    status: str
    version: int
    source_name: str
    indexed_at: datetime | None


class DocumentDetail(DocumentSummary):
    mime_type: str | None
    content_hash: str
    byte_size: int | None
    chunk_count: int
    metadata: dict[str, Any]
    error: str | None
    created_at: datetime
    updated_at: datetime


class SourceSummary(BaseModel):
    id: uuid.UUID
    name: str
    kind: str
    document_count: int
    created_at: datetime


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    mode: Literal["dense", "lexical", "hybrid"] | None = Field(
        default=None,
        description="Retrieval strategy. Defaults to the server setting. "
        "'lexical' is a diagnostic mode with no similarity gate.",
    )
    rerank: bool | None = Field(
        default=None,
        description="Override cross-encoder reranking. Costs roughly 700ms per "
        "query on CPU; see ADR-0020.",
    )
    min_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    source_ids: list[uuid.UUID] | None = None
    include_evidence: bool = Field(
        default=False,
        description="Return the full retrieved chunks, not just citations. Useful "
        "for debugging retrieval; verbose for normal use.",
    )


class CitationOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_external_id: str
    document_title: str | None
    document_uri: str | None
    page: int | None
    char_start: int
    char_end: int
    quote: str
    quote_verified: bool


class EvidenceOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    # Same stable identifier the citations carry, so the evidence panel and the
    # citation list name a document the same way.
    document_external_id: str
    document_title: str | None
    source_name: str
    heading_path: list[str]
    score: float
    component_scores: dict[str, float]
    text: str


class UsageOut(BaseModel):
    prompt_tokens: int
    output_tokens: int
    thinking_tokens: int
    total_tokens: int


class RetrievalInfo(BaseModel):
    """How this answer's evidence was produced.

    Returned on every query so a result can be attributed to a configuration
    without checking server settings, which may have changed since.
    """

    mode: str
    reranked: bool
    # Highest cosine similarity among dense candidates; None in lexical mode.
    # This is the value the refusal gate is evaluated against (ADR-0019).
    best_dense_score: float | None
    candidates_per_component: dict[str, int]


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    retrieval: RetrievalInfo
    citations: list[CitationOut]
    evidence: list[EvidenceOut] | None = None
    usage: UsageOut
    timings_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    database: bool
    embedding_model: str
    llm_model: str
    retrieval_mode: str
    rerank_model: str | None


class StatsResponse(BaseModel):
    documents: int
    indexed_documents: int
    failed_documents: int
    chunks: int
    embeddings: int
    sources: int


class ErrorResponse(BaseModel):
    detail: str
