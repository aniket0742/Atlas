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
from typing import Any

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: uuid.UUID
    version: int
    chunk_count: int
    changed: bool = Field(
        description="False when the content hash was unchanged and indexing was skipped."
    )


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


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[CitationOut]
    evidence: list[EvidenceOut] | None = None
    usage: UsageOut
    timings_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    database: bool
    embedding_model: str
    llm_model: str


class StatsResponse(BaseModel):
    documents: int
    indexed_documents: int
    failed_documents: int
    chunks: int
    embeddings: int
    sources: int


class ErrorResponse(BaseModel):
    detail: str
