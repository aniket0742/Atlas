"""Domain types shared across ingestion, retrieval and answering."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class ParsedDocument:
    """A source document after parsing and normalisation, before chunking."""

    external_id: str
    content: str
    title: str | None = None
    uri: str | None = None
    mime_type: str | None = None
    byte_size: int | None = None
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """A retrievable unit of text with a resolvable span in its document."""

    ordinal: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    heading_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with provenance for citation."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    # The source's own identifier (a path, a URL). Eval labels are written
    # against this rather than a uuid so a dataset stays readable and stays
    # valid across re-ingestion.
    document_external_id: str
    document_title: str | None
    document_uri: str | None
    source_name: str
    ordinal: int
    text: str
    char_start: int
    char_end: int
    heading_path: list[str]
    score: float
    # Which retrieval component produced this and with what raw score. Phase 1
    # only ever has "dense"; hybrid fusion in Phase 2 fills in more, and the
    # eval harness reports per-component contribution.
    component_scores: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class Citation:
    """A claim in the answer tied to the chunk that supports it."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    # The source's own identifier (a path, a URL). Eval labels are written
    # against this rather than a uuid so a dataset stays readable and stays
    # valid across re-ingestion.
    document_external_id: str
    document_title: str | None
    document_uri: str | None
    char_start: int
    char_end: int
    quote: str
    # Whether the model's quote was found verbatim in the cited chunk. A false
    # here means the citation points at a real chunk but the supporting quote
    # was paraphrased or invented -- worth surfacing, not worth discarding.
    quote_verified: bool = True
    # PDF page number, when the document has one.
    page: int | None = None


@dataclass(slots=True)
class Answer:
    text: str
    citations: list[Citation]
    # True when the model reported it could not answer from the evidence, or
    # when retrieval returned nothing above the similarity floor.
    refused: bool
    # Why the answer was refused, when it was.
    refusal_reason: str | None
    retrieved: list[RetrievedChunk]
    usage: TokenUsage
    timings_ms: dict[str, float]
    # How the evidence was produced. Returned with every answer so a result can
    # be attributed to a configuration without consulting server settings, which
    # may have changed since.
    retrieval_mode: str = "dense"
    reranked: bool = False
    best_dense_score: float | None = None
    per_component: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class IngestResult:
    document_id: uuid.UUID
    version: int
    chunk_count: int
    # False when the content hash was unchanged and the pipeline was skipped.
    changed: bool
    indexed_at: datetime | None = None
