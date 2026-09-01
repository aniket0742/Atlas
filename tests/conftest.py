from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from atlas.config import Settings
from atlas.core.models import RetrievedChunk
from atlas.providers.fake import FakeEmbeddingProvider, FakeLLMProvider


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="fake",
        embedding_provider="fake",
        chunk_target_tokens=60,
        chunk_overlap_tokens=15,
        chunk_min_tokens=8,
        retrieval_top_k=5,
        min_similarity=0.0,
    )


@pytest.fixture
def embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def llm() -> FakeLLMProvider:
    return FakeLLMProvider()


class StubCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    async def fetchall(self) -> list[dict[str, Any]]:
        return self._rows


class StubConnection:
    """Returns canned rows. Enough for the code paths that only read metadata."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.executed: list[str] = []

    async def execute(self, query: str, params: Any = None) -> StubCursor:
        self.executed.append(" ".join(str(query).split())[:120])
        return StubCursor(self.rows)


class StubDatabase:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.conn = StubConnection(rows)

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[StubConnection]:
        yield self.conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[StubConnection]:
        yield self.conn


@pytest.fixture
def stub_db() -> StubDatabase:
    # get_document returns a row whose metadata carries no page offsets, which
    # is the normal case for markdown.
    return StubDatabase([{"metadata": {}}])


def make_chunk(
    text: str,
    *,
    score: float = 0.9,
    document_external_id: str = "doc.md",
    title: str = "Doc",
    chunk_id: uuid.UUID | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_external_id=document_external_id,
        document_title=title,
        document_uri=None,
        source_name="default",
        ordinal=0,
        text=text,
        char_start=0,
        char_end=len(text),
        heading_path=[],
        score=score,
        component_scores={"dense": score},
    )
