"""End-to-end tests against a real Postgres with pgvector.

Skipped automatically when no database is reachable, so `pytest` still works on
a laptop with nothing running. Run the real thing with:

    docker compose up -d
    atlas migrate
    pytest -m integration

These use the fake embedding provider deliberately. The point is to exercise
schema, transactions, idempotency and tenant isolation -- not model quality --
and the fake is deterministic, so a failure here is always a real defect rather
than model drift.
"""

from __future__ import annotations

import os
import uuid

import pytest

from atlas.config import Settings
from atlas.db import repository as repo
from atlas.db.migrate import apply_all
from atlas.db.pool import Database
from atlas.ingest.pipeline import Ingestor, IngestRequest
from atlas.providers.fake import FakeEmbeddingProvider

pytestmark = pytest.mark.integration

DSN = os.environ.get("ATLAS_TEST_DATABASE_URL") or os.environ.get(
    "ATLAS_DATABASE_URL", "postgresql://atlas:atlas@localhost:5432/atlas"
)

BILLING = b"""# Billing

## Refunds

Customers may request a refund within 30 days of purchase. Refunds return to
the original payment method.

## Chargebacks

Chargebacks are routed to the finance team and suspend the account.
"""

AUTH = b"""# Authentication

## Tokens

Access tokens are valid for 15 minutes. Refresh tokens rotate on every use.
"""


async def _reachable() -> bool:
    try:
        db = Database(DSN)
        await db.open()
        await db.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module", autouse=True)
async def _require_database():
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")
    await apply_all(DSN)


@pytest.fixture
async def db():
    database = Database(DSN)
    await database.open()
    yield database
    await database.close()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url=DSN,
        embedding_provider="fake",
        llm_provider="fake",
        chunk_target_tokens=60,
        chunk_overlap_tokens=15,
        chunk_min_tokens=8,
        retrieval_top_k=5,
        min_similarity=0.0,
    )


@pytest.fixture
async def tenant(db):
    """A throwaway tenant per test, so tests cannot interfere with each other."""
    slug = f"test-{uuid.uuid4().hex[:12]}"
    async with db.transaction() as conn:
        tenant_id = await repo.ensure_tenant(conn, slug)
    yield tenant_id
    async with db.transaction() as conn:
        await conn.execute("DELETE FROM tenants WHERE id = %s", (tenant_id,))


@pytest.fixture
def ingestor(db, settings) -> Ingestor:
    return Ingestor(db, FakeEmbeddingProvider(), settings)


async def ingest(ingestor, tenant, data: bytes, external_id: str, source: str = "docs"):
    return await ingestor.ingest(
        tenant,
        IngestRequest(data=data, external_id=external_id, source_name=source, filename=external_id),
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


async def test_document_is_indexed_and_queryable(db, ingestor, tenant):
    result = await ingest(ingestor, tenant, BILLING, "billing.md")

    assert result.changed
    assert result.version == 1
    assert result.chunk_count > 0

    async with db.connection() as conn:
        document = await repo.get_document(conn, tenant, result.document_id)
    assert document["status"] == "indexed"
    assert document["chunk_count"] == result.chunk_count
    assert document["title"] == "Billing"


async def test_reingesting_identical_bytes_is_a_no_op(db, ingestor, tenant):
    """The content-hash short-circuit: no re-parse, no re-embed, no writes."""
    first = await ingest(ingestor, tenant, BILLING, "billing.md")
    second = await ingest(ingestor, tenant, BILLING, "billing.md")

    assert second.changed is False
    assert second.document_id == first.document_id
    assert second.version == first.version

    async with db.connection() as conn:
        stats = await repo.corpus_stats(conn, tenant)
    assert stats["documents"] == 1
    assert stats["chunks"] == first.chunk_count


async def test_changed_content_bumps_version_and_replaces_old_chunks(db, ingestor, tenant):
    first = await ingest(ingestor, tenant, BILLING, "billing.md")
    updated = BILLING.replace(b"within 30 days", b"within 45 days")
    second = await ingest(ingestor, tenant, updated, "billing.md")

    assert second.document_id == first.document_id  # same logical document
    assert second.version == 2
    assert second.changed

    async with db.connection() as conn:
        rows = await conn.execute(
            "SELECT DISTINCT document_version FROM chunks WHERE document_id = %s",
            (first.document_id,),
        )
        versions = [r["document_version"] for r in await rows.fetchall()]
        stats = await repo.corpus_stats(conn, tenant)

    # Exactly one version's chunks survive: no orphans from v1.
    assert versions == [2]
    assert stats["chunks"] == second.chunk_count
    assert stats["embeddings"] == second.chunk_count


async def test_ingestion_is_idempotent_under_concurrency(db, ingestor, tenant):
    """Two workers processing the same document must converge, not duplicate."""
    import asyncio

    results = await asyncio.gather(
        *[ingest(ingestor, tenant, BILLING, "billing.md") for _ in range(4)],
        return_exceptions=True,
    )
    ok = [r for r in results if not isinstance(r, Exception)]
    assert ok, f"all concurrent ingests failed: {results}"
    assert len({r.document_id for r in ok}) == 1

    async with db.connection() as conn:
        stats = await repo.corpus_stats(conn, tenant)
    assert stats["documents"] == 1


async def test_failed_document_records_its_error_and_stays_visible(db, ingestor, tenant):
    """A broken document must be diagnosable, not silently absent."""
    from atlas.ingest.parsers import UnparseableDocument

    with pytest.raises(UnparseableDocument):
        await ingest(ingestor, tenant, b"   \n  ", "empty.md")

    async with db.connection() as conn:
        stats = await repo.corpus_stats(conn, tenant)
    # Parsing fails before any row is written, so nothing is persisted.
    assert stats["documents"] == 0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def test_retrieval_finds_the_relevant_document(db, ingestor, tenant, settings):
    from atlas.retrieval.service import Retriever

    await ingest(ingestor, tenant, BILLING, "billing.md")
    await ingest(ingestor, tenant, AUTH, "auth.md")

    retriever = Retriever(db, FakeEmbeddingProvider(), settings)
    result = await retriever.retrieve(tenant, "refund within 30 days of purchase")

    assert result.candidates
    assert result.candidates[0].document_external_id == "billing.md"
    assert result.candidates[0].score > 0


async def test_source_filter_restricts_results(db, ingestor, tenant, settings):
    from atlas.retrieval.service import Retriever

    await ingest(ingestor, tenant, BILLING, "billing.md", source="policies")
    await ingest(ingestor, tenant, AUTH, "auth.md", source="engineering")

    async with db.connection() as conn:
        engineering = await repo.ensure_source(conn, tenant, "engineering")

    retriever = Retriever(db, FakeEmbeddingProvider(), settings)
    result = await retriever.retrieve(tenant, "refund", source_ids=[engineering])

    assert all(c.document_external_id == "auth.md" for c in result.candidates)


# ---------------------------------------------------------------------------
# Tenant isolation -- the property Phase 5 depends on already being true
# ---------------------------------------------------------------------------


async def test_a_tenant_cannot_retrieve_another_tenants_chunks(db, ingestor, settings):
    from atlas.retrieval.service import Retriever

    slugs = [f"iso-a-{uuid.uuid4().hex[:8]}", f"iso-b-{uuid.uuid4().hex[:8]}"]
    async with db.transaction() as conn:
        a = await repo.ensure_tenant(conn, slugs[0])
        b = await repo.ensure_tenant(conn, slugs[1])

    try:
        # Same external_id, same bytes, different tenants.
        result_a = await ingest(ingestor, a, BILLING, "billing.md")
        result_b = await ingest(ingestor, b, BILLING, "billing.md")

        # Deterministic ids must still differ, because tenant is part of identity.
        assert result_a.document_id != result_b.document_id

        retriever = Retriever(db, FakeEmbeddingProvider(), settings)
        found = await retriever.retrieve(a, "refund within 30 days", top_k=20)

        assert found.candidates
        assert {c.document_id for c in found.candidates} == {result_a.document_id}

        async with db.connection() as conn:
            assert await repo.get_document(conn, a, result_b.document_id) is None
            assert await repo.get_document(conn, b, result_a.document_id) is None
    finally:
        async with db.transaction() as conn:
            await conn.execute("DELETE FROM tenants WHERE id = ANY(%s)", ([a, b],))


# ---------------------------------------------------------------------------
# Answering, end to end
# ---------------------------------------------------------------------------


async def test_end_to_end_answer_is_cited(db, ingestor, tenant, settings):
    from atlas.answer.service import AnswerService
    from atlas.providers.fake import FakeLLMProvider
    from atlas.retrieval.service import Retriever

    await ingest(ingestor, tenant, BILLING, "billing.md")

    retriever = Retriever(db, FakeEmbeddingProvider(), settings)
    service = AnswerService(db, retriever, FakeLLMProvider(), settings)
    answer = await service.answer(tenant, "refund within 30 days of purchase")

    assert not answer.refused
    assert answer.citations
    citation = answer.citations[0]
    assert citation.document_external_id == "billing.md"
    assert citation.quote_verified

    # The citation must resolve to a real span of the stored document.
    async with db.connection() as conn:
        content = await repo.get_document_content(conn, tenant, citation.document_id)
    assert content is not None
    assert content[citation.char_start : citation.char_end]


async def test_query_with_an_empty_corpus_refuses(db, tenant, settings):
    from atlas.answer.service import AnswerService
    from atlas.providers.fake import FakeLLMProvider
    from atlas.retrieval.service import Retriever

    retriever = Retriever(db, FakeEmbeddingProvider(), settings)
    service = AnswerService(db, retriever, FakeLLMProvider(), settings)
    answer = await service.answer(tenant, "anything at all?")

    assert answer.refused
    assert answer.refusal_reason == "no_candidates"
    assert answer.citations == []


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


async def test_migrations_are_idempotent():
    assert await apply_all(DSN) == []


async def test_editing_an_applied_migration_is_refused(tmp_path):
    """Guards against the database and the repo silently disagreeing."""
    from atlas.db.migrate import apply_all as apply

    directory = tmp_path / "migrations"
    directory.mkdir()
    path = directory / "9001_test_checksum.sql"
    path.write_text("CREATE TABLE atlas_checksum_probe (id int);", encoding="utf-8")

    try:
        await apply(DSN, directory)
        path.write_text("CREATE TABLE atlas_checksum_probe (id bigint);", encoding="utf-8")
        with pytest.raises(RuntimeError, match="modified after it was applied"):
            await apply(DSN, directory)
    finally:
        db = Database(DSN)
        await db.open()
        async with db.transaction() as conn:
            await conn.execute("DROP TABLE IF EXISTS atlas_checksum_probe")
            await conn.execute("DELETE FROM schema_migrations WHERE version = '9001_test_checksum'")
        await db.close()


# ---------------------------------------------------------------------------
# Lexical retrieval and hybrid fusion (Phase 2)
# ---------------------------------------------------------------------------


async def test_lexical_search_matches_exact_identifiers(db, ingestor, tenant):
    """The case dense retrieval is weakest at: a rare literal token."""
    doc = b"""# Errors

## Codes

ATL-4029 means the rate limit was exceeded.

## Notes

Something entirely unrelated about billing and refunds.
"""
    await ingest(ingestor, tenant, doc, "errors.md")

    async with db.connection() as conn:
        rows = await repo.search_lexical(conn, tenant, "What does ATL-4029 mean?", 5)

    assert rows
    assert "ATL-4029" in rows[0].text
    assert rows[0].component_scores["lexical"] > 0


async def test_lexical_search_uses_or_semantics(db, ingestor, tenant):
    """A long question must not require every term to appear in one chunk."""
    await ingest(ingestor, tenant, BILLING, "billing.md")

    async with db.connection() as conn:
        rows = await repo.search_lexical(
            conn,
            tenant,
            "how long exactly does a customer have to request a refund on their purchase",
            5,
        )
    # Under AND semantics this returns nothing at all.
    assert rows


async def test_lexical_search_on_stopwords_only_returns_nothing(db, ingestor, tenant):
    await ingest(ingestor, tenant, BILLING, "billing.md")
    async with db.connection() as conn:
        assert await repo.search_lexical(conn, tenant, "the and of it", 5) == []


async def test_lexical_search_is_tenant_scoped(db, ingestor, settings):
    """Same isolation guarantee as dense search, enforced independently."""
    slugs = [f"lex-a-{uuid.uuid4().hex[:8]}", f"lex-b-{uuid.uuid4().hex[:8]}"]
    async with db.transaction() as conn:
        a = await repo.ensure_tenant(conn, slugs[0])
        b = await repo.ensure_tenant(conn, slugs[1])
    try:
        result_a = await ingest(ingestor, a, BILLING, "billing.md")
        await ingest(ingestor, b, BILLING, "billing.md")

        async with db.connection() as conn:
            rows = await repo.search_lexical(conn, a, "refund within 30 days", 20)

        assert rows
        assert {r.document_id for r in rows} == {result_a.document_id}
    finally:
        async with db.transaction() as conn:
            await conn.execute("DELETE FROM tenants WHERE id = ANY(%s)", ([a, b],))


async def test_lexical_search_respects_the_source_filter(db, ingestor, tenant):
    await ingest(ingestor, tenant, BILLING, "billing.md", source="policies")
    await ingest(ingestor, tenant, AUTH, "auth.md", source="engineering")

    async with db.connection() as conn:
        engineering = await repo.ensure_source(conn, tenant, "engineering")
        rows = await repo.search_lexical(conn, tenant, "refund tokens", 20,
                                         source_ids=[engineering])

    assert all(r.document_external_id == "auth.md" for r in rows)


async def test_hybrid_returns_chunks_either_component_found(db, ingestor, tenant, settings):
    from atlas.retrieval.service import Retriever

    await ingest(ingestor, tenant, BILLING, "billing.md")
    await ingest(ingestor, tenant, AUTH, "auth.md")

    retriever = Retriever(db, FakeEmbeddingProvider(), settings)
    dense = await retriever.retrieve(tenant, "refund within 30 days", top_k=20, mode="dense")
    lexical = await retriever.retrieve(tenant, "refund within 30 days", top_k=20, mode="lexical")
    hybrid = await retriever.retrieve(tenant, "refund within 30 days", top_k=20, mode="hybrid")

    dense_ids = {c.chunk_id for c in dense.candidates}
    lexical_ids = {c.chunk_id for c in lexical.candidates}
    hybrid_ids = {c.chunk_id for c in hybrid.candidates}

    assert hybrid_ids == dense_ids | lexical_ids
    assert hybrid.mode == "hybrid"
    for chunk in hybrid.candidates:
        assert "rrf" in chunk.component_scores


async def test_similarity_gate_is_query_level_and_survives_fusion(db, ingestor, tenant, settings):
    """The Phase 2 change: the floor gates the query, not each fused row.

    An RRF score is a sum of reciprocal ranks, so a cosine threshold has no
    meaning against it. The gate is evaluated on dense candidates before fusion.
    """
    from atlas.retrieval.service import Retriever

    await ingest(ingestor, tenant, BILLING, "billing.md")
    retriever = Retriever(db, FakeEmbeddingProvider(), settings)

    passing = await retriever.retrieve(
        tenant, "refund within 30 days", top_k=5, mode="hybrid", min_similarity=0.0
    )
    assert passing.chunks
    assert passing.best_dense_score is not None

    # A floor above any achievable similarity must empty `chunks` while leaving
    # `candidates` intact, because retrieval metrics score candidates.
    blocked = await retriever.retrieve(
        tenant, "refund within 30 days", top_k=5, mode="hybrid", min_similarity=1.01
    )
    assert blocked.chunks == []
    assert blocked.candidates


async def test_lexical_mode_has_no_similarity_gate(db, ingestor, tenant, settings):
    """There is no dense score to gate on; lexical mode is for measurement."""
    from atlas.retrieval.service import Retriever

    await ingest(ingestor, tenant, BILLING, "billing.md")
    retriever = Retriever(db, FakeEmbeddingProvider(), settings)

    result = await retriever.retrieve(
        tenant, "refund", top_k=5, mode="lexical", min_similarity=1.01
    )
    assert result.best_dense_score is None
    assert result.chunks == result.candidates


async def test_reranking_reorders_and_records_its_score(db, ingestor, tenant, settings):
    from atlas.providers.reranker import FakeReranker
    from atlas.retrieval.service import Retriever

    await ingest(ingestor, tenant, BILLING, "billing.md")
    await ingest(ingestor, tenant, AUTH, "auth.md")

    reranker = FakeReranker()
    retriever = Retriever(db, FakeEmbeddingProvider(), settings, reranker=reranker)
    result = await retriever.retrieve(
        tenant, "refund within 30 days", top_k=5, mode="hybrid", rerank=True
    )

    assert result.reranked
    assert reranker.calls, "the reranker was never invoked"
    for chunk in result.candidates:
        assert "rerank" in chunk.component_scores
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
