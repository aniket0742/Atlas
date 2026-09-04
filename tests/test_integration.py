"""End-to-end tests against a real Postgres with pgvector.

Skipped automatically when no database is reachable, so `pytest` still works on
a laptop with nothing running. Run the real thing with:

    docker compose up -d
    docker compose stop worker    # see below
    atlas migrate
    pytest -m integration

**Stop the worker container first.** `jobs.claim()` is deliberately global -- a
worker serves every tenant, which is what you want in production -- so a running
worker will happily claim jobs these tests just enqueued. The test then sees its
own worker find an empty queue, and the container worker crashes with a foreign
key violation when the test tears its tenant down mid-job.

That is not a bug in the queue; it is two consumers on one queue, working as
designed. It does mean integration tests need a database no live worker is
polling, the same way they need one no other developer is using.

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


# ---------------------------------------------------------------------------
# Ingestion queue (Phase 3)
# ---------------------------------------------------------------------------


@pytest.fixture
async def source(db, tenant):
    # Returns rather than yields: yielding inside the transaction would leave it
    # uncommitted for the test's duration, so nothing else could see the source
    # and a pool connection would be pinned the whole time.
    async with db.transaction() as conn:
        return await repo.ensure_source(conn, tenant, "docs")


async def enqueue(db, tenant, source, data: bytes, external_id: str, **kw):
    from atlas.db import jobs

    async with db.transaction() as conn:
        return await jobs.enqueue(
            conn, tenant, source, external_id=external_id, payload=data, **kw
        )


async def test_enqueue_returns_a_document_id_usable_before_any_work(db, tenant, source):
    """The 202 contract: the caller gets a durable id before the job runs."""
    from atlas.core import ids

    job_id, document_id = await enqueue(db, tenant, source, BILLING, "billing.md")
    assert job_id
    assert document_id == ids.document_id(tenant, source, "billing.md")


async def test_reuploading_while_pending_supersedes_rather_than_duplicates(db, tenant, source):
    from atlas.db import jobs

    first_job, first_doc = await enqueue(db, tenant, source, BILLING, "billing.md")
    updated = BILLING.replace(b"30 days", b"45 days")
    second_job, second_doc = await enqueue(db, tenant, source, updated, "billing.md")

    assert second_job == first_job, "a second pending job was created"
    assert second_doc == first_doc

    async with db.connection() as conn:
        stats = await jobs.queue_stats(conn, tenant)
    assert stats["pending"] == 1

    # The superseding payload is the one that gets processed.
    async with db.transaction() as conn:
        claimed = await jobs.claim(conn, "w1")
    assert b"45 days" in bytes(claimed["payload"])


async def test_claim_marks_running_and_counts_the_attempt(db, tenant, source):
    from atlas.db import jobs

    job_id, _ = await enqueue(db, tenant, source, BILLING, "billing.md")
    async with db.transaction() as conn:
        claimed = await jobs.claim(conn, "worker-a")
    assert claimed["id"] == job_id
    assert claimed["attempts"] == 1

    async with db.connection() as conn:
        row = await jobs.get_job(conn, tenant, job_id)
    assert row["status"] == "running"


async def test_two_workers_never_claim_the_same_job(db, tenant, source):
    """The property SKIP LOCKED exists for.

    Ten concurrent claimers against five jobs must produce five distinct
    claims and five empty results -- never the same job twice.
    """
    import asyncio

    from atlas.db import jobs

    for i in range(5):
        await enqueue(db, tenant, source, BILLING, f"doc-{i}.md")

    async def claim_one(worker: str):
        async with db.transaction() as conn:
            return await jobs.claim(conn, worker)

    results = await asyncio.gather(*[claim_one(f"w{i}") for i in range(10)])
    claimed = [r for r in results if r is not None]

    assert len(claimed) == 5
    assert len({c["id"] for c in claimed}) == 5, "a job was claimed twice"


async def test_claim_returns_none_on_an_empty_queue(db, tenant):
    from atlas.db import jobs

    async with db.transaction() as conn:
        assert await jobs.claim(conn, "w1") is None


async def test_a_job_in_backoff_is_not_claimable_yet(db, tenant, source):
    from atlas.db import jobs

    job_id, _ = await enqueue(db, tenant, source, BILLING, "billing.md")
    async with db.transaction() as conn:
        await jobs.claim(conn, "w1")
        status = await jobs.mark_failed(conn, job_id, "transient", base_delay_seconds=60)
    assert status == "pending"

    async with db.transaction() as conn:
        assert await jobs.claim(conn, "w2") is None, "a backing-off job was claimed early"


async def test_success_clears_the_payload_but_keeps_the_row(db, tenant, source):
    from atlas.db import jobs

    job_id, _ = await enqueue(db, tenant, source, BILLING, "billing.md")
    async with db.transaction() as conn:
        await jobs.claim(conn, "w1")
        await jobs.mark_succeeded(conn, job_id)
        cur = await conn.execute("SELECT status, payload FROM ingest_jobs WHERE id = %s", (job_id,))
        row = await cur.fetchone()

    assert row["status"] == "succeeded"
    assert row["payload"] is None, "payload retained after success"


async def test_exhausting_attempts_moves_a_job_to_the_dead_letter_state(db, tenant, source):
    from atlas.db import jobs

    job_id, _ = await enqueue(db, tenant, source, BILLING, "billing.md", max_attempts=2)
    statuses = []
    for _ in range(3):
        async with db.transaction() as conn:
            claimed = await jobs.claim(conn, "w1")
            if claimed is None:
                break
            statuses.append(
                await jobs.mark_failed(conn, job_id, "boom", base_delay_seconds=0.001)
            )

    assert statuses[-1] == "dead"
    async with db.connection() as conn:
        cur = await conn.execute("SELECT status, payload FROM ingest_jobs WHERE id = %s", (job_id,))
        row = await cur.fetchone()
    assert row["status"] == "dead"
    # A DLQ whose entries cannot be replayed is not a DLQ.
    assert row["payload"] is not None, "dead job lost its payload and cannot be requeued"


async def test_non_retryable_failure_dies_immediately(db, tenant, source):
    """The same bytes parse the same way; retrying only burns the budget."""
    from atlas.db import jobs

    job_id, _ = await enqueue(db, tenant, source, BILLING, "billing.md", max_attempts=5)
    async with db.transaction() as conn:
        await jobs.claim(conn, "w1")
        status = await jobs.mark_failed(conn, job_id, "unparseable", retryable=False)
    assert status == "dead"


async def test_a_dead_job_can_be_requeued_after_the_cause_is_fixed(db, tenant, source):
    from atlas.db import jobs

    job_id, _ = await enqueue(db, tenant, source, BILLING, "billing.md", max_attempts=1)
    async with db.transaction() as conn:
        await jobs.claim(conn, "w1")
        await jobs.mark_failed(conn, job_id, "boom", base_delay_seconds=0.001)
        assert await jobs.requeue_dead(conn, tenant, job_id) is True
        row = await jobs.get_job(conn, tenant, job_id)

    assert row["status"] == "pending"
    assert row["attempts"] == 0, "attempt budget was not reset"


async def test_queue_stats_report_depth_and_oldest_wait(db, tenant, source):
    from atlas.db import jobs

    for i in range(3):
        await enqueue(db, tenant, source, BILLING, f"doc-{i}.md")
    async with db.transaction() as conn:
        await jobs.claim(conn, "w1")
        stats = await jobs.queue_stats(conn, tenant)

    assert stats["pending"] == 2
    assert stats["running"] == 1
    assert stats["oldest_pending_seconds"] >= 0


@pytest.fixture
def worker(db, ingestor, settings):
    from atlas.ingest.worker import Worker

    return Worker(db, ingestor, settings, worker_id="test-worker")


async def test_worker_indexes_a_queued_document(db, tenant, source, worker):
    from atlas.db import jobs

    job_id, document_id = await enqueue(db, tenant, source, BILLING, "billing.md")

    assert await worker.run_once() is True
    assert await worker.run_once() is False, "queue should be empty"

    async with db.connection() as conn:
        job = await jobs.get_job(conn, tenant, job_id)
        document = await repo.get_document(conn, tenant, document_id)

    assert job["status"] == "succeeded"
    assert document is not None
    assert document["status"] == "indexed"
    assert document["chunk_count"] > 0


async def test_worker_sends_an_unparseable_document_straight_to_the_dead_letter(
    db, tenant, source, worker
):
    """No retries: the same bytes fail identically every time."""
    from atlas.db import jobs

    job_id, _ = await enqueue(db, tenant, source, b"   \n  ", "empty.md")
    await worker.run_once()

    async with db.connection() as conn:
        job = await jobs.get_job(conn, tenant, job_id)

    assert job["status"] == "dead"
    assert job["attempts"] == 1, "a non-retryable failure was retried"
    assert "extractable text" in (job["last_error"] or "")


async def test_a_crashed_worker_loses_no_work(db, tenant, source, worker):
    """Failure injection: a worker dies mid-job.

    Simulated by claiming a job and never finishing it, which is exactly the
    state a killed process leaves behind. The lease must expire, the reaper must
    return the job to the queue, and a second worker must complete it.

    Without the reaper the job sits in `running` forever and the document never
    becomes searchable, with nothing in the system saying why.
    """
    from atlas.db import jobs

    job_id, document_id = await enqueue(db, tenant, source, BILLING, "billing.md")

    # A worker claims it, then dies.
    async with db.transaction() as conn:
        claimed = await jobs.claim(conn, "doomed-worker")
    assert claimed["id"] == job_id

    # Nothing else can touch it while the lease holds.
    assert await worker.run_once() is False, "a leased job was claimable by another worker"

    # The lease expires and the reaper returns it.
    async with db.transaction() as conn:
        reclaimed = await jobs.reap_expired_leases(conn, lease_seconds=0)
    assert reclaimed == 1

    assert await worker.run_once() is True

    async with db.connection() as conn:
        job = await jobs.get_job(conn, tenant, job_id)
        document = await repo.get_document(conn, tenant, document_id)
        stats = await repo.corpus_stats(conn, tenant)

    assert job["status"] == "succeeded"
    assert job["attempts"] == 2, "the crashed attempt was not counted"
    assert document["status"] == "indexed"
    # Exactly once: the crash did not produce a second document or duplicate chunks.
    assert stats["documents"] == 1
    assert stats["chunks"] == document["chunk_count"]


async def test_a_job_that_always_kills_its_worker_eventually_dies(db, tenant, source):
    """A poison message must not cycle forever.

    Attempts are counted at claim time precisely so that a job which never
    reaches a completion handler still exhausts its budget.
    """
    from atlas.db import jobs

    job_id, _ = await enqueue(db, tenant, source, BILLING, "poison.md", max_attempts=3)

    for _ in range(5):
        async with db.transaction() as conn:
            claimed = await jobs.claim(conn, "doomed")
            if claimed is None:
                break
        # Worker dies without reporting; lease expires immediately.
        async with db.transaction() as conn:
            await jobs.reap_expired_leases(conn, lease_seconds=0)

    async with db.connection() as conn:
        job = await jobs.get_job(conn, tenant, job_id)

    assert job["attempts"] >= job["max_attempts"], "attempts were not counted at claim time"


async def test_processing_the_same_document_twice_is_idempotent(db, tenant, source, worker):
    """Duplicate delivery is survivable: at-least-once plus an idempotent pipeline."""
    await enqueue(db, tenant, source, BILLING, "billing.md")
    await worker.run_once()

    async with db.connection() as conn:
        first = await repo.corpus_stats(conn, tenant)

    # The same bytes queued again after the first job finished.
    await enqueue(db, tenant, source, BILLING, "billing.md")
    await worker.run_once()

    async with db.connection() as conn:
        second = await repo.corpus_stats(conn, tenant)

    assert second["documents"] == first["documents"] == 1
    assert second["chunks"] == first["chunks"]
    assert second["embeddings"] == first["embeddings"]


async def test_worker_drains_a_backlog(db, tenant, source, worker):
    from atlas.db import jobs

    for i in range(5):
        await enqueue(db, tenant, source, BILLING, f"doc-{i}.md")

    processed = 0
    while await worker.run_once():
        processed += 1

    assert processed == 5
    async with db.connection() as conn:
        stats = await jobs.queue_stats(conn, tenant)
        corpus = await repo.corpus_stats(conn, tenant)
    assert stats["pending"] == 0 and stats["running"] == 0 and stats["succeeded"] == 5
    assert corpus["documents"] == 5


async def test_run_forever_stops_promptly_when_asked(db, tenant, source, worker):
    """Graceful shutdown must not wait out a full poll interval."""
    import asyncio

    stop = asyncio.Event()
    await enqueue(db, tenant, source, BILLING, "billing.md")

    task = asyncio.create_task(worker.run_forever(stop))
    await asyncio.sleep(0.4)
    stop.set()
    await asyncio.wait_for(task, timeout=5.0)

    async with db.connection() as conn:
        corpus = await repo.corpus_stats(conn, tenant)
    assert corpus["documents"] == 1
