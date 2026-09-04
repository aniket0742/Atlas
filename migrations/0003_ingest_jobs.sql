-- Asynchronous ingestion queue (Phase 3).
--
-- A Postgres job table rather than Kafka or Redis Streams, for the reason in
-- ADR-0002: enqueue happens in the SAME TRANSACTION as the document write. An
-- external broker cannot do that without an outbox table, at which point the
-- outbox is a Postgres queue and the broker is an extra hop.
--
-- Where the bytes live
-- --------------------
-- The uploaded payload is stored in the job row. Postgres TOASTs and compresses
-- anything past ~2KB, storing it out of line, so a 20MB upload does not sit in
-- the main heap. The alternative -- writing to object storage and referencing a
-- key -- means the payload and the job can disagree after a crash, and it adds
-- infrastructure this system does not otherwise need.
--
-- The payload is cleared when a job SUCCEEDS, so history survives for inspection
-- without keeping every uploaded document stored twice. A job that reaches the
-- dead-letter state keeps its payload: a DLQ whose entries cannot be replayed is
-- not a DLQ. See ADR-0022.

CREATE TABLE ingest_jobs (
    id            uuid PRIMARY KEY,
    tenant_id     uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    source_id     uuid NOT NULL REFERENCES sources(id) ON DELETE CASCADE,

    -- The document this job will produce. Derived deterministically at enqueue
    -- time (uuid5 over tenant/source/external_id), so a caller can poll the
    -- document before the job has run.
    document_id   uuid NOT NULL,
    external_id   text NOT NULL,

    -- Raw uploaded bytes. Cleared on success; retained for dead jobs so they
    -- can be requeued after the cause is fixed.
    payload       bytea,
    filename      text,
    mime_type     text,
    uri           text,
    byte_size     bigint,

    status        text NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'running', 'succeeded', 'dead')),

    attempts      integer NOT NULL DEFAULT 0,
    max_attempts  integer NOT NULL DEFAULT 4,

    -- Backoff: a retryable failure returns the job to 'pending' with run_after
    -- in the future. The claim query filters on it, so a backing-off job simply
    -- is not visible yet rather than needing a separate state.
    run_after     timestamptz NOT NULL DEFAULT now(),

    -- Lease. A worker that dies leaves status='running' forever; the reaper
    -- reclaims rows whose lease has expired.
    locked_by     text,
    locked_at     timestamptz,

    last_error    text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    finished_at   timestamptz
);

-- The claim query's index: it looks for pending work that is due, oldest first.
-- Partial, because succeeded and dead rows accumulate and must not slow the
-- hot path.
CREATE INDEX ingest_jobs_claim_idx
    ON ingest_jobs (run_after, created_at)
    WHERE status = 'pending';

-- The reaper's index: running jobs whose lease may have expired.
CREATE INDEX ingest_jobs_lease_idx
    ON ingest_jobs (locked_at)
    WHERE status = 'running';

CREATE INDEX ingest_jobs_tenant_idx ON ingest_jobs (tenant_id, status);
CREATE INDEX ingest_jobs_document_idx ON ingest_jobs (document_id);

-- Duplicate suppression.
--
-- Re-uploading a document while an earlier upload is still queued should replace
-- the queued payload, not add a second job. Enforced as a partial unique index
-- over pending rows only: a job already running cannot be superseded (a worker
-- holds it), and historical rows must be free to repeat.
--
-- This does not make ingestion idempotent on its own -- the deterministic ids
-- and the content-hash short-circuit in the pipeline do that, and they still
-- apply if a duplicate slips through. This just avoids the wasted work.
CREATE UNIQUE INDEX ingest_jobs_pending_uniq
    ON ingest_jobs (tenant_id, source_id, external_id)
    WHERE status = 'pending';
