"""Ingestion job queue.

A Postgres table driven by `SELECT ... FOR UPDATE SKIP LOCKED`, chosen over a
broker in ADR-0002 because enqueue can then share a transaction with the document
write.

The three operations that have to be right under concurrency:

* **claim** — two workers must never take the same job. `SKIP LOCKED` makes each
  concurrent claimer step over rows another transaction has locked rather than
  blocking on them, so N workers claim N distinct jobs in one round trip each.
* **fail** — a retryable failure returns the job to `pending` with `run_after`
  in the future. Backoff is expressed as a timestamp rather than a separate
  state, so the claim query needs no extra branch: a job in backoff simply is
  not due yet.
* **reap** — a worker that dies leaves its job `running` forever. Jobs carry a
  lease; the reaper returns rows whose lease expired to `pending`. This is the
  only mechanism that makes worker crashes recoverable, and it is why `attempts`
  is incremented at claim time rather than at completion: a job that repeatedly
  kills its worker must eventually reach the dead-letter state instead of
  cycling forever.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

from psycopg import AsyncConnection

from atlas.core import ids

# Terminal states. `succeeded` and `dead` rows are kept for inspection; their
# payloads are cleared.
TERMINAL = ("succeeded", "dead")


async def enqueue(
    conn: AsyncConnection,
    tenant_id: uuid.UUID,
    source_id: uuid.UUID,
    *,
    external_id: str,
    payload: bytes,
    filename: str | None = None,
    mime_type: str | None = None,
    uri: str | None = None,
    max_attempts: int = 4,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Queue a document for ingestion. Returns (job_id, document_id).

    The document id is derived here rather than by the worker, so the caller can
    be handed something durable to poll before any work has happened.

    Re-uploading while an earlier upload is still pending replaces that job's
    payload instead of queueing a second one (see the partial unique index in
    migration 0003). A job already `running` cannot be superseded, because a
    worker holds it; that case falls through to the pipeline's content-hash
    short-circuit, which makes the duplicate nearly free.

    Must be called inside a transaction that also writes whatever the caller
    considers the document's arrival — that atomicity is the whole argument for
    a database-backed queue.
    """
    document_id = ids.document_id(tenant_id, source_id, external_id)

    cur = await conn.execute(
        """
        INSERT INTO ingest_jobs (
            id, tenant_id, source_id, document_id, external_id,
            payload, filename, mime_type, uri, byte_size, max_attempts
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (tenant_id, source_id, external_id) WHERE status = 'pending'
        DO UPDATE SET
            payload    = EXCLUDED.payload,
            filename   = EXCLUDED.filename,
            mime_type  = EXCLUDED.mime_type,
            uri        = EXCLUDED.uri,
            byte_size  = EXCLUDED.byte_size,
            attempts   = 0,
            run_after  = now(),
            last_error = NULL,
            updated_at = now()
        RETURNING id
        """,
        (
            uuid.uuid4(),
            tenant_id,
            source_id,
            document_id,
            external_id,
            payload,
            filename,
            mime_type,
            uri,
            len(payload),
            max_attempts,
        ),
    )
    row = await cur.fetchone()
    assert row is not None
    return row["id"], document_id


async def claim(
    conn: AsyncConnection, worker_id: str, *, lease_seconds: int = 300
) -> dict[str, Any] | None:
    """Take one due job, or return None.

    The inner SELECT locks a single candidate row with SKIP LOCKED; the outer
    UPDATE flips it to `running` in the same statement, so there is no window in
    which a claimed job looks available. `attempts` increments here, not on
    completion, so a job that crashes its worker still counts towards
    max_attempts and cannot cycle forever.
    """
    cur = await conn.execute(
        """
        UPDATE ingest_jobs
        SET status     = 'running',
            locked_by  = %s,
            locked_at  = now(),
            attempts   = attempts + 1,
            updated_at = now()
        WHERE id = (
            SELECT id FROM ingest_jobs
            WHERE status = 'pending' AND run_after <= now()
            ORDER BY run_after, created_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING id, tenant_id, source_id, document_id, external_id,
                  payload, filename, mime_type, uri, attempts, max_attempts
        """,
        (worker_id,),
    )
    return await cur.fetchone()  # type: ignore[return-value]


async def mark_succeeded(conn: AsyncConnection, job_id: uuid.UUID) -> None:
    """Finish a job and drop its payload.

    The row is kept: queue history is how "where did ingestion break" stays
    answerable. The bytes are not, because retaining every uploaded document
    twice is a storage bug waiting to be discovered in production.
    """
    await conn.execute(
        """
        UPDATE ingest_jobs
        SET status = 'succeeded', payload = NULL, last_error = NULL,
            locked_by = NULL, locked_at = NULL,
            finished_at = now(), updated_at = now()
        WHERE id = %s
        """,
        (job_id,),
    )


async def mark_failed(
    conn: AsyncConnection,
    job_id: uuid.UUID,
    error: str,
    *,
    retryable: bool = True,
    base_delay_seconds: float = 2.0,
) -> str:
    """Record a failure. Returns the resulting status: 'pending' or 'dead'.

    Exponential backoff with full jitter. Jitter matters because a provider
    outage fails every in-flight job at once; without it they all retry in
    lockstep and reproduce the outage's load pattern against a service that is
    still recovering.

    `retryable=False` is for failures that cannot succeed on a second attempt --
    an unparseable document is the same bytes next time -- so retrying only burns
    the attempt budget and delays the operator seeing a real error.

    A dead job KEEPS its payload, unlike a successful one. A dead-letter queue
    whose entries cannot be replayed is not a dead-letter queue: the point is to
    fix the cause and requeue. Storage is bounded by how many jobs are dead, and
    if that is large enough to matter it is itself the signal.
    """
    cur = await conn.execute(
        "SELECT attempts, max_attempts FROM ingest_jobs WHERE id = %s FOR UPDATE",
        (job_id,),
    )
    row = await cur.fetchone()
    if row is None:
        return "dead"

    exhausted = row["attempts"] >= row["max_attempts"]
    if exhausted or not retryable:
        await conn.execute(
            """
            UPDATE ingest_jobs
            SET status = 'dead', last_error = %s,
                locked_by = NULL, locked_at = NULL,
                finished_at = now(), updated_at = now()
            WHERE id = %s
            """,
            (error[:4000], job_id),
        )
        return "dead"

    delay = min(base_delay_seconds * (2 ** (row["attempts"] - 1)), 600.0)
    delay *= 0.5 + random.random() / 2  # full-ish jitter, never below half
    await conn.execute(
        """
        UPDATE ingest_jobs
        SET status = 'pending', last_error = %s,
            locked_by = NULL, locked_at = NULL,
            run_after = now() + make_interval(secs => %s),
            updated_at = now()
        WHERE id = %s
        """,
        (error[:4000], delay, job_id),
    )
    return "pending"


async def reap_expired_leases(conn: AsyncConnection, *, lease_seconds: int = 300) -> int:
    """Return jobs whose worker died back to the queue. Returns how many.

    Without this a crashed worker's job is stuck in `running` forever and its
    document never becomes queryable, with nothing in the system indicating why.
    """
    cur = await conn.execute(
        """
        UPDATE ingest_jobs
        SET status = 'pending',
            locked_by = NULL,
            locked_at = NULL,
            last_error = coalesce(last_error, 'worker lease expired'),
            updated_at = now()
        WHERE status = 'running'
          AND locked_at < now() - make_interval(secs => %s)
        RETURNING id
        """,
        (lease_seconds,),
    )
    return len(await cur.fetchall())


async def get_job(
    conn: AsyncConnection, tenant_id: uuid.UUID, job_id: uuid.UUID
) -> dict[str, Any] | None:
    cur = await conn.execute(
        """
        SELECT id, document_id, external_id, status, attempts, max_attempts,
               last_error, run_after, created_at, updated_at, finished_at,
               byte_size, filename
        FROM ingest_jobs
        WHERE id = %s AND tenant_id = %s
        """,
        (job_id, tenant_id),
    )
    return await cur.fetchone()  # type: ignore[return-value]


async def list_jobs(
    conn: AsyncConnection,
    tenant_id: uuid.UUID,
    *,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    filters = ["tenant_id = %(tenant)s"]
    params: dict[str, Any] = {"tenant": tenant_id, "limit": limit}
    if status:
        filters.append("status = %(status)s")
        params["status"] = status

    cur = await conn.execute(
        f"""
        SELECT id, document_id, external_id, status, attempts, max_attempts,
               last_error, created_at, updated_at, finished_at, filename
        FROM ingest_jobs
        WHERE {" AND ".join(filters)}
        ORDER BY created_at DESC
        LIMIT %(limit)s
        """,
        params,
    )
    return await cur.fetchall()  # type: ignore[return-value]


async def queue_stats(conn: AsyncConnection, tenant_id: uuid.UUID) -> dict[str, Any]:
    """Queue depth by state, plus the age of the oldest waiting job.

    Depth alone cannot distinguish "busy" from "stuck": a queue holding steady at
    50 with jobs completing is healthy, and one holding steady at 50 because
    nothing is progressing is not. Oldest-pending age separates them.
    """
    cur = await conn.execute(
        """
        SELECT
            count(*) FILTER (WHERE status = 'pending')   AS pending,
            count(*) FILTER (WHERE status = 'running')   AS running,
            count(*) FILTER (WHERE status = 'succeeded') AS succeeded,
            count(*) FILTER (WHERE status = 'dead')      AS dead,
            -- FILTER attaches to the aggregate, not to extract(): the
            -- aggregate is what is being filtered.
            coalesce(
                extract(
                    epoch FROM now() - min(created_at) FILTER (WHERE status = 'pending')
                ),
                0
            ) AS oldest_pending_seconds
        FROM ingest_jobs
        WHERE tenant_id = %s
        """,
        (tenant_id,),
    )
    row = await cur.fetchone()
    assert row is not None
    return {k: (int(v) if k != "oldest_pending_seconds" else float(v)) for k, v in row.items()}


async def requeue_dead(conn: AsyncConnection, tenant_id: uuid.UUID, job_id: uuid.UUID) -> bool:
    """Move a dead-letter job back to pending, resetting its attempt budget.

    Deliberately manual. A dead job failed `max_attempts` times, so automatic
    requeueing would just reproduce the failure; an operator requeues after
    fixing the cause. Dead jobs retain their payload precisely so this works.
    """
    cur = await conn.execute(
        """
        UPDATE ingest_jobs
        SET status = 'pending', attempts = 0, run_after = now(),
            last_error = NULL, finished_at = NULL, updated_at = now()
        WHERE id = %s AND tenant_id = %s AND status = 'dead' AND payload IS NOT NULL
        RETURNING id
        """,
        (job_id, tenant_id),
    )
    return await cur.fetchone() is not None
