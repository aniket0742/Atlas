"""Ingestion worker.

Claims jobs from the Postgres queue and runs them through the same pipeline the
synchronous path used in Phase 1. That reuse is the payoff for building
deterministic ids and the content-hash short-circuit early: moving ingestion
behind a queue changed the call site, not the pipeline.

## Transaction shape

Three separate transactions per job, deliberately:

1. claim — commits immediately, so the row is marked `running` and released
2. process — parse, chunk, embed, index. No transaction held.
3. finish — mark succeeded or failed

Holding the claim transaction open across step 2 would keep a row lock for the
tens of seconds a large document takes to embed, block the reaper, and consume a
connection from a small pool. The lease exists precisely so the claim can be
committed and the work done unlocked.

## Which failures are retryable

A document that failed to parse will fail identically on the next attempt: the
bytes are the same. Retrying it burns the attempt budget and delays an operator
seeing a real error, so parse failures go straight to the dead-letter state.
Everything else -- a database blip, a model that ran out of memory -- is assumed
retryable.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time

from atlas.config import Settings
from atlas.db import jobs
from atlas.db.pool import Database
from atlas.ingest.parsers import UnparseableDocument, UnsupportedDocument
from atlas.ingest.pipeline import Ingestor, IngestRequest

logger = logging.getLogger(__name__)


def default_worker_id() -> str:
    """Host and pid, so a stuck lease names something an operator can find."""
    return f"{socket.gethostname()}:{os.getpid()}"


class Worker:
    def __init__(
        self,
        db: Database,
        ingestor: Ingestor,
        settings: Settings,
        *,
        worker_id: str | None = None,
    ) -> None:
        self._db = db
        self._ingestor = ingestor
        self._settings = settings
        self._worker_id = worker_id or default_worker_id()
        self._last_reap = 0.0

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def run_once(self) -> bool:
        """Claim and process one job. Returns False when the queue is empty."""
        async with self._db.transaction() as conn:
            job = await jobs.claim(
                conn, self._worker_id, lease_seconds=self._settings.worker_lease_seconds
            )
        if job is None:
            return False

        await self._process(job)
        return True

    async def _process(self, job: dict) -> None:
        job_id = job["id"]
        started = time.perf_counter()
        try:
            result = await self._ingestor.ingest(
                job["tenant_id"],
                IngestRequest(
                    data=bytes(job["payload"]),
                    external_id=job["external_id"],
                    # The source is passed explicitly below; the name is unused.
                    source_name="",
                    filename=job["filename"],
                    mime_type=job["mime_type"],
                    uri=job["uri"],
                ),
                source_id=job["source_id"],
            )
        except (UnsupportedDocument, UnparseableDocument) as exc:
            # Deterministic: the same bytes will fail the same way.
            async with self._db.transaction() as conn:
                await jobs.mark_failed(conn, job_id, str(exc), retryable=False)
            logger.warning("job %s dead (not retryable): %s", job_id, exc)
            return
        except Exception as exc:
            async with self._db.transaction() as conn:
                status = await jobs.mark_failed(conn, job_id, f"{type(exc).__name__}: {exc}")
            logger.exception(
                "job %s failed on attempt %s/%s -> %s",
                job_id,
                job["attempts"],
                job["max_attempts"],
                status,
            )
            return

        async with self._db.transaction() as conn:
            await jobs.mark_succeeded(conn, job_id)

        logger.info(
            "job %s ok document=%s v%s chunks=%s changed=%s in %.0fms",
            job_id,
            result.document_id,
            result.version,
            result.chunk_count,
            result.changed,
            (time.perf_counter() - started) * 1000,
        )

    async def _maybe_reap(self) -> None:
        """Return jobs from dead workers to the queue, at most once per interval."""
        now = time.monotonic()
        if now - self._last_reap < self._settings.worker_reap_interval_seconds:
            return
        self._last_reap = now
        async with self._db.transaction() as conn:
            reclaimed = await jobs.reap_expired_leases(
                conn, lease_seconds=self._settings.worker_lease_seconds
            )
        if reclaimed:
            logger.warning("reclaimed %s job(s) from expired leases", reclaimed)

    async def run_forever(self, stop: asyncio.Event | None = None) -> None:
        """Poll for work until stopped.

        Polling rather than LISTEN/NOTIFY. NOTIFY would cut idle-to-start latency
        from up to one poll interval to near zero, but it needs a dedicated
        connection per worker held open indefinitely and a fallback poll anyway
        (a NOTIFY delivered while a worker is busy is not queued). At current
        volumes a one-second poll is not the bottleneck; ingestion of a single
        document takes far longer than the wait to start it.
        """
        stop = stop or asyncio.Event()
        logger.info(
            "worker %s started (concurrency=%s poll=%.1fs lease=%ss)",
            self._worker_id,
            self._settings.worker_concurrency,
            self._settings.worker_poll_interval_seconds,
            self._settings.worker_lease_seconds,
        )

        while not stop.is_set():
            await self._maybe_reap()

            # Claim up to `concurrency` jobs and run them together. Embedding
            # releases the GIL inside ONNX Runtime, so concurrent jobs overlap;
            # beyond the core count they contend rather than speed up, which is
            # why this is configurable and defaults low.
            running = [
                asyncio.create_task(self.run_once())
                for _ in range(self._settings.worker_concurrency)
            ]
            results = await asyncio.gather(*running, return_exceptions=True)

            for outcome in results:
                if isinstance(outcome, BaseException):
                    logger.exception("worker loop error", exc_info=outcome)

            if any(r is True for r in results):
                continue  # queue still has work; go straight round again

            # Idle. Wait for the poll interval, but wake immediately on stop so
            # shutdown is not delayed by a full interval.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    stop.wait(), timeout=self._settings.worker_poll_interval_seconds
                )

        logger.info("worker %s stopped", self._worker_id)

