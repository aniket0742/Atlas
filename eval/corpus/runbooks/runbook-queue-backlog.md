# Runbook: Ingestion Queue Backlog

## Symptom

`atlas_queue_depth` rising steadily, documents remaining in `pending` for longer
than usual, or customers reporting that uploads are not searchable.

## First checks

1. Are workers alive? A crashed worker pool shows depth rising with zero
   throughput.
2. Is throughput non-zero but insufficient? That is a capacity problem, not a
   failure.
3. Is one document repeatedly failing and being retried? A poison message can
   consume the whole pool.

## Poison messages

A document that fails deterministically will be retried until it exhausts its
attempts. Check `atlas_ingest_failures_total` grouped by error class. If a single
document id dominates, move it to the dead letter queue by hand rather than
waiting for it to age out.

## Increasing capacity

Raise `ATLAS_WORKER_CONCURRENCY` before adding worker processes. Embedding is
CPU bound, so concurrency beyond the available cores adds contention rather than
throughput.

## Shedding load

If depth continues to rise, ingestion returns ATL-5031 and sheds new work.
Query traffic is unaffected: ingestion and query capacity are separate, and
degrading search to accept uploads would be the wrong trade.

## Do not

Do not clear the queue to make the graph look better. The jobs are the only
record that those documents need indexing.
