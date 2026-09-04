# Runbook: Database Failover

## When to fail over

Only when the primary is unrecoverable or unreachable for more than five
minutes. Failover is disruptive; a slow primary is usually better than an
unnecessary failover.

## Procedure

1. Confirm the primary is genuinely unreachable, not merely slow. Check from
   two networks.
2. Verify replica lag is under 5 seconds. Failing over to a lagging replica
   loses the difference.
3. Promote the replica.
4. Update the connection string secret and restart API processes.
5. Confirm writes succeed and `atlas_ingest_failures_total` stops rising.

## Data loss window

Replication is asynchronous. Failover can lose up to the replication lag at the
moment of promotion, typically under 2 seconds. Documents ingested inside that
window may need re-ingesting; their content hashes make re-ingestion safe and
idempotent.

## After failover

The old primary must not be restarted into the cluster without being rebuilt
from the new primary. Two nodes both believing they are primary is worse than
the outage that caused the failover.

## Rebuilding

Rebuild the failed node as a replica, verify it catches up, and only then
consider failing back. There is no requirement to fail back at all.
