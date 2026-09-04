# Database Migrations

## Forward only

Migrations are forward only. There is no down migration. Reverting a schema
change means writing a new migration that compensates for it.

This is why a release containing a schema change cannot simply be rolled back:
rolling back the code leaves the new schema in place, and the old code may not
tolerate it.

## Expand and contract

A change that would break running code is split across releases:

1. Expand — add the new column or table, write to both, read from the old
2. Backfill — populate the new shape for existing rows
3. Contract — read from the new shape, stop writing the old, drop it

Each step is a separate deployment. Skipping the expand step is the most common
cause of deployment-time errors.

## Locks

Adding a column with a non-null default rewrites the table on older Postgres
versions and holds an exclusive lock for the duration. Add the column nullable,
backfill in batches, then add the constraint.

Creating an index on a large table must use `CREATE INDEX CONCURRENTLY`, which
cannot run inside a transaction.

## Checksums

An applied migration's checksum is recorded. Editing a migration after it has
been applied is refused at startup, because it means the database and the
repository disagree about the schema.
