# Integrations

## Available connectors

| Source | Sync | Incremental |
|---|---|---|
| File upload | Manual | n/a |
| Public website crawl | Scheduled, daily | Yes, by content hash |
| GitHub repository | Scheduled, hourly | Yes, by commit |
| S3-compatible bucket | Scheduled, hourly | Yes, by ETag |

## GitHub

The connector indexes Markdown, plain text and source files from a chosen branch.
It does not index issues or pull requests; those change too frequently to be
worth re-indexing and are better answered live.

Repository access uses a GitHub App installation, not a personal access token,
so access survives the departure of whoever configured it.

## Website crawl

Respects `robots.txt` and a configurable path allowlist. Pages that return
non-200 are retried on the next scheduled run rather than immediately, because a
crawl that hammers a failing site is indistinguishable from an attack.

## What incremental means

Only changed documents are re-embedded. Change is detected by content hash, so a
re-crawl that finds identical content does no work and costs nothing. This is
what makes an hourly sync affordable.

## Requesting a connector

Connector requests go through the product team. The threshold is more than one
customer asking, because a connector is a permanent maintenance commitment.
