# Environments and Configuration

## Environments

| Name | Purpose | Data |
|---|---|---|
| `local` | Developer machine, Docker Compose | Synthetic |
| `staging` | Pre-release verification | Anonymised copy |
| `production` | Customer traffic | Real |

There is no shared development environment. Anything that cannot be reproduced
locally is reproduced in staging, never by testing in production.

## Configuration keys

All configuration is environment variables prefixed `ATLAS_`. The ones that most
often need tuning:

- `ATLAS_WORKER_CONCURRENCY` — parallel ingestion workers per process, default 4
- `ATLAS_RETRIEVAL_TOP_K` — chunks retrieved per query, default 8
- `ATLAS_MIN_SIMILARITY` — similarity floor below which evidence is discarded
- `ATLAS_LLM_TIMEOUT_SECONDS` — per-call ceiling on generation, default 60
- `ATLAS_DB_POOL_MAX_SIZE` — database connections per process, default 10

## Secrets

Secrets are never environment variables in production. They are mounted from the
secret store at `/run/secrets` and read at startup. The `ATLAS_` variables above
carry configuration only, never credentials.

## Changing configuration

Configuration changes go through the same review as code. Editing environment
variables directly on a running production host is not permitted, because the
change is invisible to the next deployment and will be silently reverted.
