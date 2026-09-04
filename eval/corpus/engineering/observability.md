# Observability

## Traces

Every request carries a trace id, returned to the client in the
`X-Atlas-Request-Id` header. Quoting that header in a support ticket lets us
find the exact request without asking the customer to reproduce it.

Spans are emitted for query embedding, vector search, reranking and generation,
so a slow request can be attributed to a stage rather than guessed at.

## Metrics

| Metric | Type | Notes |
|---|---|---|
| `atlas_query_duration_seconds` | histogram | labelled by stage |
| `atlas_retrieval_score` | histogram | top-1 similarity per query |
| `atlas_refusals_total` | counter | labelled by refusal reason |
| `atlas_ingest_failures_total` | counter | labelled by error class |
| `atlas_queue_depth` | gauge | pending ingestion jobs |
| `atlas_tokens_total` | counter | prompt and output, per model |

## Why refusals are labelled by reason

A rise in refusals is ambiguous on its own. Refusals below the similarity floor
suggest a corpus gap. Refusals from the model reporting insufficient evidence
suggest a retrieval or prompting problem. Aggregating them into one number
throws away the distinction that makes the metric actionable.

## Logs

Logs are structured JSON. Document content is never logged, only identifiers and
counts. A log line that quotes chunk text will leak customer data into a system
with different access controls than the database.
