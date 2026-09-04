# Release Notes — 2026 Q2

## v2.5.0 — 12 April 2026

**Added.** Hybrid retrieval combining vector similarity with lexical matching.
Disabled by default behind `feature.retrieval.hybrid_search`.

**Added.** `X-Atlas-Request-Id` on every response, including error responses.

**Changed.** Ingestion moved to a queue. `POST /v1/documents` now returns 202
with a job id rather than blocking until the document is queryable.

## v2.5.1 — 30 April 2026

**Fixed.** Queue backlog could grow unbounded when a poison document was retried
indefinitely. Attempts are now capped and exhausted jobs move to a dead letter
queue.

## v2.6.0 — 26 May 2026

**Added.** Cross-encoder reranking, off by default.

**Changed.** Refusal responses now carry a machine-readable reason, so clients
can distinguish "nothing retrieved" from "the model judged the evidence
insufficient".

**Removed.** The `/v1/search` endpoint deprecated in v2.4.0.

## v2.6.2 — 18 June 2026

**Fixed.** Webhook signatures were computed over a re-serialised payload rather
than the raw body, so signatures failed to verify for payloads containing
non-ASCII characters.
