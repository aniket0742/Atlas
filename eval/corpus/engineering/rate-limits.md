# Rate Limits

## Default limits

| Plan | Requests per minute | Concurrent ingestions |
|---|---|---|
| Free | 20 | 1 |
| Standard | 300 | 8 |
| Enterprise | negotiated | negotiated |

Limits are applied per tenant, not per API token. Issuing more tokens does not
increase throughput.

## Response headers

Every response carries the current limit state:

- `X-Atlas-RateLimit-Limit` — the ceiling for this window
- `X-Atlas-RateLimit-Remaining` — requests left in this window
- `X-Atlas-RateLimit-Reset` — Unix timestamp when the window resets

A rejected request returns HTTP 429 with code ATL-4029 and a `Retry-After`
header in seconds. Clients should honour `Retry-After` rather than applying
their own backoff, because the server knows when the window actually resets.

## Burst behaviour

Limits use a token bucket with a burst allowance of twice the per-minute rate.
A client that has been idle can therefore issue a short burst above its nominal
limit before being throttled.

## Ingestion is limited separately

Document ingestion counts against the concurrent ingestion limit, not the
request limit. A tenant can be throttled for ingestion while its query traffic
is unaffected.
