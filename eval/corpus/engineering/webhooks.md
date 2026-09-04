# Webhooks

## Events

| Event | Fired when |
|---|---|
| `document.indexed` | A document version becomes queryable |
| `document.failed` | Ingestion failed; payload carries the error |
| `source.sync.completed` | A scheduled source crawl finished |
| `tenant.quota.warning` | A tenant crosses 80% of a quota |

## Delivery guarantees

Delivery is at-least-once. Handlers must be idempotent. Every payload carries a
`delivery_id` that is stable across retries of the same event, so a handler can
deduplicate on it.

## Retries

Failed deliveries retry on an exponential schedule at 1, 5, 25 and 125 minutes.
After the fourth failure the endpoint is marked unhealthy and deliveries are
paused until it is re-enabled from the console.

A delivery is considered failed if the endpoint does not return a 2xx within 10
seconds. Slow endpoints are the most common cause of unhealthy webhooks.

## Signature verification

Payloads are signed with HMAC-SHA256 over the raw request body using the
endpoint's signing secret. The signature is in `X-Atlas-Signature`. Verify
against the raw bytes, not a re-serialised JSON object, or the signature will
never match.
