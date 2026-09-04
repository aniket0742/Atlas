# Caching

## What is cached

| Layer | Key | TTL |
|---|---|---|
| Query embeddings | hash of normalised query text | 24 hours |
| Retrieval results | tenant + query hash + retrieval config | 10 minutes |
| Document metadata | document id | until write |

## Cache keys must include the tenant

Every cache key is prefixed with the tenant id. A cache key that omits it will
serve one customer's results to another, and no amount of correct database
filtering will prevent that. This is the single most dangerous class of bug in
the system.

## Retrieval config in the key

The retrieval cache key includes the retrieval configuration, not just the
query. Otherwise changing the similarity floor or the number of retrieved chunks
serves stale results computed under the old settings, which makes evaluation
runs silently wrong.

## Invalidation

Ingesting a document version invalidates the retrieval cache for its tenant.
This is deliberately coarse: precise invalidation would require knowing which
queries a chunk could affect, which is the retrieval problem itself.

## What is not cached

Generated answers are not cached. The same question asked twice may retrieve
different evidence if the corpus changed between them, and serving a stale
answer with fresh-looking citations would misrepresent the system.
