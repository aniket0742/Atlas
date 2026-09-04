# Search Architecture

## Retrieval pipeline

A query is embedded, matched against chunk embeddings by cosine distance, and
the surviving chunks become evidence for the model. Chunks below the similarity
floor are discarded entirely rather than passed on as weak evidence.

## Why a similarity floor exists

Nearest-neighbour search always returns its top k, even when nothing in the
corpus is relevant. Without a floor, a question the corpus cannot answer still
produces k confident-looking chunks, and the model is asked to answer from
them. The floor converts "no answer exists" into an explicit refusal.

## Index configuration

Embeddings are indexed with HNSW over cosine distance, built with `m = 16` and
`ef_construction = 64`. Query-time breadth is controlled by `hnsw.ef_search`,
which trades latency for recall.

## Known limitation: filtered search

The HNSW index is searched before tenant and source predicates are applied. A
highly selective filter can therefore return fewer rows than requested even when
more matching rows exist in the table. With a single tenant this cannot occur.
The fix is either partial indexes per tenant or pgvector's iterative scan.

## Chunk boundaries

Chunks never span a section heading. A chunk covering two sections would be
labelled with only one section's heading path, and a citation naming the wrong
section is worse than one naming no section at all.
