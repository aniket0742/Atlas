-- Lexical retrieval over chunks (Phase 2).
--
-- This is PostgreSQL full-text search, NOT BM25. ts_rank_cd is a coverage
-- density ranking: it rewards proximity and count of matching lexemes but does
-- not implement Okapi BM25's term saturation or its document-length
-- normalisation. Calling it BM25 would be wrong, and the distinction is
-- recorded in ADR-0017 rather than glossed.
--
-- The generated column means the tsvector is maintained by Postgres on every
-- write, so it cannot drift from `text` the way a trigger-maintained or
-- application-maintained column can.

ALTER TABLE chunks
    ADD COLUMN text_search tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

-- The two-argument form of to_tsvector is IMMUTABLE; the one-argument form
-- depends on default_text_search_config and is only STABLE, which a generated
-- column will not accept. Pinning 'english' is therefore required, not a
-- preference -- and it also means the index cannot silently change meaning if a
-- database-level setting changes.

CREATE INDEX chunks_text_search_idx ON chunks USING gin (text_search);

-- Tenant filtering happens in the query, as it does for vector search. Unlike
-- the HNSW index, a GIN index composes normally with a WHERE clause -- Postgres
-- can combine the GIN scan with the tenant predicate via a bitmap scan, so the
-- ADR-0003 under-fill problem does not apply to lexical retrieval.
CREATE INDEX chunks_tenant_lexical_idx ON chunks (tenant_id) INCLUDE (document_id);
