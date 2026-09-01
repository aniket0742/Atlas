-- Atlas initial schema.
--
-- Design notes (rationale lives in Decision.md, ADR-0003 / ADR-0004):
--
--  * tenant_id is on every row from the first migration even though Phase 1
--    runs with a single tenant and no auth. Retrofitting tenancy into indexes
--    and queries later is how cross-tenant leaks happen.
--
--  * Embeddings live in their own table, not as a column on chunks, so that
--    two embedding models can be indexed over identical chunks and compared
--    head-to-head by the eval harness without re-chunking.
--
--  * Chunks carry character offsets into the normalised document text so a
--    citation can be resolved back to an exact span, not just "some chunk".

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- --------------------------------------------------------------------------
-- Tenancy
-- --------------------------------------------------------------------------

CREATE TABLE tenants (
    id          uuid PRIMARY KEY,
    slug        text NOT NULL UNIQUE,
    name        text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- --------------------------------------------------------------------------
-- Sources: a connected knowledge source (an upload bucket, a repo, a site).
-- --------------------------------------------------------------------------

CREATE TABLE sources (
    id          uuid PRIMARY KEY,
    tenant_id   uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    kind        text NOT NULL CHECK (kind IN ('upload', 'filesystem', 'web', 'github')),
    name        text NOT NULL,
    config      jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, name)
);

CREATE INDEX sources_tenant_idx ON sources (tenant_id);

-- --------------------------------------------------------------------------
-- Documents
--
-- external_id is the source's own stable identifier (a path, a URL, a blob
-- key). (tenant_id, source_id, external_id) is the natural key, and the
-- document uuid is derived from it (uuid5) so re-ingesting the same logical
-- document is idempotent without a lookup.
--
-- content_hash drives change detection: identical hash means skip the whole
-- parse/chunk/embed pipeline.
-- --------------------------------------------------------------------------

CREATE TABLE documents (
    id             uuid PRIMARY KEY,
    tenant_id      uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    source_id      uuid NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    external_id    text NOT NULL,
    uri            text,
    title          text,
    mime_type      text,
    content_hash   text NOT NULL,
    version        integer NOT NULL DEFAULT 1,
    -- Normalised plain text. Kept so citations can be resolved to a span and
    -- so re-chunking never needs the original binary.
    content        text NOT NULL,
    byte_size      bigint,
    metadata       jsonb NOT NULL DEFAULT '{}'::jsonb,
    status         text NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending', 'indexed', 'failed')),
    error          text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    indexed_at     timestamptz,
    UNIQUE (tenant_id, source_id, external_id)
);

CREATE INDEX documents_tenant_source_idx ON documents (tenant_id, source_id);
CREATE INDEX documents_status_idx        ON documents (tenant_id, status);

-- --------------------------------------------------------------------------
-- Chunks
--
-- document_version is denormalised onto the chunk so that a re-index can write
-- the new version's chunks and delete the old version's in one transaction,
-- without a window where the document has no retrievable content.
-- --------------------------------------------------------------------------

CREATE TABLE chunks (
    id                uuid PRIMARY KEY,
    tenant_id         uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    document_id       uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    document_version  integer NOT NULL,
    ordinal           integer NOT NULL,
    text              text NOT NULL,
    token_count       integer NOT NULL,
    -- Character offsets into documents.content. Half-open [start, end).
    char_start        integer NOT NULL,
    char_end          integer NOT NULL,
    -- Breadcrumb of headings this chunk sits under, e.g. {"Billing","Refunds"}.
    heading_path      text[] NOT NULL DEFAULT '{}',
    metadata          jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at        timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, document_version, ordinal)
);

CREATE INDEX chunks_tenant_idx   ON chunks (tenant_id);
CREATE INDEX chunks_document_idx ON chunks (document_id, document_version);

-- --------------------------------------------------------------------------
-- Embeddings
--
-- pgvector requires a fixed dimensionality per column to build an index, so
-- the dimension is pinned here. Adding an embedding model with a different
-- dimension means a new table, not a new row -- see ADR-0004.
--
-- 384 dimensions matches the BAAI/bge-small-en-v1.5 family used in Phase 1.
-- --------------------------------------------------------------------------

CREATE TABLE chunk_embeddings (
    chunk_id    uuid NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    tenant_id   uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    model       text NOT NULL,
    embedding   vector(384) NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, model)
);

-- HNSW over cosine distance.
--
-- Note this index is NOT tenant-partitioned: pgvector applies the ANN index
-- first and filters afterwards, so a highly selective tenant filter can under-
-- fill the result set. Phase 1 has one tenant so it is not yet a problem; the
-- fix (partial indexes per tenant, or iterative scan) is recorded in ADR-0003
-- and revisited in Phase 5 with numbers.
CREATE INDEX chunk_embeddings_hnsw_cosine_idx
    ON chunk_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX chunk_embeddings_tenant_model_idx ON chunk_embeddings (tenant_id, model);
