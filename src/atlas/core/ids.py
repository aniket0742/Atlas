"""Deterministic identifiers.

Ingestion must be idempotent: submitting the same document twice has to produce
the same rows, not duplicates. Rather than SELECT-then-INSERT (which races), all
primary keys are derived from the logical identity of the thing via uuid5. Two
workers processing the same document concurrently compute the same id and the
second one's INSERT ... ON CONFLICT collapses into an update.
"""

from __future__ import annotations

import hashlib
import uuid

# Arbitrary but fixed. Changing this invalidates every id in the system.
ATLAS_NAMESPACE = uuid.UUID("8f7d3a1e-4b6c-5d2f-9e18-0a3c7b5d9e41")


def tenant_id(slug: str) -> uuid.UUID:
    return uuid.uuid5(ATLAS_NAMESPACE, f"tenant:{slug}")


def source_id(tenant: uuid.UUID, name: str) -> uuid.UUID:
    return uuid.uuid5(ATLAS_NAMESPACE, f"source:{tenant}:{name}")


def document_id(tenant: uuid.UUID, source: uuid.UUID, external_id: str) -> uuid.UUID:
    """Identity of a document is (tenant, source, source's own id for it).

    Deliberately excludes content: a document that changes is the *same*
    document at a new version, not a new document.
    """
    return uuid.uuid5(ATLAS_NAMESPACE, f"document:{tenant}:{source}:{external_id}")


def chunk_id(document: uuid.UUID, version: int, ordinal: int) -> uuid.UUID:
    return uuid.uuid5(ATLAS_NAMESPACE, f"chunk:{document}:{version}:{ordinal}")


def content_hash(data: bytes) -> str:
    """Change-detection hash over raw bytes.

    sha256 rather than a faster non-cryptographic hash: this decides whether we
    skip re-indexing, and a collision would silently serve stale content.
    """
    return hashlib.sha256(data).hexdigest()
