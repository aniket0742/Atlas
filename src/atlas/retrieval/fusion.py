"""Reciprocal Rank Fusion.

Combining a dense result list with a lexical one requires deciding what to
combine. The two obvious options:

*Weighted score blending.* Normalise both score sets and take a weighted sum.
This fails on the scales involved. Cosine similarity from this embedding model
occupies a narrow band around 0.6-0.9, while `ts_rank_cd` with normalisation 32
is bounded to [0, 1) but distributed completely differently and is not
comparable across queries -- a lexical score of 0.56 means "best match for this
query", not "56% relevant". Min-max normalising per query makes the top result
of every query score 1.0 regardless of whether it is any good, which actively
destroys the signal that a query has no good matches at all.

*Rank fusion.* Combine positions instead of scores. Ranks are on the same scale
by construction, so no normalisation is needed and no assumption about score
distributions is made.

RRF, from Cormack, Clarke and Buettcher (2009), scores a document as the sum
over result lists of 1 / (k + rank). The constant k damps the influence of top
ranks so that a document appearing at rank 1 in one list and nowhere in the
other does not automatically beat a document ranked 2 and 3 in both. k=60 is the
value from the original paper and is the usual default; it is exposed as
configuration so it can be swept rather than trusted.

The trade-off, stated plainly: RRF discards score magnitude. A dense match at
0.95 and one at 0.62 contribute identically if they hold the same rank. That
loses real information, and it is the price of not having to pretend two
incomparable scales are comparable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from atlas.core.models import RetrievedChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievedChunk]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse several ranked lists into one.

    `rankings` maps a component name ("dense", "lexical") to that component's
    results in rank order. The returned chunks carry each component's raw score
    and 1-based rank in `component_scores`, plus the fused `rrf` score, so a
    result can be explained rather than just displayed.
    """
    if not rankings:
        return []

    merged: dict[object, RetrievedChunk] = {}
    fused: dict[object, float] = {}

    for component, results in rankings.items():
        for position, chunk in enumerate(results):
            key = chunk.chunk_id
            if key not in merged:
                # Copy so the caller's lists are not mutated, and so a chunk
                # object shared between components does not accumulate state.
                merged[key] = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_external_id=chunk.document_external_id,
                    document_title=chunk.document_title,
                    document_uri=chunk.document_uri,
                    source_name=chunk.source_name,
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    heading_path=list(chunk.heading_path),
                    score=0.0,
                    component_scores={},
                )
                fused[key] = 0.0

            target = merged[key]
            # Raw component score, for display and for the similarity gate.
            target.component_scores[component] = chunk.component_scores.get(
                component, chunk.score
            )
            target.component_scores[f"{component}_rank"] = float(position + 1)
            fused[key] += 1.0 / (k + position + 1)

    for key, total in fused.items():
        merged[key].score = total
        merged[key].component_scores["rrf"] = total

    # Ties broken by chunk id so the ordering is deterministic across runs --
    # an eval harness comparing two configurations must not see spurious
    # differences from dictionary or database ordering.
    ordered = sorted(merged.values(), key=lambda c: (-c.score, str(c.chunk_id)))
    return ordered[:limit] if limit is not None else ordered
