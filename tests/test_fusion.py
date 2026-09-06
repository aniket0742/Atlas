"""Reciprocal rank fusion.

These run without a database: fusion is pure arithmetic over ranked lists, and
the properties worth pinning are properties of that arithmetic.
"""

from __future__ import annotations

import uuid

import pytest

from atlas.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from tests.conftest import make_chunk


def chunks(n: int, prefix: str = "c") -> list:
    """n chunks with stable ids so tests can refer to them by position."""
    out = []
    for i in range(n):
        chunk_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{prefix}-{i}")
        out.append(make_chunk(f"text {prefix}{i}", chunk_id=chunk_id, score=1.0 - i * 0.1))
    return out


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"dense": []}) == []


def test_single_list_preserves_its_order():
    items = chunks(4)
    fused = reciprocal_rank_fusion({"dense": items})
    assert [c.chunk_id for c in fused] == [c.chunk_id for c in items]


def test_rrf_score_is_the_sum_of_reciprocal_ranks():
    items = chunks(2)
    # Same chunk at rank 1 in one list and rank 2 in the other.
    fused = reciprocal_rank_fusion(
        {"dense": [items[0], items[1]], "lexical": [items[1], items[0]]},
        k=DEFAULT_RRF_K,
    )
    expected = 1 / (DEFAULT_RRF_K + 1) + 1 / (DEFAULT_RRF_K + 2)
    for chunk in fused:
        assert chunk.score == pytest.approx(expected)


def test_agreement_beats_a_single_strong_hit():
    """The property RRF exists for.

    A chunk ranked 2nd by both components should outrank one ranked 1st by a
    single component and missing from the other -- that is the whole reason to
    fuse rather than take the best single list.
    """
    only_dense = make_chunk("only dense", chunk_id=uuid.uuid4())
    agreed = make_chunk("both agree", chunk_id=uuid.uuid4())
    filler = make_chunk("filler", chunk_id=uuid.uuid4())

    fused = reciprocal_rank_fusion(
        {"dense": [only_dense, agreed], "lexical": [filler, agreed]}
    )
    assert fused[0].chunk_id == agreed.chunk_id


def test_component_scores_and_ranks_are_recorded():
    """A fused result has to be explainable, not just ordered."""
    a, b = chunks(2)
    fused = reciprocal_rank_fusion({"dense": [a, b], "lexical": [b, a]})
    by_id = {c.chunk_id: c for c in fused}

    first = by_id[a.chunk_id]
    assert first.component_scores["dense_rank"] == 1.0
    assert first.component_scores["lexical_rank"] == 2.0
    assert "dense" in first.component_scores
    assert "lexical" in first.component_scores
    assert "rrf" in first.component_scores


def test_chunk_appearing_in_one_list_only_still_survives():
    a, b = chunks(2)
    fused = reciprocal_rank_fusion({"dense": [a], "lexical": [b]})
    assert {c.chunk_id for c in fused} == {a.chunk_id, b.chunk_id}
    # Each was seen once at rank 1, so scores tie.
    assert fused[0].score == pytest.approx(fused[1].score)


def test_ordering_is_deterministic_under_ties():
    """Two runs of the same eval must not differ because of dict ordering."""
    a, b = chunks(2)
    first = reciprocal_rank_fusion({"dense": [a], "lexical": [b]})
    second = reciprocal_rank_fusion({"lexical": [b], "dense": [a]})
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_limit_truncates_after_fusing_not_before():
    items = chunks(5)
    fused = reciprocal_rank_fusion(
        {"dense": items, "lexical": list(reversed(items))}, limit=2
    )
    assert len(fused) == 2


def test_smaller_k_sharpens_the_influence_of_top_ranks():
    """k damps rank-1 dominance; a smaller k should let a rank-1 hit matter more.

    Asserted on the two scores directly rather than on final positions. Other
    chunks in these lists can legitimately tie with them, and a tie is broken by
    chunk id -- so an assertion about position would depend on unrelated ids and
    be flaky, which it was.
    """
    top_only = make_chunk("rank 1 in dense only", chunk_id=uuid.uuid4())
    mid_both = make_chunk("rank 4 in both", chunk_id=uuid.uuid4())
    # Distinct padding per list, so mid_both is the only chunk both lists share.
    pad = [make_chunk(f"pad{i}", chunk_id=uuid.uuid4()) for i in range(5)]

    lists = {
        "dense": [top_only, pad[0], pad[1], mid_both],
        "lexical": [pad[2], pad[3], pad[4], mid_both],
    }

    def score_of(fused, chunk):
        return next(c.score for c in fused if c.chunk_id == chunk.chunk_id)

    big_k = reciprocal_rank_fusion(lists, k=1000)
    small_k = reciprocal_rank_fusion(lists, k=1)

    # Heavy damping: agreement across both lists outweighs one rank-1 hit.
    assert score_of(big_k, mid_both) > score_of(big_k, top_only)
    # Light damping: the rank-1 hit dominates.
    assert score_of(small_k, top_only) > score_of(small_k, mid_both)


def test_inputs_are_not_mutated():
    """Fusion must not write into the caller's chunks.

    The retriever reuses component result lists for diagnostics, and a fused
    copy that shared objects would let a later stage corrupt them.
    """
    a, b = chunks(2)
    original = dict(a.component_scores)
    reciprocal_rank_fusion({"dense": [a, b], "lexical": [b, a]})
    assert a.component_scores == original
