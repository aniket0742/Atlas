"""Retrieval metrics.

Definitions are written out explicitly rather than pulled from a library so that
what is being measured is auditable. Every one of these has more than one
convention in circulation, and a benchmark whose definitions are implicit is not
a benchmark.

All metrics here are computed against *graded-as-binary* relevance: a retrieved
chunk either satisfies a gold label or it does not. That is the honest ceiling
for a hand-labelled set of this size -- graded relevance would require multiple
annotators to be meaningful, and there is one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class QueryScores:
    query_id: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    matched: int
    total_relevant: int
    retrieved: int


def recall_at_k(matched_labels: int, total_labels: int) -> float:
    """Fraction of the gold labels that were found in the top k.

    Note this is recall over *labels*, not over chunks: a label can be satisfied
    by any chunk that carries the required evidence, because with overlapping
    chunks several chunks legitimately contain the same fact.
    """
    if total_labels == 0:
        return 0.0
    return matched_labels / total_labels


def precision_at_k(relevant_positions: list[int], k: int) -> float:
    """Fraction of the top k retrieved chunks that were relevant.

    Reported with a caveat: with overlapping chunks and one gold label, the
    theoretical maximum precision@8 may be well below 1.0, so the absolute value
    is not meaningful on its own. It is comparable *between* retrieval
    configurations on the same dataset, which is what it is used for.
    """
    if k == 0:
        return 0.0
    return len([p for p in relevant_positions if p < k]) / k


def reciprocal_rank(relevant_positions: list[int]) -> float:
    """1 / (1-based rank of the first relevant result), 0 if none."""
    if not relevant_positions:
        return 0.0
    return 1.0 / (min(relevant_positions) + 1)


def ndcg_at_k(relevant_positions: list[int], total_labels: int, k: int) -> float:
    """Binary-gain nDCG with the standard log2(rank+1) discount.

    The ideal ranking places min(total_labels, k) relevant items at the top,
    so IDCG is computed from the number of labels rather than from the number
    of relevant items actually retrieved -- otherwise a run that retrieves one
    of three relevant items at rank 1 would score a perfect 1.0.
    """
    dcg = sum(1.0 / math.log2(p + 2) for p in relevant_positions if p < k)
    ideal_hits = min(total_labels, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def score_query(
    query_id: str,
    relevant_positions: list[int],
    matched_labels: int,
    total_labels: int,
    retrieved: int,
    k: int,
) -> QueryScores:
    return QueryScores(
        query_id=query_id,
        recall_at_k=recall_at_k(matched_labels, total_labels),
        precision_at_k=precision_at_k(relevant_positions, k),
        reciprocal_rank=reciprocal_rank(relevant_positions),
        ndcg_at_k=ndcg_at_k(relevant_positions, total_labels, k),
        matched=matched_labels,
        total_relevant=total_labels,
        retrieved=retrieved,
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(
    values: list[float], *, confidence: float = 0.95, resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    This exists because the eval set is small. With 30 queries, a two-point
    difference in mean nDCG between two retrieval configurations is well inside
    sampling noise, and reporting it as an improvement would be dishonest. The
    interval makes the uncertainty visible in the report itself rather than
    leaving it as a caveat nobody reads.
    """
    if not values:
        return (0.0, 0.0)
    import random

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower = (1 - confidence) / 2
    return (
        means[int(lower * resamples)],
        means[min(resamples - 1, int((1 - lower) * resamples))],
    )
