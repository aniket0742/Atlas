"""Identifiers, metrics and eval-dataset validation."""

from __future__ import annotations

import json
import uuid

import pytest

from atlas.core import ids
from atlas.eval import metrics
from atlas.eval.dataset import Label, load

# ---------------------------------------------------------------------------
# Deterministic ids -- the basis of idempotent ingestion
# ---------------------------------------------------------------------------


def test_ids_are_stable_across_calls():
    tenant = ids.tenant_id("acme")
    source = ids.source_id(tenant, "handbook")
    assert ids.tenant_id("acme") == tenant
    assert ids.source_id(tenant, "handbook") == source
    assert ids.document_id(tenant, source, "a.md") == ids.document_id(tenant, source, "a.md")


def test_document_identity_excludes_content():
    """A changed document is the same document at a new version."""
    tenant = ids.tenant_id("acme")
    source = ids.source_id(tenant, "handbook")
    assert ids.document_id(tenant, source, "a.md") == ids.document_id(tenant, source, "a.md")


def test_different_tenants_never_share_a_document_id():
    """The property that makes cross-tenant collision impossible by construction."""
    a, b = ids.tenant_id("acme"), ids.tenant_id("globex")
    source_a, source_b = ids.source_id(a, "kb"), ids.source_id(b, "kb")
    assert a != b
    assert source_a != source_b
    assert ids.document_id(a, source_a, "same.md") != ids.document_id(b, source_b, "same.md")


def test_chunk_id_changes_with_version():
    doc = uuid.uuid4()
    assert ids.chunk_id(doc, 1, 0) != ids.chunk_id(doc, 2, 0)
    assert ids.chunk_id(doc, 1, 0) != ids.chunk_id(doc, 1, 1)


def test_content_hash_detects_any_change():
    assert ids.content_hash(b"hello") == ids.content_hash(b"hello")
    assert ids.content_hash(b"hello") != ids.content_hash(b"hello ")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_recall_counts_labels_not_chunks():
    """Several chunks can satisfy one label; recall must not double-count."""
    assert metrics.recall_at_k(1, 2) == 0.5
    assert metrics.recall_at_k(2, 2) == 1.0
    assert metrics.recall_at_k(0, 0) == 0.0


def test_reciprocal_rank_is_one_based():
    assert metrics.reciprocal_rank([0]) == 1.0
    assert metrics.reciprocal_rank([1]) == 0.5
    assert metrics.reciprocal_rank([3, 1]) == 0.5  # uses the best rank
    assert metrics.reciprocal_rank([]) == 0.0


def test_precision_only_counts_within_k():
    assert metrics.precision_at_k([0, 1], 4) == 0.5
    assert metrics.precision_at_k([0, 9], 4) == 0.25
    assert metrics.precision_at_k([], 4) == 0.0


def test_ndcg_is_perfect_only_when_all_labels_are_at_the_top():
    assert metrics.ndcg_at_k([0], 1, 10) == pytest.approx(1.0)
    assert metrics.ndcg_at_k([0, 1], 2, 10) == pytest.approx(1.0)
    # One of two labels found at rank 1 must not score 1.0.
    assert metrics.ndcg_at_k([0], 2, 10) < 1.0
    assert metrics.ndcg_at_k([], 2, 10) == 0.0


def test_ndcg_never_exceeds_one_when_several_chunks_share_a_label():
    """Regression: the first baseline run reported nDCG 1.0164.

    Overlapping chunks mean one label can be satisfied at several ranks. nDCG
    takes one position per label, so passing three positions for a single label
    is a caller error -- but the metric must stay bounded for any input it is
    given, because an unbounded 'normalised' score is silently meaningless.
    """
    for total_labels in (1, 2, 3):
        for positions in ([0], [0, 1], [0, 1, 2], [0, 1, 2, 3, 4]):
            score = metrics.ndcg_at_k(positions[:total_labels], total_labels, 10)
            assert 0.0 <= score <= 1.0, (positions, total_labels, score)


def test_score_query_uses_label_positions_for_ndcg_not_every_hit():
    """Three chunks satisfying one label is still a perfect ranking, not 2x."""
    scores = metrics.score_query(
        query_id="q",
        relevant_positions=[0, 1, 2],   # three chunks matched
        label_positions=[0],            # but they all satisfy the same label
        matched_labels=1,
        total_labels=1,
        retrieved=8,
        k=8,
    )
    assert scores.ndcg_at_k == pytest.approx(1.0)
    assert scores.recall_at_k == pytest.approx(1.0)
    # Precision legitimately counts all three retrieved relevant chunks.
    assert scores.precision_at_k == pytest.approx(3 / 8)


def test_ndcg_rewards_higher_ranks():
    assert metrics.ndcg_at_k([0], 1, 10) > metrics.ndcg_at_k([5], 1, 10)


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.3, 0.7]
    low, high = metrics.bootstrap_ci(values)
    assert low <= metrics.mean(values) <= high
    assert metrics.bootstrap_ci(values) == (low, high)


def test_bootstrap_ci_of_identical_values_has_zero_width():
    low, high = metrics.bootstrap_ci([0.5] * 10)
    assert low == high == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Eval dataset
# ---------------------------------------------------------------------------


def test_label_matching_requires_the_right_document():
    label = Label(document="a.md", contains="30 days")
    assert label.matches("a.md", "refunds within 30 days")
    assert not label.matches("b.md", "refunds within 30 days")


def test_label_matching_ignores_case_and_whitespace():
    """Labels must survive re-chunking, which reflows whitespace."""
    label = Label(document="a.md", contains="within 30 days")
    assert label.matches("a.md", "refunds\nWITHIN   30\ndays of purchase")


def test_label_without_a_snippet_matches_any_chunk_of_the_document():
    assert Label(document="a.md").matches("a.md", "anything at all")


def write(tmp_path, records):
    path = tmp_path / "set.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def test_loads_a_valid_dataset(tmp_path):
    path = write(
        tmp_path,
        [
            {"id": "a", "question": "q?", "labels": [{"document": "d.md", "contains": "x"}]},
            {"id": "b", "question": "q2?", "answerable": False},
        ],
    )
    queries = load(path)
    assert [q.id for q in queries] == ["a", "b"]
    assert queries[1].answerable is False


def test_rejects_an_answerable_query_with_no_labels(tmp_path):
    """The silent-zero-recall trap: an unlabelled query scores 0 forever."""
    path = write(tmp_path, [{"id": "a", "question": "q?"}])
    with pytest.raises(ValueError, match="marked answerable but has"):
        load(path)


def test_rejects_an_unanswerable_query_that_has_labels(tmp_path):
    path = write(
        tmp_path,
        [{"id": "a", "question": "q?", "answerable": False, "labels": [{"document": "d.md"}]}],
    )
    with pytest.raises(ValueError, match="unanswerable"):
        load(path)


def test_rejects_duplicate_ids(tmp_path):
    path = write(
        tmp_path,
        [
            {"id": "a", "question": "q?", "labels": [{"document": "d.md"}]},
            {"id": "a", "question": "q2?", "labels": [{"document": "d.md"}]},
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        load(path)


def test_comments_and_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "set.jsonl"
    path.write_text(
        '// a comment\n\n{"id": "a", "question": "q?", "labels": [{"document": "d.md"}]}\n',
        encoding="utf-8",
    )
    assert len(load(path)) == 1


@pytest.mark.parametrize("name", ["smoke.jsonl", "main.jsonl"])
def test_shipped_datasets_are_valid_and_their_labels_exist(name):
    """Guards against a label whose snippet was edited out of the corpus.

    A label that matches nothing scores zero forever and is indistinguishable
    from a retrieval failure, so editing the corpus without updating the dataset
    must fail here rather than quietly degrade the numbers.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    queries = load(root / "eval" / "datasets" / name)
    assert len(queries) >= 15
    assert any(not q.answerable for q in queries), "need unanswerable queries to score refusal"

    for query in queries:
        for label in query.labels:
            text = (root / "eval" / "corpus" / label.document).read_text(encoding="utf-8")
            assert label.matches(label.document, text), (
                f"{query.id}: snippet {label.contains!r} is not in {label.document}"
            )


def test_main_dataset_classifies_every_query():
    """Per-kind metric breakdowns are meaningless if queries are unclassified."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    queries = load(root / "eval" / "datasets" / "main.jsonl")
    unclassified = [q.id for q in queries if q.kind == "unclassified"]
    assert not unclassified, f"unclassified queries: {unclassified}"
    assert sum(not q.answerable for q in queries) >= 10, (
        "too few unanswerable queries to measure refusal behaviour"
    )


def test_paired_bootstrap_detects_a_consistent_small_difference():
    """The reason paired testing exists.

    Every query improves by exactly 0.1, but query difficulty varies hugely.
    Independent intervals would be dominated by that spread and overlap; the
    paired interval sees a constant difference and excludes zero.
    """
    baseline = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.15, 0.85, 0.25, 0.75]
    candidate = [b + 0.1 for b in baseline]

    delta, (low, high) = metrics.paired_bootstrap_delta(baseline, candidate)
    assert delta == pytest.approx(0.1)
    assert low > 0, "a constant improvement must be detected"

    # The unpaired view of the same data cannot tell them apart.
    assert metrics.bootstrap_ci(baseline)[1] > metrics.bootstrap_ci(candidate)[0]


def test_paired_bootstrap_reports_no_difference_for_noise():
    baseline = [0.5, 0.4, 0.6, 0.55, 0.45, 0.5, 0.6, 0.4]
    candidate = [0.4, 0.5, 0.55, 0.6, 0.5, 0.45, 0.4, 0.6]
    delta, (low, high) = metrics.paired_bootstrap_delta(baseline, candidate)
    assert low <= 0 <= high, "noise must not be reported as a difference"


def test_paired_bootstrap_is_deterministic_and_signed():
    a = [0.2, 0.4, 0.6]
    b = [0.3, 0.5, 0.7]
    assert metrics.paired_bootstrap_delta(a, b) == metrics.paired_bootstrap_delta(a, b)
    up, _ = metrics.paired_bootstrap_delta(a, b)
    down, _ = metrics.paired_bootstrap_delta(b, a)
    assert up == pytest.approx(-down)


def test_paired_bootstrap_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="align"):
        metrics.paired_bootstrap_delta([0.1, 0.2], [0.1])
