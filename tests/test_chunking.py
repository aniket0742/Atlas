"""Chunking invariants.

These are the properties citations depend on. If any of them breaks, a citation
can point at text the document does not contain, which is a groundedness failure
dressed up as a formatting bug.
"""

from __future__ import annotations

import pytest

from atlas.ingest.chunking import chunk_document
from atlas.ingest.normalize import normalize_text

TARGET = 60
OVERLAP = 15
MIN = 8


def chunk(content: str, embedder, **kwargs):
    return chunk_document(
        content,
        embedder,
        target_tokens=kwargs.get("target", TARGET),
        overlap_tokens=kwargs.get("overlap", OVERLAP),
        min_tokens=kwargs.get("minimum", MIN),
    )


SAMPLE = normalize_text(
    """# Handbook

## Refunds

Customers may request a refund within 30 days of purchase. Refunds go back to
the original payment method.

Annual plans are prorated from the cancellation date.

## Disputes

Disputes are escalated to the billing team.

### Chargebacks

Chargebacks are handled by finance.

## Contact

Email billing@example.com.
"""
)


def test_text_is_exactly_the_span_it_claims(embedder):
    """The invariant citation resolution rests on."""
    for c in chunk(SAMPLE, embedder):
        assert c.text == SAMPLE[c.char_start : c.char_end]


def test_no_chunk_exceeds_the_token_budget(embedder):
    for c in chunk(SAMPLE, embedder):
        assert c.token_count <= TARGET


def test_budget_is_clamped_to_model_limit(embedder):
    """A configured target larger than the model's window must not be honoured."""
    chunks = chunk(SAMPLE, embedder, target=100_000)
    for c in chunks:
        assert c.token_count <= embedder.max_tokens


def test_ordinals_are_contiguous_from_zero(embedder):
    chunks = chunk(SAMPLE, embedder)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_covers_document_without_gaps(embedder):
    """Every character of the document is inside some chunk, or is whitespace.

    Gaps are the blank lines between blocks; losing non-whitespace content would
    make it permanently unretrievable.
    """
    chunks = chunk(SAMPLE, embedder)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(SAMPLE)
    for a, b in zip(chunks, chunks[1:], strict=False):
        if b.char_start > a.char_end:
            assert SAMPLE[a.char_end : b.char_start].strip() == ""


def test_chunk_does_not_span_a_section_boundary(embedder):
    """A chunk's heading path must actually describe all the text in it.

    Regression guard: an earlier version merged the trailing "## Contact"
    section into the preceding chunk and labelled the whole thing "Chargebacks".
    """
    for c in chunk(SAMPLE, embedder):
        body = c.text
        if c.heading_path:
            # Any heading inside the chunk must be the one it is labelled with,
            # or one of its ancestors (a title stack at the start of a chunk).
            headings = [
                line.lstrip("#").strip()
                for line in body.split("\n")
                if line.startswith("#")
            ]
            for heading in headings:
                assert heading in c.heading_path, (
                    f"chunk labelled {c.heading_path} contains heading {heading!r}"
                )


def test_short_section_survives_as_its_own_chunk(embedder):
    """Undersized sections must not be silently dropped."""
    chunks = chunk(SAMPLE, embedder)
    assert any("billing@example.com" in c.text for c in chunks)
    contact = next(c for c in chunks if "billing@example.com" in c.text)
    assert contact.heading_path[-1] == "Contact"


def test_heading_path_is_the_deepest_containing_section(embedder):
    chunks = chunk(SAMPLE, embedder)
    chargebacks = next(c for c in chunks if "handled by finance" in c.text)
    assert chargebacks.heading_path == ["Handbook", "Disputes", "Chargebacks"]


def test_overlap_does_not_loop_forever(embedder):
    """A pathological overlap setting must still terminate."""
    chunks = chunk(SAMPLE, embedder, target=20, overlap=19)
    assert 0 < len(chunks) < 500
    for c in chunks:
        assert c.text == SAMPLE[c.char_start : c.char_end]


def test_single_oversized_block_is_split(embedder):
    """One enormous paragraph with no sentence breaks still has to fit."""
    content = normalize_text("word " * 400)
    chunks = chunk(content, embedder)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= TARGET
        assert c.text == content[c.char_start : c.char_end]


@pytest.mark.parametrize(
    "content",
    ["", "   ", "\n\n\n", "single", "# Only a heading"],
)
def test_degenerate_inputs_do_not_raise(content, embedder):
    normalized = normalize_text(content)
    chunks = chunk(normalized, embedder)
    for c in chunks:
        assert c.text == normalized[c.char_start : c.char_end]
    if not normalized:
        assert chunks == []


def test_chunking_is_deterministic(embedder):
    first = chunk(SAMPLE, embedder)
    second = chunk(SAMPLE, embedder)
    assert [(c.char_start, c.char_end, c.token_count) for c in first] == [
        (c.char_start, c.char_end, c.token_count) for c in second
    ]
