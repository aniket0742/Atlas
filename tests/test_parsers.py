"""Parsing and normalisation.

Malformed input arrives here first. The tests assert that bad documents fail
loudly with a diagnosis rather than indexing partially, because a document that
silently indexes half its content is undetectable downstream.
"""

from __future__ import annotations

import pytest

from atlas.ingest.normalize import normalize_text
from atlas.ingest.parsers import (
    UnparseableDocument,
    UnsupportedDocument,
    guess_mime_type,
    page_for_offset,
    parse,
)

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_line_endings_are_unified():
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_invisible_characters_are_stripped():
    """PDF extraction emits zero-width characters that break quote matching."""
    assert normalize_text("re​fund­ ed") == "refund ed"
    assert normalize_text("﻿hello") == "hello"


def test_runs_of_blank_lines_collapse():
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"


def test_trailing_whitespace_is_removed_per_line():
    assert normalize_text("a   \nb\t\n") == "a\nb"


def test_normalisation_is_idempotent():
    """It runs before offsets are computed, so a second pass must be a no-op."""
    messy = "  # Title \r\n\r\n\r\n\r\nbody​ text   \n\n"
    once = normalize_text(messy)
    assert normalize_text(once) == once


# ---------------------------------------------------------------------------
# Mime detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a.md", "text/markdown"),
        ("a.txt", "text/plain"),
        ("a.pdf", "application/pdf"),
        ("a.json", "application/json"),
        ("README", "text/plain"),
    ],
)
def test_mime_type_from_extension(filename, expected):
    assert guess_mime_type(filename) == expected


def test_declared_mime_type_wins_except_when_it_is_a_non_answer():
    assert guess_mime_type("a.md", "text/plain") == "text/plain"
    # Browsers send this for anything they cannot identify; fall back to the name.
    assert guess_mime_type("a.pdf", "application/octet-stream") == "application/pdf"
    assert guess_mime_type("a.md", "text/markdown; charset=utf-8") == "text/markdown"


# ---------------------------------------------------------------------------
# Text documents
# ---------------------------------------------------------------------------


def test_parses_markdown_and_derives_the_title_from_the_h1():
    doc = parse(b"# Billing Policy\n\nRefunds within 30 days.", external_id="billing.md")
    assert doc.title == "Billing Policy"
    assert doc.mime_type == "text/markdown"
    assert "Refunds within 30 days." in doc.content
    assert doc.byte_size == 41


def test_title_falls_back_to_the_first_non_empty_line():
    doc = parse(b"Some plain document\n\nmore text", external_id="a.txt")
    assert doc.title == "Some plain document"


def test_content_hash_is_over_raw_bytes():
    a = parse(b"# T\n\nbody", external_id="a.md")
    b = parse(b"# T\n\nbody", external_id="a.md")
    c = parse(b"# T\n\nbody!", external_id="a.md")
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash


def test_non_utf8_text_is_decoded_rather_than_rejected():
    doc = parse("café".encode("cp1252"), external_id="a.txt")
    assert "caf" in doc.content


def test_utf8_bom_is_handled():
    doc = parse("﻿# Title\n\nbody".encode(), external_id="a.md")
    assert doc.title == "Title"


def test_empty_document_is_rejected():
    with pytest.raises(UnparseableDocument):
        parse(b"   \n\n  ", external_id="a.md")


def test_unsupported_type_is_rejected_with_a_useful_message():
    with pytest.raises(UnsupportedDocument, match="Cannot parse"):
        parse(b"\x00\x01binary", external_id="a.bin", mime_type="image/png")


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal real PDF with one text run per page.

    pypdf's writer cannot lay out text, so the page is assembled by hand: a
    content stream with a Tj operator, plus a /Resources dictionary declaring
    the font it references. Without the font resource extract_text returns
    nothing, which would make this fixture test the wrong thing.
    """
    pypdf = pytest.importorskip("pypdf")
    import io

    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)

        stream = DecodedStreamObject()
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)

        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_corrupt_pdf_fails_with_a_diagnosis_not_silently():
    with pytest.raises(UnparseableDocument):
        parse(b"%PDF-1.4\nnot actually a pdf", external_id="a.pdf")


def test_pdf_with_no_extractable_text_is_rejected():
    """A scanned PDF needs OCR; indexing it as empty would hide the problem."""
    pdf = make_pdf([])
    with pytest.raises(UnparseableDocument, match="no extractable text|OCR|Could not read"):
        parse(pdf, external_id="scan.pdf")


def test_pdf_records_page_offsets_for_citation():
    pdf = make_pdf(["Refunds are available within 30 days.", "Chargebacks go to finance."])
    doc = parse(pdf, external_id="policy.pdf")

    assert doc.mime_type == "application/pdf"
    offsets = doc.metadata["page_offsets"]
    assert len(offsets) == 2

    # Offsets must index into the content that is actually stored.
    for offset, page in offsets:
        assert 0 <= offset <= len(doc.content)
        assert page_for_offset(offsets, offset) == page

    first_page_start = offsets[0][0]
    second_page_start = offsets[1][0]
    assert page_for_offset(offsets, first_page_start) == 1
    assert page_for_offset(offsets, second_page_start - 1) == 1
    assert page_for_offset(offsets, second_page_start) == 2


def test_page_lookup_is_none_for_documents_without_pages():
    assert page_for_offset(None, 10) is None
    assert page_for_offset([], 10) is None
