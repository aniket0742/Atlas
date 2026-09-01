"""Source parsers: bytes in, normalised text plus metadata out.

Parsing is where malformed input arrives, so every parser is expected to fail
loudly on content it cannot handle rather than return partial text. A document
that silently indexes as three pages of an eighty-page PDF is worse than one
that fails ingestion, because nothing downstream can detect it.

PDF page boundaries are recorded as character offsets in metadata so a citation
can name a page number. Without that, a PDF citation can only say "somewhere in
this document", which is not useful to a reader trying to verify a claim.
"""

from __future__ import annotations

import io
import re
from typing import Any

from atlas.core.ids import content_hash
from atlas.core.models import ParsedDocument
from atlas.ingest.normalize import normalize_text

TEXT_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "text/csv",
    "application/json",
    "application/x-yaml",
    "text/yaml",
}

_EXTENSION_MIME = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".rst": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".pdf": "application/pdf",
    ".py": "text/plain",
    ".sql": "text/plain",
}

_MD_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class UnsupportedDocument(ValueError):
    """The document's type is not something Atlas can parse."""


class UnparseableDocument(ValueError):
    """The document is of a supported type but could not be read."""


def guess_mime_type(filename: str, declared: str | None = None) -> str:
    if declared and declared != "application/octet-stream":
        return declared.split(";")[0].strip()
    lowered = filename.lower()
    for ext, mime in _EXTENSION_MIME.items():
        if lowered.endswith(ext):
            return mime
    return "text/plain"


def parse(
    data: bytes,
    *,
    external_id: str,
    filename: str | None = None,
    mime_type: str | None = None,
    uri: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ParsedDocument:
    mime = guess_mime_type(filename or external_id, mime_type)
    meta: dict[str, Any] = dict(metadata or {})

    if mime == "application/pdf":
        content, pdf_meta = _parse_pdf(data)
        meta.update(pdf_meta)
    elif mime in TEXT_MIME_TYPES:
        content = _parse_text(data)
    else:
        raise UnsupportedDocument(
            f"Cannot parse {mime!r}. Supported: application/pdf, "
            + ", ".join(sorted(TEXT_MIME_TYPES))
        )

    if not content.strip():
        raise UnparseableDocument(
            f"{external_id!r} produced no extractable text. If it is a scanned "
            "PDF it needs OCR, which Atlas does not do."
        )

    title = meta.pop("_title", None) or _derive_title(content, filename or external_id)

    return ParsedDocument(
        external_id=external_id,
        content=content,
        title=title,
        uri=uri,
        mime_type=mime,
        byte_size=len(data),
        # Hash the raw bytes, not the normalised text: it is change detection
        # for the source artefact, and it stays valid if normalisation changes.
        content_hash=content_hash(data),
        metadata=meta,
    )


def _parse_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return normalize_text(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise UnparseableDocument("Could not decode as text in utf-8, cp1252 or latin-1")


def _parse_pdf(data: bytes) -> tuple[str, dict[str, Any]]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedDocument("pypdf is not installed") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # An empty-password decrypt succeeds for PDFs that are "protected"
            # only against editing, which is common for policy documents.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise UnparseableDocument("PDF is encrypted and needs a password") from exc

        pages: list[str] = []
        for index, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:
                raise UnparseableDocument(f"Failed to extract page {index + 1}: {exc}") from exc
    except UnparseableDocument:
        raise
    except PdfReadError as exc:
        raise UnparseableDocument(f"Malformed PDF: {exc}") from exc
    except Exception as exc:
        raise UnparseableDocument(f"Could not read PDF: {exc}") from exc

    # Build the text and the page offset map together so the offsets describe
    # the text that is actually stored.
    parts: list[str] = []
    page_offsets: list[list[int]] = []
    cursor = 0
    for page_number, raw_page in enumerate(pages, start=1):
        page_text = normalize_text(raw_page)
        if not page_text:
            continue
        if parts:
            cursor += 2  # the "\n\n" joiner
        page_offsets.append([cursor, page_number])
        parts.append(page_text)
        cursor += len(page_text)

    content = "\n\n".join(parts)

    meta: dict[str, Any] = {
        "page_count": len(reader.pages),
        "pages_with_text": len(parts),
        # [[char_offset, page_number], ...]; citations resolve a char offset to
        # the last entry whose offset is <= it.
        "page_offsets": page_offsets,
    }
    info = reader.metadata
    if info:
        if info.title:
            meta["_title"] = str(info.title).strip() or None
        if info.author:
            meta["author"] = str(info.author)
    return content, meta


def _derive_title(content: str, fallback: str) -> str:
    """Prefer a markdown H1, then the first non-empty line, then the filename."""
    match = _MD_H1.search(content[:2000])
    if match:
        return match.group(1).strip()[:300]
    for line in content.split("\n", 20):
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:300]
    return fallback


def page_for_offset(page_offsets: list[list[int]] | None, offset: int) -> int | None:
    """Resolve a character offset to a PDF page number, if the doc has pages."""
    if not page_offsets:
        return None
    page = None
    for start, number in page_offsets:
        if start <= offset:
            page = number
        else:
            break
    return page
