"""Text normalisation.

Normalisation happens once, before chunking, and the normalised text is what
gets stored in documents.content. Every character offset in the system -- and
therefore every citation -- is an offset into this normalised text. If
normalisation ran after offsets were computed, citations would drift.
"""

from __future__ import annotations

import re

_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
# Zero-width space/non-joiner/joiner, BOM, and soft hyphen. PDF text extraction
# emits these routinely; they corrupt token counts and break quote matching
# between the model's output and the stored chunk text.
_INVISIBLE = re.compile("[​‌‍﻿­]")


def normalize_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    text = _TRAILING_WS.sub("", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()
