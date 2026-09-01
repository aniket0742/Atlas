"""Structure-aware chunking.

Approach, and why:

Fixed-size character windows are the usual default and they fail in a specific
way -- they cut mid-sentence and mid-table, so a retrieved chunk often cannot be
read on its own and the model has to guess at the missing half. Doing better is
cheap here because most of the corpus (markdown, docs, READMEs) carries explicit
structure.

So: split on structure first (headings, then blank-line-delimited blocks), then
*pack* those blocks into token-budgeted windows. A chunk boundary therefore
always falls on a block boundary, unless a single block is itself over budget --
in which case it is split on sentence boundaries, and only if a single sentence
is over budget is it hard-split.

Two invariants the rest of the system relies on:

  1. chunk.text == document.content[chunk.char_start:chunk.char_end], exactly.
     Citations resolve by slicing, so any deviation surfaces as a wrong quote.
  2. chunk.token_count <= the embedding model's max_tokens, so nothing is
     silently truncated at embed time.

Whether this beats naive fixed-size chunking is an empirical question, and it is
measured by the eval harness rather than asserted -- see docs/evaluation.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from atlas.core.models import Chunk
from atlas.providers.base import EmbeddingProvider

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
# Sentence-ish boundary: terminator, optional closing quote/bracket, whitespace.
# Deliberately conservative -- over-splitting is recoverable, under-splitting
# blows the token budget.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


@dataclass(slots=True)
class _Block:
    start: int
    end: int
    heading_path: tuple[str, ...]
    tokens: int
    is_heading: bool = False


def _split_into_blocks(content: str) -> list[tuple[int, int, tuple[str, ...], bool]]:
    """Split content into blank-line-delimited blocks, tracking heading context.

    Returns (char_start, char_end, heading_path, is_heading) tuples. Headings are
    emitted as blocks of their own so that a section's title is retrievable
    alongside its body rather than being stripped out, and they carry a path that
    includes themselves so a chunk starting at a heading is labelled with that
    section.
    """
    blocks: list[tuple[int, int, tuple[str, ...], bool]] = []
    heading_stack: list[tuple[int, str]] = []

    offset = 0
    para_start: int | None = None
    para_end = 0

    def flush() -> None:
        nonlocal para_start
        if para_start is not None and para_end > para_start:
            blocks.append((para_start, para_end, tuple(h for _, h in heading_stack), False))
        para_start = None

    for line in content.splitlines(keepends=True):
        stripped = line.strip()
        line_start = offset
        offset += len(line)

        if not stripped:
            flush()
            continue

        match = _HEADING.match(stripped)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            # The heading block carries the path *including* itself, so a chunk
            # that starts at this heading is attributed to this section.
            path = tuple(h for _, h in heading_stack)
            indent = len(line) - len(line.lstrip())
            blocks.append(
                (line_start + indent, line_start + indent + len(stripped), path, True)
            )
            continue

        if para_start is None:
            para_start = line_start + (len(line) - len(line.lstrip()))
        para_end = line_start + len(line.rstrip())

    flush()
    return blocks


def _split_oversized(
    content: str, start: int, end: int, budget: int, embedder: EmbeddingProvider
) -> list[tuple[int, int]]:
    """Split a single over-budget span on sentence boundaries, then hard-split."""
    text = content[start:end]
    pieces: list[tuple[int, int]] = []

    cursor = start
    for part in _SENTENCE_END.split(text):
        if not part:
            continue
        # Locate each piece in the original string rather than reconstructing
        # offsets from the split, so offsets stay exact.
        idx = content.find(part, cursor, end)
        if idx == -1:
            continue
        pieces.append((idx, idx + len(part)))
        cursor = idx + len(part)

    if not pieces:
        pieces = [(start, end)]

    out: list[tuple[int, int]] = []
    for p_start, p_end in pieces:
        if embedder.count_tokens(content[p_start:p_end]) <= budget:
            out.append((p_start, p_end))
            continue
        # Still over budget: a giant table row, minified JSON, a wall of code.
        # Hard-split proportionally by character count.
        span = p_end - p_start
        tokens = max(1, embedder.count_tokens(content[p_start:p_end]))
        parts = -(-tokens // budget)  # ceil division
        step = max(1, span // parts)
        for cut in range(p_start, p_end, step):
            out.append((cut, min(cut + step, p_end)))
    return out


def chunk_document(
    content: str,
    embedder: EmbeddingProvider,
    *,
    target_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
) -> list[Chunk]:
    """Chunk normalised document text into token-budgeted, structure-aligned spans."""
    if not content.strip():
        return []

    # Never let the configured target exceed what the model can actually encode.
    budget = min(target_tokens, embedder.max_tokens)
    overlap = min(overlap_tokens, budget - 1) if budget > 1 else 0

    raw_blocks = _split_into_blocks(content)
    blocks: list[_Block] = []
    for start, end, heading_path, is_heading in raw_blocks:
        span_tokens = embedder.count_tokens(content[start:end])
        if span_tokens <= budget:
            blocks.append(_Block(start, end, heading_path, span_tokens, is_heading))
            continue
        for s, e in _split_oversized(content, start, end, budget, embedder):
            blocks.append(
                _Block(s, e, heading_path, embedder.count_tokens(content[s:e]), is_heading)
            )

    chunks: list[Chunk] = []
    current: list[_Block] = []
    current_tokens = 0

    def build(blocks_in: list[_Block]) -> Chunk:
        start = blocks_in[0].start
        end = blocks_in[-1].end
        # Label with the last heading the chunk contains, not the first block's
        # path. A chunk that opens with "# Title" then "## Section" belongs to
        # the section, not the title. Chunks with no heading inherit the context
        # their first block was already in.
        heading_path = next(
            (b.heading_path for b in reversed(blocks_in) if b.is_heading),
            blocks_in[0].heading_path,
        )
        return Chunk(
            ordinal=len(chunks),
            # Slice rather than join the blocks: this is what makes the offsets
            # truthful, and citation resolution depends on it.
            text=content[start:end],
            token_count=sum(b.tokens for b in blocks_in),
            char_start=start,
            char_end=end,
            heading_path=list(heading_path),
        )

    for block in blocks:
        # A heading starts a new section, so it starts a new chunk. Without this
        # a chunk can span a section boundary and then be labelled with only the
        # first section's heading path -- a citation that points at the wrong
        # section is worse than no heading at all.
        #
        # Exception: a run of consecutive headings ("# Title" immediately
        # followed by "## Section") stays together, so a title is not stranded
        # in a chunk of its own.
        section_break = block.is_heading and any(not b.is_heading for b in current)
        over_budget = bool(current) and current_tokens + block.tokens > budget

        if section_break or over_budget:
            chunks.append(build(current))
            if section_break:
                # Do not carry overlap across a section boundary: the trailing
                # text of the previous section is not context for this one.
                current, current_tokens = [], 0
            else:
                # Carry trailing blocks forward as overlap so a fact straddling
                # a boundary is fully present in at least one chunk.
                carry: list[_Block] = []
                carried = 0
                for prev in reversed(current):
                    if carried + prev.tokens > overlap:
                        break
                    carry.insert(0, prev)
                    carried += prev.tokens
                # Never carry the entire chunk forward -- that would not terminate.
                if len(carry) == len(current):
                    carry = carry[1:]
                    carried = sum(b.tokens for b in carry)
                current, current_tokens = carry, carried
        current.append(block)
        current_tokens += block.tokens

    if current:
        chunks.append(build(current))

    # Fold undersized chunks backwards rather than dropping them. Dropping is
    # the tempting option and it is wrong: a short section ("## Contact" plus an
    # email address) is small but is exactly the kind of thing a user asks about,
    # and a dropped chunk is unretrievable content with no signal that it is
    # missing. Merging only happens within a section and only where the spans
    # touch, so offsets stay contiguous and heading paths stay truthful; a short
    # section that cannot merge is kept as a small chunk.
    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and chunk.token_count < min_tokens
            and chunk.heading_path == merged[-1].heading_path
            and chunk.char_start <= merged[-1].char_end
        ):
            prev = merged[-1]
            prev.char_end = chunk.char_end
            prev.text = content[prev.char_start : prev.char_end]
            prev.token_count = embedder.count_tokens(prev.text)
            continue
        merged.append(chunk)

    for i, c in enumerate(merged):
        c.ordinal = i

    return merged
