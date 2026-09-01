"""Grounded answering.

The generation call is the least trustworthy step in the system, so its output
is treated as a proposal to be checked rather than as the answer. Three checks
run on every response:

  1. **Citation resolution.** A cited id must be one of the ids that were
     actually supplied for this question. The model cannot invent a source,
     because the ids are server-generated per request and validated against that
     exact set.

  2. **Quote verification.** The quote must appear verbatim in the chunk it
     cites. A citation that resolves but whose quote does not match is kept and
     flagged rather than dropped -- it is usually paraphrase, occasionally
     fabrication, and both are worth being able to see and count.

  3. **Refusal downgrade.** If the model claims sufficient evidence but produces
     no citation that resolves, the answer is converted into a refusal. An
     uncited answer from a retrieval system is indistinguishable from a guess,
     so it is not served as though it were grounded.

The empty-retrieval case never reaches the model at all: with no evidence there
is nothing to be faithful to, so refusing directly is both more reliable and one
fewer API call against a rate-limited free tier.
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from atlas.answer.prompts import SYSTEM_INSTRUCTION, AnswerOut, build_prompt
from atlas.config import Settings
from atlas.core.models import Answer, Citation, RetrievedChunk, TokenUsage
from atlas.db import repository as repo
from atlas.db.pool import Database
from atlas.ingest.parsers import page_for_offset
from atlas.providers.base import LLMError, LLMProvider
from atlas.retrieval.service import Retriever

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")

NO_EVIDENCE_MESSAGE = (
    "I could not find anything in the knowledge base that answers this question."
)
UNCITED_MESSAGE = (
    "I found related material but could not ground an answer in it, so I am not "
    "answering rather than guessing."
)


def _canonical(text: str) -> str:
    """Whitespace- and case-insensitive form for quote matching.

    Models reflow whitespace when quoting, and chunk text carries the source's
    line breaks. Comparing raw strings would report almost every correct quote
    as unverified.
    """
    return _WHITESPACE.sub(" ", text).strip().casefold()


class AnswerService:
    def __init__(
        self,
        db: Database,
        retriever: Retriever,
        llm: LLMProvider,
        settings: Settings,
    ) -> None:
        self._db = db
        self._retriever = retriever
        self._llm = llm
        self._settings = settings

    async def answer(
        self,
        tenant_id: uuid.UUID,
        question: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
        source_ids: list[uuid.UUID] | None = None,
    ) -> Answer:
        started = time.perf_counter()
        retrieval = await self._retriever.retrieve(
            tenant_id,
            question,
            top_k=top_k,
            min_similarity=min_similarity,
            source_ids=source_ids,
        )
        timings = dict(retrieval.timings_ms)

        if not retrieval.chunks:
            best = retrieval.candidates[0].score if retrieval.candidates else None
            reason = (
                "no_candidates"
                if not retrieval.candidates
                else f"below_similarity_floor (best={best:.3f})"
            )
            timings["total_ms"] = (time.perf_counter() - started) * 1000
            return Answer(
                text=NO_EVIDENCE_MESSAGE,
                citations=[],
                refused=True,
                refusal_reason=reason,
                retrieved=[],
                usage=TokenUsage(),
                timings_ms=timings,
            )

        prompt = build_prompt(question, retrieval.chunks)

        t0 = time.perf_counter()
        try:
            parsed, usage = await self._generate(prompt)
        except LLMError:
            logger.exception("generation failed")
            raise
        timings["llm_ms"] = (time.perf_counter() - t0) * 1000

        citations, unresolved = await self._validate_citations(
            tenant_id, parsed.citations, retrieval.chunks
        )

        refused = not parsed.sufficient_evidence
        refusal_reason = "model_reported_insufficient_evidence" if refused else None
        text = parsed.answer

        if not refused and not citations:
            # The model asserted it had grounds but named no resolvable source.
            refused = True
            refusal_reason = "no_resolvable_citations"
            text = UNCITED_MESSAGE

        if unresolved:
            logger.warning(
                "discarded %s citation(s) naming ids not supplied for this question: %s",
                len(unresolved),
                unresolved,
            )

        timings["total_ms"] = (time.perf_counter() - started) * 1000
        return Answer(
            text=text,
            citations=citations,
            refused=refused,
            refusal_reason=refusal_reason,
            retrieved=retrieval.chunks,
            usage=usage,
            timings_ms=timings,
        )

    async def _generate(self, prompt: str) -> tuple[AnswerOut, TokenUsage]:
        import asyncio

        # Provider SDKs here are synchronous; keep them off the event loop.
        return await asyncio.to_thread(
            lambda: self._llm.generate_structured(
                system_instruction=SYSTEM_INSTRUCTION,
                prompt=prompt,
                response_schema=AnswerOut,
                timeout_seconds=self._settings.llm_timeout_seconds,
            )
        )

    async def _validate_citations(
        self,
        tenant_id: uuid.UUID,
        proposed: list,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[Citation], list[str]]:
        by_id = {str(c.chunk_id): c for c in chunks}
        citations: list[Citation] = []
        unresolved: list[str] = []
        seen: set[str] = set()

        # Page resolution needs the owning documents' metadata; fetch once for
        # the documents actually cited rather than per citation.
        page_offsets: dict[uuid.UUID, list[list[int]] | None] = {}

        for item in proposed:
            raw_id = str(getattr(item, "chunk_id", "")).strip()
            chunk = by_id.get(raw_id)
            if chunk is None:
                unresolved.append(raw_id[:64])
                continue
            if raw_id in seen:
                continue
            seen.add(raw_id)

            quote = str(getattr(item, "quote", "") or "").strip()
            verified = bool(quote) and _canonical(quote) in _canonical(chunk.text)

            if chunk.document_id not in page_offsets:
                page_offsets[chunk.document_id] = await self._page_offsets(
                    tenant_id, chunk.document_id
                )

            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_external_id=chunk.document_external_id,
                    document_title=chunk.document_title,
                    document_uri=chunk.document_uri,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    # Fall back to the chunk text when the model returned no
                    # quote, so a citation always shows the reader something.
                    quote=quote or chunk.text[:300],
                    quote_verified=verified,
                    page=page_for_offset(page_offsets[chunk.document_id], chunk.char_start),
                )
            )

        return citations, unresolved

    async def _page_offsets(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID
    ) -> list[list[int]] | None:
        async with self._db.connection() as conn:
            document = await repo.get_document(conn, tenant_id, document_id)
        if not document:
            return None
        metadata = document.get("metadata") or {}
        offsets = metadata.get("page_offsets")
        return offsets if isinstance(offsets, list) else None
