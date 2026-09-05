"""The `search_knowledge_base` tool.

This is the first real tool, and it is deliberately a thin wrapper over the
retrieval stack Phases 1 and 2 already built and measured. It introduces no new
retrieval behaviour: same modes, same fusion, same reranker, same similarity
gate. Anything the agent turns out to do better, it does by *choosing what to
search for* and *searching more than once* -- not by searching differently.

That constraint is what makes the Phase 4 evaluation readable. If this tool also
changed how retrieval worked, an improvement over plain RAG could not be
attributed to agency.

## The model sees snippets; the answer sees chunks

A tool result is fed back to the model as a function response, so every byte it
returns is spent again on every subsequent turn of the loop. Returning full
chunk text would cost roughly `top_k * iterations` chunks of context, most of it
restating text the server already holds in memory.

So the return value is split (see `ToolOutput`):

* **content** -- what the model gets: an evidence id, the document, the section
  path, the score and a snippet. Enough to judge *coverage* -- "I have the
  refund window but nothing on the approval chain" -- which is the only decision
  the loop actually asks the model to make.
* **artifacts** -- what the server keeps: the full `RetrievedChunk` objects,
  with offsets and provenance intact, for the grounded answering path to cite
  against.

The cost of this split is real and worth stating: the model judges sufficiency
from truncated text, so it can believe a snippet answers a question when the
full chunk would have shown otherwise. The alternative -- full text in every
tool response -- pays that token cost on every turn to remove a judgement error
that only affects when the loop *stops*, not what it cites. Citations are always
resolved against the full chunk regardless.

## Statelessness is load-bearing

The tool holds no per-call state. One instance is registered at startup and
serves every request concurrently, so a ledger stored on the tool would mix
tenants' evidence across requests -- exactly the failure the authorization
boundary exists to prevent, arrived at from the other direction. Accumulating
evidence across iterations is the agent loop's job, from the `artifacts` it
collects.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import Field

from atlas.agent.tools import Tool, ToolArgs, ToolContext, ToolOutput
from atlas.core.models import RetrievedChunk
from atlas.retrieval.service import Retriever

#: How much of a chunk the model sees per hit.
#:
#: Chunks run to roughly 1200 characters. This is enough to tell whether a
#: passage is on-topic without paying for the whole thing on every turn.
SNIPPET_CHARS = 480


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Truncate on a word boundary so the model is not shown a broken token."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[:limit]
    spaced = cut.rsplit(" ", 1)[0]
    # A single very long word would otherwise collapse the whole snippet away.
    return (spaced if len(spaced) > limit // 2 else cut) + "..."


class SearchKnowledgeBaseArgs(ToolArgs):
    """Arguments the model may supply.

    Note what is *absent*: no tenant, no source filter, no similarity floor. The
    tenant is refused at registration (`RESERVED_ARGUMENT_NAMES`); the other two
    are omitted because they are scoring policy, and a model that can lower the
    relevance floor can talk itself into evidence the system already judged too
    weak to answer from.
    """

    query: str = Field(
        min_length=1,
        max_length=512,
        description=(
            "A focused natural-language search phrase describing one thing you "
            "need to find. Search for a single fact or topic at a time rather "
            "than restating the whole question."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="How many passages to return. Use a larger value only for broad questions.",
    )


class SearchKnowledgeBaseTool(Tool):
    name = "search_knowledge_base"

    # This string is prompt text and it is the main lever on routing behaviour,
    # so it says *when* to call the tool, not merely what it does. The
    # instruction to search repeatedly is the whole hypothesis under test: the
    # queries stuck at Recall@1 = 0.400 are ones needing two documents, which a
    # single embedding of the full question does not retrieve.
    description = (
        "Search the knowledge base for passages relevant to a search phrase. "
        "Use this for any question about the documents. Call it more than once, "
        "with different phrasings, when a question has several parts or is "
        "likely answered across multiple documents -- each call is independent "
        "and cheap. Returns snippets; the full text is retained for citation."
    )

    Args = SearchKnowledgeBaseArgs

    # Generous relative to a plain database query: this embeds the query with a
    # local ONNX model, runs a vector search, and may then rerank 30 candidates
    # with a cross-encoder. On cold model load the first call is much slower
    # than the rest, and a timeout there would fail the request rather than
    # merely make it slow.
    timeout_seconds = 30.0

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def execute(self, context: ToolContext, args: SearchKnowledgeBaseArgs) -> ToolOutput:
        result = await self._retriever.retrieve(
            # The one argument that matters, and it comes from the context.
            context.tenant_id,
            args.query,
            top_k=args.top_k,
        )

        chunks: list[RetrievedChunk] = result.chunks
        content: dict[str, Any] = {
            "query": args.query,
            "result_count": len(chunks),
            "results": [_render(chunk) for chunk in chunks],
        }

        if not chunks:
            # An empty result is the model's cue to reformulate, so it is told
            # *why* it was empty. "Nothing matched" and "something matched but
            # was too weak to use" call for different next queries, and the
            # difference is invisible from an empty list alone.
            content["note"] = (
                "No passages passed the relevance threshold."
                if result.best_dense_score is None
                else (
                    f"Nothing matched closely enough (best relevance "
                    f"{result.best_dense_score:.2f}). Try different wording or "
                    "more specific terms from the documents."
                )
            )

        return ToolOutput(content=content, artifacts=chunks)


def _render(chunk: RetrievedChunk) -> dict[str, Any]:
    """One hit as the model sees it."""
    return {
        # Named "evidence_id" rather than "chunk_id" because that is what it is
        # for: the handle the model uses to cite this passage later. The value
        # is server-generated and validated against the set actually supplied,
        # so it cannot name a passage that was never retrieved.
        "evidence_id": str(chunk.chunk_id),
        "document": chunk.document_external_id,
        "title": chunk.document_title,
        # Joined with "/" rather than ">" so a heading containing the separator
        # cannot look like structure. Same reasoning as the evidence blocks.
        "section": "/".join(chunk.heading_path) if chunk.heading_path else None,
        "relevance": round(float(chunk.score), 3),
        # Untrusted document text. It reaches the model as a JSON string value,
        # which contains it syntactically but not semantically: a passage saying
        # "ignore previous instructions" is still read by the model. That risk
        # is unchanged from plain RAG and bounded by the authorization boundary
        # -- it can waste a turn, it cannot change whose data is searched.
        "snippet": _snippet(chunk.text),
    }


def evidence_from(
    results: list[Any], *, limit: int | None = None
) -> list[RetrievedChunk]:
    """Collect chunks from tool results into one ordered evidence set.

    ## Why the order is round-robin and not by score

    The obvious thing is to sort everything by score. It would be wrong. Scores
    from different searches are not comparable: the reranker is a cross-encoder
    whose outputs are unnormalised logits with a per-query scale (see
    `RerankProvider`), so a 4.1 from one search and a 2.8 from another say
    nothing about which passage is better. Sorting them would invent an ordering
    out of noise.

    The obvious fallback -- concatenating the searches -- is also wrong, for a
    different reason. It puts the whole of the first search ahead of the whole
    of the second, so under any truncation a two-part question loses the half it
    searched for last. That is precisely the case the agent exists to serve.

    So results are interleaved by rank: every search's best passage, then every
    search's second, and so on. Each search's own ordering is internally valid
    and is preserved; what is never invented is an ordering *between* them.
    Truncation then costs each search its tail rather than costing one search
    everything.

    Deduplicated by chunk id, because two phrasings of the same question should
    find the same best passage, and the earliest occurrence wins so a chunk
    found by an earlier search keeps its stronger position.
    """
    ranked: list[list[RetrievedChunk]] = [
        list(getattr(result, "artifacts", None) or []) for result in results
    ]
    ranked = [chunks for chunks in ranked if chunks]

    seen: set[uuid.UUID] = set()
    evidence: list[RetrievedChunk] = []

    for rank in range(max((len(chunks) for chunks in ranked), default=0)):
        for chunks in ranked:
            if rank >= len(chunks):
                continue
            chunk = chunks[rank]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            evidence.append(chunk)
            if limit is not None and len(evidence) >= limit:
                return evidence

    return evidence
