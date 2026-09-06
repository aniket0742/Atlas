"""The global union rerank.

The agent may search several times. Scores from separate searches are not
comparable -- the cross-encoder emits unnormalised per-query logits -- so before
this the answer model received a set ordered by a rule that deliberately says
nothing about cross-search quality. These tests pin the fix: one rerank pass
over the deduplicated union, against the original question, before answering.

What must survive it: deduplication, provenance, citation resolution, quote
verification against full server-side text, and the plain path being untouched.
"""

from __future__ import annotations

import asyncio
import uuid

from atlas.agent.knowledge_base import SearchKnowledgeBaseTool
from atlas.agent.loop import AgentPlanner
from atlas.agent.service import AgentAnswerService
from atlas.agent.tools import ToolContext, ToolRegistry
from atlas.answer.service import AnswerService
from atlas.config import Settings
from atlas.providers.fake import FakeLLMProvider, ScriptedToolCallingLLM
from tests.conftest import StubDatabase, make_chunk
from tests.test_search_tool import FakeRetriever

TENANT = uuid.uuid5(uuid.NAMESPACE_DNS, "caller-tenant")
QUESTION = "What is the refund window and who approves exceptions?"


class RecordingReranker:
    """Scores by a table the test supplies, and records how it was called.

    Deterministic on purpose: a real cross-encoder would make these tests
    assertions about a model rather than about the wiring.
    """

    model_id = "fake-cross-encoder"

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self._scores = scores or {}
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return [self._scores.get(text, 0.0) for text in passages]


def settings(**overrides) -> Settings:
    return Settings(gemini_api_key="test-key", **overrides)


def build(script, *, searches, reranker=None, answer_llm=None, **overrides):
    """`searches` is a list of chunk-lists, one per tool call in order."""
    config = settings(**overrides)
    db = StubDatabase([{"metadata": {}}])

    class SequencedRetriever(FakeRetriever):
        def __init__(self) -> None:
            super().__init__([])
            self._queue = list(searches)

        async def retrieve(self, tenant_id, query, *, top_k=None, **kwargs):
            self._chunks = self._queue.pop(0) if self._queue else []
            return await super().retrieve(tenant_id, query, top_k=top_k, **kwargs)

    retriever = SequencedRetriever()
    registry = ToolRegistry([SearchKnowledgeBaseTool(retriever)])  # type: ignore[arg-type]
    llm = answer_llm or FakeLLMProvider()
    answerer = AnswerService(db, retriever, llm, config)  # type: ignore[arg-type]
    planner = AgentPlanner(ScriptedToolCallingLLM(script), registry, config)
    service = AgentAnswerService(planner, answerer, config, reranker=reranker)
    return service, llm, retriever


def context(tenant_id=TENANT) -> ToolContext:
    return ToolContext(tenant_id=tenant_id, request_id="req-1")


def search(query: str):
    return ("search_knowledge_base", {"query": query})


# ---------------------------------------------------------------------------
# Union and deduplication
# ---------------------------------------------------------------------------


async def test_overlapping_searches_produce_one_deduplicated_union():
    """Two phrasings finding the same passage must not present it twice."""
    shared = make_chunk("shared passage")
    only_a = make_chunk("only in the first search")
    only_b = make_chunk("only in the second search")
    reranker = RecordingReranker()

    service, _, _ = build(
        [[search("a")], [search("b")], "done"],
        searches=[[shared, only_a], [shared, only_b]],
        reranker=reranker,
    )

    answer = await service.answer(QUESTION, context())

    texts = [c.text for c in answer.retrieved]
    assert sorted(texts) == sorted(
        ["shared passage", "only in the first search", "only in the second search"]
    )
    assert len(texts) == len(set(texts)), "a chunk reached the answer twice"
    assert answer.agent_trace["evidence"]["unique_before_rerank"] == 3


async def test_the_reranker_sees_the_union_exactly_once():
    """Not once per search: the point is a single comparable ordering."""
    reranker = RecordingReranker()
    service, _, _ = build(
        [[search("a")], [search("b")], "done"],
        searches=[[make_chunk("a1"), make_chunk("a2")], [make_chunk("b1")]],
        reranker=reranker,
    )

    await service.answer(QUESTION, context())

    assert len(reranker.calls) == 1
    _, passages = reranker.calls[0]
    assert sorted(passages) == ["a1", "a2", "b1"]


async def test_the_rerank_uses_the_original_question_not_a_sub_query():
    """The user's question is the only one the answer is judged against."""
    reranker = RecordingReranker()
    service, _, _ = build(
        [[search("refund window policy")], [search("who approves exceptions")], "done"],
        searches=[[make_chunk("a")], [make_chunk("b")]],
        reranker=reranker,
    )

    await service.answer(QUESTION, context())

    query, _ = reranker.calls[0]
    assert query == QUESTION
    assert query != "refund window policy"


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


async def test_evidence_reaches_the_answer_in_globally_reranked_order():
    """The ordering the answer model sees is the reranker's, end to end."""
    low, mid, high = make_chunk("low"), make_chunk("mid"), make_chunk("high")
    reranker = RecordingReranker({"low": -2.0, "mid": 1.0, "high": 9.0})

    service, answer_llm, _ = build(
        [[search("a")], [search("b")], "done"],
        searches=[[low, mid], [high]],
        reranker=reranker,
    )

    answer = await service.answer(QUESTION, context())

    assert [c.text for c in answer.retrieved] == ["high", "mid", "low"]
    prompt = answer_llm.calls[0]["prompt"]
    assert prompt.index("high") < prompt.index("mid") < prompt.index("low")


async def test_a_passage_from_the_last_search_can_outrank_the_first():
    """The failure mode this fixes: interleaving pinned rank 1 to search one."""
    weak_first = make_chunk("weak but found first")
    strong_last = make_chunk("strong but found last")
    reranker = RecordingReranker({"weak but found first": 0.1, "strong but found last": 8.0})

    service, _, _ = build(
        [[search("a")], [search("b")], "done"],
        searches=[[weak_first], [strong_last]],
        reranker=reranker,
    )

    answer = await service.answer(QUESTION, context())

    assert answer.retrieved[0].text == "strong but found last"


async def test_the_cap_is_applied_after_reranking_not_before():
    """Otherwise the cap keeps the best of an ordering that means nothing."""
    chunks = [make_chunk(f"c{i}") for i in range(6)]
    # The last chunk found is the best one; a pre-rerank cap would drop it.
    reranker = RecordingReranker({f"c{i}": float(i) for i in range(6)})

    service, _, _ = build(
        [[search("a")], [search("b")], "done"],
        searches=[chunks[:3], chunks[3:]],
        reranker=reranker,
        agent_max_evidence=2,
    )

    answer = await service.answer(QUESTION, context())

    assert [c.text for c in answer.retrieved] == ["c5", "c4"]
    trace = answer.agent_trace["evidence"]
    assert trace["unique_before_rerank"] == 6
    assert trace["rerank_count"] == 6, "the reranker saw only the capped set"
    assert trace["final_count"] == 2


async def test_ties_are_broken_deterministically():
    """Concurrent searches finish in arbitrary order; the ranking must not."""
    a, b = make_chunk("a"), make_chunk("b")
    reranker = RecordingReranker({"a": 5.0, "b": 5.0})

    service, _, _ = build(
        [[search("q")], "done"], searches=[[a, b]], reranker=reranker
    )
    first = await service.answer(QUESTION, context())

    service2, _, _ = build(
        [[search("q")], "done"], searches=[[b, a]], reranker=reranker
    )
    second = await service2.answer(QUESTION, context())

    assert [c.chunk_id for c in first.retrieved] == [c.chunk_id for c in second.retrieved]


# ---------------------------------------------------------------------------
# Provenance survives the rerank
# ---------------------------------------------------------------------------


async def test_reranking_preserves_identity_offsets_and_text():
    """Everything a citation resolves against must be untouched."""
    chunk = make_chunk("Refunds are issued within 30 days of purchase.")
    original = (chunk.chunk_id, chunk.document_id, chunk.char_start, chunk.char_end, chunk.text)
    reranker = RecordingReranker({chunk.text: 3.0})

    service, _, _ = build([[search("q")], "done"], searches=[[chunk]], reranker=reranker)
    answer = await service.answer(QUESTION, context())

    kept = answer.retrieved[0]
    assert (kept.chunk_id, kept.document_id, kept.char_start, kept.char_end, kept.text) == original


async def test_the_first_stage_score_is_not_lost():
    """Only the ordering changes; how a chunk was found stays on the record."""
    chunk = make_chunk("passage", score=0.77)
    other = make_chunk("other", score=0.5)
    reranker = RecordingReranker({"passage": 4.0, "other": 1.0})

    service, _, _ = build(
        [[search("q")], "done"], searches=[[chunk, other]], reranker=reranker
    )
    answer = await service.answer(QUESTION, context())

    scores = answer.retrieved[0].component_scores
    assert scores["dense"] == 0.77
    assert scores["union_rerank"] == 4.0


async def test_citations_still_resolve_after_reranking():
    chunk = make_chunk("Refunds are issued within 30 days of purchase.")
    reranker = RecordingReranker({chunk.text: 2.0})

    service, _, _ = build([[search("q")], "done"], searches=[[chunk]], reranker=reranker)
    answer = await service.answer(QUESTION, context())

    assert not answer.refused
    assert answer.citations[0].chunk_id == chunk.chunk_id


async def test_quote_verification_still_uses_full_chunk_text():
    """Not the snippet the agent model saw, which is truncated at 480 chars."""
    body = "Filler. " * 200 + "The approval must be recorded in writing."
    chunk = make_chunk(body)
    reranker = RecordingReranker({body: 1.0})

    class QuotesPastTheSnippet:
        model_id = "quotes-late"

        def generate_structured(self, *, system_instruction, prompt, response_schema, **kwargs):
            from atlas.core.models import TokenUsage

            return (
                response_schema(
                    answer="It must be in writing.",
                    citations=[
                        {
                            "chunk_id": str(chunk.chunk_id),
                            "quote": "The approval must be recorded in writing.",
                        }
                    ],
                    sufficient_evidence=True,
                ),
                TokenUsage(),
            )

    service, _, _ = build(
        [[search("q")], "done"],
        searches=[[chunk]],
        reranker=reranker,
        answer_llm=QuotesPastTheSnippet(),
    )
    answer = await service.answer(QUESTION, context())

    assert answer.citations[0].quote_verified, "verification saw only the snippet"


# ---------------------------------------------------------------------------
# The switch, and what it must not touch
# ---------------------------------------------------------------------------


async def test_the_rerank_can_be_turned_off_for_comparison():
    low, high = make_chunk("low"), make_chunk("high")
    reranker = RecordingReranker({"low": 0.0, "high": 9.0})

    service, _, _ = build(
        [[search("a")], [search("b")], "done"],
        searches=[[low], [high]],
        reranker=reranker,
        agent_union_rerank=False,
    )

    answer = await service.answer(QUESTION, context())

    # Interleaved order, unchanged from before the fix.
    assert [c.text for c in answer.retrieved] == ["low", "high"]
    assert reranker.calls == []
    assert answer.agent_trace["evidence"]["reranked"] is False


async def test_no_reranker_configured_degrades_to_the_interleave():
    service, _, _ = build(
        [[search("a")], "done"], searches=[[make_chunk("x")]], reranker=None
    )

    answer = await service.answer(QUESTION, context())

    assert answer.retrieved
    assert answer.agent_trace["evidence"]["reranked"] is False


async def test_a_single_passage_is_not_sent_to_the_reranker():
    """Nothing to reorder, and a cross-encoder pass is not free."""
    reranker = RecordingReranker({"only": 1.0})
    service, _, _ = build(
        [[search("a")], "done"], searches=[[make_chunk("only")]], reranker=reranker
    )

    await service.answer(QUESTION, context())

    assert reranker.calls == []


async def test_the_plain_path_does_no_union_rerank():
    """agent=false must be byte-for-byte the behaviour it always had."""
    config = settings()
    chunk = make_chunk("Refunds are issued within 30 days.")
    retriever = FakeRetriever([chunk])
    answerer = AnswerService(
        StubDatabase([{"metadata": {}}]), retriever, FakeLLMProvider(), config  # type: ignore[arg-type]
    )

    answer = await answerer.answer(TENANT, QUESTION)

    assert answer.agent_trace is None
    assert "union_rerank" not in answer.retrieved[0].component_scores
    assert answer.citations


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


async def test_the_trace_explains_the_final_ordering():
    a, b = make_chunk("alpha"), make_chunk("bravo")
    reranker = RecordingReranker({"alpha": 1.0, "bravo": 7.0})

    service, _, _ = build(
        [[search("one")], [search("two")], "done"],
        searches=[[a], [b]],
        reranker=reranker,
    )

    trace = (await service.answer(QUESTION, context())).agent_trace

    assert trace["tool_calls"] == 2
    evidence = trace["evidence"]
    assert evidence["unique_before_rerank"] == 2
    assert evidence["rerank_count"] == 2
    assert evidence["final_count"] == 2
    assert evidence["reranked"] is True
    assert evidence["reranker"] == "fake-cross-encoder"
    assert evidence["rerank_ms"] >= 0
    assert [entry["evidence_id"] for entry in evidence["order"]] == [
        str(b.chunk_id),
        str(a.chunk_id),
    ]
    assert evidence["order"][0]["score"] == 7.0


async def test_the_trace_stays_json_serialisable():
    import json

    reranker = RecordingReranker({"x": 1.0})
    service, _, _ = build(
        [[search("a")], "done"], searches=[[make_chunk("x"), make_chunk("y")]], reranker=reranker
    )

    json.dumps((await service.answer(QUESTION, context())).agent_trace)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


async def test_concurrent_requests_do_not_share_evidence():
    """The service and reranker are shared; the evidence must not be."""
    other = uuid.uuid5(uuid.NAMESPACE_DNS, "other-tenant")
    mine, theirs = make_chunk("mine"), make_chunk("theirs")
    reranker = RecordingReranker({"mine": 1.0, "theirs": 1.0})

    class PerTenantRetriever:
        async def retrieve(self, tenant_id, query, *, top_k=None, **kwargs):
            from atlas.retrieval.service import RetrievalResult

            await asyncio.sleep(0.01)
            chunks = [mine] if tenant_id == TENANT else [theirs]
            return RetrievalResult(
                chunks=chunks, candidates=chunks, timings_ms={}, mode="dense",
                best_dense_score=0.8,
            )

    class SearchOnceThenFinish:
        model_id = "stateless-fake"

        def generate_with_tools(self, *, system_instruction, history, tools, timeout_seconds=None):
            from atlas.core.models import TokenUsage
            from atlas.providers.base import AgentTurn, ToolCall

            if len(history) == 1:
                return AgentTurn(
                    text=None,
                    tool_calls=(ToolCall(name="search_knowledge_base", arguments={"query": "x"}),),
                    usage=TokenUsage(),
                )
            return AgentTurn(text="done", tool_calls=(), usage=TokenUsage())

    config = settings()
    registry = ToolRegistry([SearchKnowledgeBaseTool(PerTenantRetriever())])  # type: ignore[arg-type]
    answerer = AnswerService(
        StubDatabase([{"metadata": {}}]), PerTenantRetriever(), FakeLLMProvider(), config  # type: ignore[arg-type]
    )
    service = AgentAnswerService(
        AgentPlanner(SearchOnceThenFinish(), registry, config),  # type: ignore[arg-type]
        answerer,
        config,
        reranker=reranker,
    )

    a, b = await asyncio.gather(
        service.answer(QUESTION, context(TENANT)),
        service.answer(QUESTION, context(other)),
    )

    assert [c.text for c in a.retrieved] == ["mine"]
    assert [c.text for c in b.retrieved] == ["theirs"]
    assert a.citations[0].chunk_id == mine.chunk_id
    assert b.citations[0].chunk_id == theirs.chunk_id
