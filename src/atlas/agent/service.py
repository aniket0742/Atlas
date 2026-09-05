"""Agentic answering: the loop's evidence, the unchanged grounded path.

This module is deliberately small, and that is the point of it. It runs the
agent loop, caps the evidence, and hands it to `AnswerService.answer_from_evidence`
-- the same function the plain path calls, with the same prompt, the same
server-generated ids, the same citation resolution, the same quote verification
and the same refusal downgrade.

There is no agentic answering *path*. There is one answering path, and two ways
of deciding what evidence reaches it.

## Why that matters more than it looks

Agent frameworks tend to let the loop produce the final text, because the model
is already there and one more turn is free. Doing that here would move answer
generation off the grounded path: no evidence blocks, no id validation, no quote
check. Every groundedness property Phase 1 built would silently not apply to the
new feature -- and it would look like it was working, because the answers would
read fine.

So the agent model never writes the answer, and is never asked for a citation.
Its output is used for exactly one thing: choosing which chunks to retrieve.

## Ordering the union

The agent may search several times, and passages from different searches carry
scores that cannot be compared: the cross-encoder produces unnormalised
per-query logits, so a 4.1 from one search and a 2.8 from another say nothing
about which passage is better (ADR-0020).

`evidence_from` therefore interleaves by rank rather than sorting, which is
honest about that incomparability but leaves the answer model with a set that is
*less* well ordered than the single globally-reranked list the plain path hands
it. The first live comparison suggested that costs answer quality (ADR-0030).

So the union is reranked once, against the **original user question**, before
answering. One cross-encoder pass over the deduplicated union produces a single
comparable ordering by construction rather than by assumption. It ranks against
the question the user actually asked, not the sub-queries the agent invented,
because that is the only question the answer is judged against.

Nothing about retrieval changes and the reranker itself is untouched: this is
the same `RerankProvider.rerank` call the retriever already makes, applied once
more to a set it has never seen as a whole.

`agent_union_rerank` exists so the two behaviours can be compared directly on
the same questions rather than the fix being assumed to work.

## The evidence cap

Eight tool calls of up to ten passages each can produce far more evidence than
is useful. Past some point more evidence makes answers *worse*: the answer model
has more irrelevant material to be distracted by, the prompt costs more, and
long-context attention degrades in the middle.

`agent_max_evidence` caps it, applied **after** the union rerank so the cap
keeps the globally best passages rather than the best of an arbitrary ordering.
It stays a separate variable from the reranking change so the two can be
measured apart.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from atlas.agent.knowledge_base import evidence_from
from atlas.agent.loop import AgentPlan, AgentPlanner
from atlas.agent.tools import ToolContext
from atlas.answer.service import AnswerService
from atlas.config import Settings
from atlas.core.models import Answer, RetrievedChunk, TokenUsage
from atlas.providers.base import RerankProvider

logger = logging.getLogger(__name__)


class AgentAnswerService:
    def __init__(
        self,
        planner: AgentPlanner,
        answerer: AnswerService,
        settings: Settings,
        reranker: RerankProvider | None = None,
    ) -> None:
        self._planner = planner
        self._answerer = answerer
        self._settings = settings
        self._reranker = reranker

    async def answer(self, question: str, context: ToolContext) -> Answer:
        started = time.perf_counter()

        plan = await self._planner.gather(question, context)

        # The full deduplicated union first, uncapped: the cap is applied after
        # ordering so it keeps the globally best passages rather than the best
        # of an ordering that was never meant to be compared across searches.
        union = evidence_from(
            [result for step in plan.steps for result in step.tool_results]
        )
        evidence, rerank_info = await self._order(question, union)

        answer = await self._answerer.answer_from_evidence(
            context.tenant_id,
            question,
            evidence,
            started=started,
            timings={"agent_ms": plan.duration_ms},
            provenance={
                "retrieval_mode": self._settings.retrieval_mode,
                "reranked": self._settings.rerank_enabled,
                # Deliberately None. The similarity gate ran per search inside
                # the tool, so evidence reaching here already passed it; there
                # is no single query whose best dense score would mean anything.
                "best_dense_score": None,
                "per_component": {"agent_searches": plan.tool_call_count},
            },
            no_evidence_reason=f"agent_found_no_evidence ({plan.stop_reason.value})",
        )

        answer.agent_trace = plan.trace()
        answer.agent_trace["evidence"] = {
            "unique_before_rerank": len(union),
            "final_count": len(evidence),
            **rerank_info,
            # The order actually handed to the answer model. Without it a trace
            # cannot explain why one passage was cited and another ignored,
            # which is the question this instrumentation exists to answer.
            "order": [
                {
                    "evidence_id": str(chunk.chunk_id),
                    "document": chunk.document_external_id,
                    "section": "/".join(chunk.heading_path) or None,
                    "score": round(float(chunk.score), 4),
                }
                for chunk in evidence
            ],
        }
        # Two models ran, so the cost of an agent answer is both. Keeping one
        # would understate it by however many searches the agent chose to make.
        answer.usage = _combine(plan.usage, answer.usage)
        answer.timings_ms["total_ms"] = (time.perf_counter() - started) * 1000

        logger.info(
            "agent answer refused=%s citations=%s evidence=%s degraded=%s tenant=%s",
            answer.refused,
            len(answer.citations),
            len(evidence),
            plan.degraded,
            context.tenant_id,
        )
        return answer

    async def _order(
        self, question: str, union: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        """Rank the union against the original question, then cap it.

        Returns the evidence and the numbers the trace needs to explain it.

        When reranking is off, unavailable, or there is nothing to reorder, the
        interleaved order from `evidence_from` stands. That is the pre-fix
        behaviour, kept reachable so the two can be compared on identical
        questions.
        """
        cap = self._settings.agent_max_evidence
        info: dict[str, Any] = {
            "reranked": False,
            "rerank_count": 0,
            "rerank_ms": 0.0,
            "reranker": None,
        }

        if not (self._settings.agent_union_rerank and self._reranker and len(union) > 1):
            return union[:cap], info

        t0 = time.perf_counter()
        # Synchronous ONNX inference; keep it off the event loop.
        scores = await asyncio.to_thread(
            self._reranker.rerank, question, [chunk.text for chunk in union]
        )
        info["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        for chunk, score in zip(union, scores, strict=True):
            # Recorded under its own name. The per-search score stays in
            # `component_scores`, so nothing about how the chunk was found is
            # lost. Only the ordering changes; identity, offsets and text --
            # everything a citation resolves against -- are untouched.
            chunk.component_scores["union_rerank"] = float(score)
            chunk.score = float(score)

        # Tie-break on chunk id so an ordering is never left to the order two
        # concurrent searches happened to finish in.
        ordered = sorted(union, key=lambda c: (-c.score, str(c.chunk_id)))

        info["reranked"] = True
        info["rerank_count"] = len(union)
        info["reranker"] = getattr(self._reranker, "model_id", None)
        return ordered[:cap], info

    async def plan_only(self, question: str, context: ToolContext) -> AgentPlan:
        """Gather evidence without answering.

        Used by the evaluation harness, which measures retrieval and routing
        separately from answer quality and should not pay for a generation call
        it will not score.
        """
        return await self._planner.gather(question, context)


def _combine(agent: TokenUsage, answer: TokenUsage) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=agent.prompt_tokens + answer.prompt_tokens,
        output_tokens=agent.output_tokens + answer.output_tokens,
        thinking_tokens=agent.thinking_tokens + answer.thinking_tokens,
        total_tokens=agent.total_tokens + answer.total_tokens,
    )


__all__ = ["AgentAnswerService"]
