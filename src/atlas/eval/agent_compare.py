"""Paired evaluation: plain RAG vs agent mode, on the same questions.

ADR-0030 recorded a live comparison that went against the agent, and ADR-0031's
diagnostic (7 questions, hand-picked) suggested the union rerank fixes the
ordering failure it found. Neither is a measurement across the labelled eval
set. This is that measurement.

## Why this is not `EvalRunner` with a flag

`EvalRunner` (`atlas.eval.runner`) scores one system: one retrieval config, one
answer call per query. Comparing two systems that must run on *identical
questions* and be scored *pairwise* is different work -- each query produces two
answers, and the comparison that matters is the per-query delta, not two
separate summaries read side by side. Retrofitting that shape into `EvalRunner`
would touch code that Phase 2's frozen baselines depend on, for a comparison
Phase 2 was never asked to make. This module is separate and `EvalRunner` is
untouched.

## What "answer quality" means here

Citation *presence* is already measured (`EvalRunner`'s `citations` block): did
a non-refused answer cite anything. That says nothing about whether it cited the
*right* thing, which is what a quality comparison needs.

So this reuses the dataset's own label definition -- `Label.matches(document,
chunk_text)`, unchanged from `atlas.eval.dataset` -- applied to the chunks the
answer actually **cited**, not the chunks that were merely retrieved. A label is
satisfied only if some cited chunk carries it. That is a stricter, more honest
notion of quality than "produced a citation", and it costs nothing new: citation
resolution already guarantees every `Citation.chunk_id` names a chunk present in
`Answer.retrieved`, so the lookup is exact, not approximate.

Refusal correctness, unverified-quote counting and the paired bootstrap are the
same definitions and the same function (`metrics.paired_bootstrap_delta`)
already used for the retrieval comparisons this project trusts.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from atlas.agent.service import AgentAnswerService
from atlas.answer.service import AnswerService
from atlas.config import Settings
from atlas.core.models import Answer
from atlas.eval import metrics
from atlas.eval.dataset import EvalQuery, Label, load

# Bounds an agent run is considered to have hit, worth surfacing separately from
# an ordinary "finished" stop -- they mean the loop was cut off, not that the
# model decided it was done.
_BOUND_STOP_REASONS = {"max_iterations", "max_tool_calls", "budget_exhausted"}


@dataclass(slots=True)
class SystemOutcome:
    """One system's answer to one question, scored against its labels."""

    refused: bool
    refusal_reason: str | None
    citation_count: int
    unverified_quotes: int
    matched_labels: int
    total_labels: int
    citation_recall: float | None  # None when the query has no labels to score
    input_tokens: int
    output_tokens: int
    latency_ms: float
    error: str | None = None
    # Populated for the agent system only.
    tool_calls: int | None = None
    iterations: int | None = None
    stop_reason: str | None = None
    degraded: bool | None = None
    searches: list[str] | None = None
    evidence_unique_before_rerank: int | None = None
    evidence_final_count: int | None = None
    agent_prompt_tokens: int | None = None
    agent_output_tokens: int | None = None
    answer_prompt_tokens: int | None = None
    answer_output_tokens: int | None = None


@dataclass(slots=True)
class PairedReport:
    query_id: str
    question: str
    answerable: bool
    kind: str
    plain: SystemOutcome
    agent: SystemOutcome


def citation_recall(
    labels: list[Label], citations: list, retrieved: list
) -> tuple[int, int, float | None]:
    """Fraction of labels satisfied by a chunk the answer actually cited.

    Distinct from retrieval recall, which asks whether a satisfying chunk was
    anywhere in the candidate list. This asks whether the model used it: a
    citation naming a chunk that does not satisfy a label does not count, even
    if some other retrieved-but-uncited chunk would have.

    Returns (matched, total, recall) with recall None when there are no labels
    to score, mirroring how `EvalRunner` leaves unanswerable queries unscored
    rather than reporting a 0 that would be misread as a failure.
    """
    if not labels:
        return 0, 0, None

    by_id = {chunk.chunk_id: chunk for chunk in retrieved}
    cited_chunks = [by_id[c.chunk_id] for c in citations if c.chunk_id in by_id]

    matched = sum(
        1
        for label in labels
        if any(label.matches(chunk.document_external_id, chunk.text) for chunk in cited_chunks)
    )
    return matched, len(labels), matched / len(labels)


async def _score_plain(
    answerer: AnswerService, tenant_id: uuid.UUID, query: EvalQuery
) -> SystemOutcome:
    t0 = time.perf_counter()
    try:
        answer = await answerer.answer(tenant_id, query.question)
    except Exception as exc:  # noqa: BLE001 - one query's failure must not sink the run
        return SystemOutcome(
            refused=True,
            refusal_reason=None,
            citation_count=0,
            unverified_quotes=0,
            matched_labels=0,
            total_labels=len(query.labels),
            citation_recall=None,
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(exc).__name__}: {exc}"[:300],
        )
    return _outcome_from_answer(answer, query, latency_ms=(time.perf_counter() - t0) * 1000)


async def _score_agent(
    agent: AgentAnswerService, context: Any, query: EvalQuery
) -> SystemOutcome:
    t0 = time.perf_counter()
    try:
        answer = await agent.answer(query.question, context)
    except Exception as exc:  # noqa: BLE001
        return SystemOutcome(
            refused=True,
            refusal_reason=None,
            citation_count=0,
            unverified_quotes=0,
            matched_labels=0,
            total_labels=len(query.labels),
            citation_recall=None,
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(exc).__name__}: {exc}"[:300],
        )

    outcome = _outcome_from_answer(answer, query, latency_ms=(time.perf_counter() - t0) * 1000)

    trace = answer.agent_trace or {}
    evidence = trace.get("evidence", {})
    agent_usage = trace.get("usage", {})
    agent_prompt = agent_usage.get("prompt_tokens", 0)
    agent_output = agent_usage.get("output_tokens", 0) + agent_usage.get("thinking_tokens", 0)

    outcome.tool_calls = trace.get("tool_calls")
    outcome.iterations = trace.get("iterations")
    outcome.stop_reason = trace.get("stop_reason")
    outcome.degraded = trace.get("degraded")
    outcome.searches = [
        call["arguments"].get("query")
        for step in trace.get("steps", [])
        for call in step.get("tool_calls", [])
        if call.get("tool") == "search_knowledge_base"
    ]
    outcome.evidence_unique_before_rerank = evidence.get("unique_before_rerank")
    outcome.evidence_final_count = evidence.get("final_count")
    # The combined total (already on Answer.usage, ADR-0029) minus the agent's
    # own portion (carried on the trace before it was combined) leaves the
    # answer model's share. Two prices apply to one request, so the split has
    # to survive for the cost accounting to mean anything.
    outcome.agent_prompt_tokens = agent_prompt
    outcome.agent_output_tokens = agent_output
    outcome.answer_prompt_tokens = max(0, outcome.input_tokens - agent_prompt)
    outcome.answer_output_tokens = max(0, outcome.output_tokens - agent_output)
    return outcome


def _outcome_from_answer(answer: Answer, query: EvalQuery, *, latency_ms: float) -> SystemOutcome:
    matched, total, recall = citation_recall(query.labels, answer.citations, answer.retrieved)
    return SystemOutcome(
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        citation_count=len(answer.citations),
        unverified_quotes=sum(1 for c in answer.citations if not c.quote_verified),
        matched_labels=matched,
        total_labels=total,
        citation_recall=recall,
        input_tokens=answer.usage.prompt_tokens,
        output_tokens=answer.usage.output_tokens + answer.usage.thinking_tokens,
        latency_ms=latency_ms,
    )


async def run_paired(
    tenant_id: uuid.UUID,
    dataset_path: Path,
    answerer: AnswerService,
    agent: AgentAnswerService,
    settings: Settings,
    context_factory,
) -> list[PairedReport]:
    """Run both systems on every query in the dataset, concurrently.

    `context_factory` builds a fresh `ToolContext` per call -- a callable rather
    than a shared instance so nothing here assumes anything about how the
    caller constructs authorization, matching the rule that identity is never
    reused or inferred (ADR-0027).

    Order is preserved (`asyncio.gather` on a list), so a report at index *i* in
    two different runs names the same question -- required for the paired
    bootstrap, which compares by position.
    """
    queries = load(dataset_path)
    limit = asyncio.Semaphore(settings.eval_concurrency)

    async def run_one(query: EvalQuery) -> PairedReport:
        async with limit:
            # Sequential, not gathered: both calls hit the same reranker
            # instance's ONNX session, and comparing "which finished first"
            # would say nothing about the systems being compared.
            plain = await _score_plain(answerer, tenant_id, query)
            agent_outcome = await _score_agent(agent, context_factory(tenant_id), query)
        return PairedReport(
            query_id=query.id,
            question=query.question,
            answerable=query.answerable,
            kind=query.kind,
            plain=plain,
            agent=agent_outcome,
        )

    return list(await asyncio.gather(*(run_one(q) for q in queries)))


def summarize(reports: list[PairedReport]) -> dict[str, Any]:
    """Everything the report needs to defend a recommendation."""
    answerable = [r for r in reports if r.answerable]

    def recalls(system: str, rows: list[PairedReport]) -> list[float]:
        return [
            getattr(r, system).citation_recall
            for r in rows
            if getattr(r, system).citation_recall is not None
        ]

    def block(rows: list[PairedReport]) -> dict[str, Any]:
        plain_recall = recalls("plain", rows)
        agent_recall = recalls("agent", rows)
        out: dict[str, Any] = {"n": len(rows)}
        for name, values in (("plain", plain_recall), ("agent", agent_recall)):
            low, high = metrics.bootstrap_ci(values)
            out[f"{name}_citation_recall"] = {"mean": round(metrics.mean(values), 4),
                                               "ci95": [round(low, 4), round(high, 4)]}
        if len(plain_recall) == len(agent_recall) and plain_recall:
            delta, (low, high) = metrics.paired_bootstrap_delta(plain_recall, agent_recall)
            out["paired_delta_agent_minus_plain"] = {
                "mean": round(delta, 4), "ci95": [round(low, 4), round(high, 4)]
            }
        return out

    summary: dict[str, Any] = {"overall": block(answerable)}

    by_kind: dict[str, list[PairedReport]] = {}
    for r in answerable:
        by_kind.setdefault(r.kind, []).append(r)
    summary["by_kind"] = {kind: block(rows) for kind, rows in sorted(by_kind.items())}

    unanswerable = [r for r in reports if not r.answerable]
    for system in ("plain", "agent"):
        outcomes = [getattr(r, system) for r in unanswerable]
        answerable_outcomes = [getattr(r, system) for r in answerable]
        summary.setdefault("refusal", {})[system] = {
            "unanswerable_queries": len(unanswerable),
            "correctly_refused": sum(1 for o in outcomes if o.refused),
            "answerable_queries": len(answerable_outcomes),
            "incorrectly_refused": sum(1 for o in answerable_outcomes if o.refused),
        }
        summary.setdefault("unverified_quotes", {})[system] = sum(
            getattr(r, system).unverified_quotes for r in reports
        )
        summary.setdefault("errors", {})[system] = sum(
            1 for r in reports if getattr(r, system).error
        )

    agent_outcomes = [r.agent for r in reports]
    tool_calls = [o.tool_calls for o in agent_outcomes if o.tool_calls is not None]
    degraded = [o for o in agent_outcomes if o.degraded]
    bound_hit = [o for o in agent_outcomes if o.stop_reason in _BOUND_STOP_REASONS]
    summary["agent_behaviour"] = {
        "degraded_count": len(degraded),
        "degraded_rate": round(len(degraded) / len(agent_outcomes), 4) if agent_outcomes else 0.0,
        "bound_hit_count": len(bound_hit),
        "tool_calls_mean": round(metrics.mean([float(t) for t in tool_calls]), 2),
        "tool_calls_max": max(tool_calls, default=0),
        "tool_calls_by_kind": {
            kind: round(
                metrics.mean(
                    [float(r.agent.tool_calls) for r in rows if r.agent.tool_calls is not None]
                ),
                2,
            )
            for kind, rows in sorted(by_kind.items())
        },
    }

    def latency_stats(system: str) -> dict[str, float]:
        values = sorted(getattr(r, system).latency_ms for r in reports)
        if not values:
            return {}
        return {
            "p50": round(values[len(values) // 2], 1),
            "p95": round(values[min(len(values) - 1, int(len(values) * 0.95))], 1),
            "max": round(values[-1], 1),
        }

    summary["latency_ms"] = {"plain": latency_stats("plain"), "agent": latency_stats("agent")}

    summary["tokens"] = {
        "plain": {
            "input": sum(r.plain.input_tokens for r in reports),
            "output": sum(r.plain.output_tokens for r in reports),
        },
        "agent": {
            "agent_model_input": sum(r.agent.agent_prompt_tokens or 0 for r in reports),
            "agent_model_output": sum(r.agent.agent_output_tokens or 0 for r in reports),
            "answer_model_input": sum(r.agent.answer_prompt_tokens or 0 for r in reports),
            "answer_model_output": sum(r.agent.answer_output_tokens or 0 for r in reports),
        },
    }

    return summary


def to_json(
    reports: list[PairedReport], summary: dict[str, Any], *, config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "config": config,
        "summary": summary,
        "queries": [
            {
                "query_id": r.query_id,
                "question": r.question,
                "answerable": r.answerable,
                "kind": r.kind,
                "plain": asdict(r.plain),
                "agent": asdict(r.agent),
            }
            for r in reports
        ],
    }
